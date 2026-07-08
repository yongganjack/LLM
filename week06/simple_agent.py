"""
Week06 Simple Agent — 基于 Week05 聊天机器人扩展的 LangChain Agent

架构层次：
  底层 → Week04 qa_assistant_lc.get_model()       （LLM 实例，与 Week05 相同）
  中层 → ChatPromptTemplate + MessagesPlaceholder  （Prompt 模式，与 Week05 相同）
  上层 → create_tool_calling_agent + AgentExecutor （LangChain Agent 框架，Week06 新增）

Agent 工作流（ReAct 循环）：
  1. 接收用户输入
  2. LangChain Agent 自主判断是否需要调用工具
  3. 需要时调用 read_local_knowledge 读取本地资料
  4. Agent 根据工具返回结果组织最终回答
  5. 按固定三段格式（结论 / 依据 / 是否调用工具）输出

使用方式:
  # 单次问答
  python week06/simple_agent.py "请根据本地资料说明 Agent 的工作流程"

  # 交互模式
  python week06/simple_agent.py
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Iterator

# ── Windows 控制台 UTF-8 编码修复 ──────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

# 延迟导入的默认值（与 Week04/Week05 保持一致）
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_MAX_TOKENS = 1024
_DEFAULT_PROVIDER = "cloud"
_DEFAULT_DEVICE = "gpu"
_DEFAULT_STREAM = True

_LAST_TOKEN_USAGE: Dict[str, Any] = {
    "prompt": 0,
    "completion": 0,
    "total": 0,
    "source": "none",
}

# 项目根路径（供延迟导入 week04 使用，与 Week05 相同的做法）
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ═══════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════

# 工具关键词（供 should_call_tool 本地规则判断 + Agent 系统提示词参考）
TOOL_KEYWORDS: List[str] = [
    "资料", "知识库", "文档", "根据本地", "本地资料",
    "week06", "Agent", "工具", "流程", "依据",
]

_KNOWLEDGE_FILENAME = "local_knowledge.md"

_SECTION_CONCLUSION = "结论："
_SECTION_EVIDENCE = "依据："
_SECTION_TOOL_STATUS = "是否调用工具："

_VALID_TOOL_STATUSES = frozenset({"是", "否", "是，但失败"})


# ═══════════════════════════════════════════════════════════
#  工具路径解析
# ═══════════════════════════════════════════════════════════

def _resolve_knowledge_path(path: Optional[str] = None) -> Path:
    """解析 local_knowledge.md 的绝对路径。"""
    if path:
        return Path(path).resolve()
    return (Path(__file__).parent / _KNOWLEDGE_FILENAME).resolve()


# ═══════════════════════════════════════════════════════════
#  底层文件读取（不依赖 LangChain，可独立测试和调用）
# ═══════════════════════════════════════════════════════════

def _read_knowledge_file(path: Optional[str] = None) -> Dict[str, Any]:
    """底层文件读取，返回 dict。失败不抛异常。"""
    file_path = _resolve_knowledge_path(path)
    try:
        content = file_path.read_text(encoding="utf-8")
        return {
            "ok": True,
            "content": content,
            "source": str(file_path),
            "error": "",
        }
    except FileNotFoundError:
        return {"ok": False, "content": "", "source": str(file_path),
                "error": f"文件不存在: {file_path}"}
    except PermissionError:
        return {"ok": False, "content": "", "source": str(file_path),
                "error": f"无权限读取文件: {file_path}"}
    except OSError as e:
        return {"ok": False, "content": "", "source": str(file_path),
                "error": f"文件读取失败: {e}"}
    except Exception as e:
        return {"ok": False, "content": "", "source": str(file_path),
                "error": f"未知错误: {e}"}


def read_local_knowledge(path: Optional[str] = None) -> Dict[str, Any]:
    """读取本地知识文件 week06/local_knowledge.md。

    这是 Agent 唯一允许调用的工具：
      - 只读文件，不产生副作用
      - 不访问网络，不读取其他目录
      - 失败时返回错误 dict，不抛出异常

    Args:
        path: 可选的知识文件路径，默认 week06/local_knowledge.md。

    Returns:
        dict: 始终包含 ok, content, source, error 四个字段。
    """
    return _read_knowledge_file(path)


# ═══════════════════════════════════════════════════════════
#  should_call_tool — 本地规则判断（用于测试 + 降级 fallback）
# ═══════════════════════════════════════════════════════════

def should_call_tool(user_input: str) -> bool:
    """根据用户输入判断是否需要调用本地知识工具（纯规则匹配）。

    注意：LangChain Agent 模式下，工具调用由 Agent 自主决策。
    此函数用于：
      - 测试验证关键词覆盖
      - LangChain 不可用时的纯规则 fallback

    Args:
        user_input: 用户输入的自然语言问题。

    Returns:
        True 表示命中工具关键词，False 表示未命中。
    """
    if not user_input:
        return False
    return any(kw in user_input for kw in TOOL_KEYWORDS)


# ═══════════════════════════════════════════════════════════
#  build_agent_prompt — 组织 prompt（用于测试 + 降级 fallback）
# ═══════════════════════════════════════════════════════════

def build_agent_prompt(
    user_input: str,
    tool_result: Optional[Dict[str, Any]],
    need_tool: bool,
) -> str:
    """将用户问题、工具调用状态和工具结果组织成 prompt 文本。

    注意：LangChain Agent 模式下，run_agent() 使用 AGENT_SYSTEM_PROMPT +
    ChatPromptTemplate 构建 prompt，此函数保留用于：
      - 独立测试
      - LangChain 不可用时的降级场景

    三种情况：
    1. need_tool=True, tool_result["ok"]=True  → 附带资料内容
    2. need_tool=True, tool_result["ok"]=False → 告知工具失败
    3. need_tool=False                         → 告知未调用工具

    Args:
        user_input: 用户原始输入。
        tool_result: read_local_knowledge() 的返回 dict，未调用工具时为 None。
        need_tool: 是否判定需要调用工具。

    Returns:
        str: 组装好的 prompt 文本。
    """
    format_tail = (
        "\n\n"
        "【输出格式要求】\n"
        "请严格按照以下三段格式输出，不要添加额外的解释或前缀：\n\n"
        "结论：\n"
        "<面向用户的最终回答>\n\n"
        "依据：\n"
        "<说明回答依据，包括是否使用了本地资料>\n\n"
        "是否调用工具：\n"
        "<是 / 否 / 是，但失败>\n"
    )

    if need_tool and tool_result is not None and tool_result.get("ok"):
        knowledge = tool_result["content"]
        source = tool_result.get("source", "本地知识文件")
        return (
            f"用户问题：{user_input}\n\n"
            f"以下是从 {source} 读取的本地知识资料，请基于这些资料回答用户问题：\n"
            f"--- 本地资料开始 ---\n"
            f"{knowledge}\n"
            f"--- 本地资料结束 ---\n"
            f"{format_tail}"
        )

    if need_tool:
        error_msg = (
            tool_result.get("error", "未知错误")
            if tool_result is not None
            else "工具未执行"
        )
        return (
            f"用户问题：{user_input}\n\n"
            f"注意：本地工具读取失败，错误信息：{error_msg}\n"
            f"请输出\"未获取到资料 + 当前可回答部分\"，即在结论中说明未获取到资料，"
            f"并基于你的当前能力给出尽可能有限的回答。\n"
            f"{format_tail}"
        )

    return (
        f"用户问题：{user_input}\n\n"
        f"注意：本次未调用本地工具，请基于当前能力回答，并在依据中说明未调用工具。\n"
        f"{format_tail}"
    )


# ═══════════════════════════════════════════════════════════
#  format_final_answer — 兜底格式化
# ═══════════════════════════════════════════════════════════

def format_final_answer(
    conclusion: str,
    evidence: str,
    tool_status: str,
) -> str:
    """对最终结果做兜底格式化，确保输出始终包含固定三段标题。

    Args:
        conclusion: 结论内容。
        evidence: 依据内容。
        tool_status: 工具调用状态，"是" / "否" / "是，但失败"，
                     非法值会被修正为 "否"。

    Returns:
        str: 固定三段格式的回答。
    """
    if tool_status not in _VALID_TOOL_STATUSES:
        tool_status = "否"

    return (
        f"{_SECTION_CONCLUSION}\n"
        f"{conclusion}\n\n"
        f"{_SECTION_EVIDENCE}\n"
        f"{evidence}\n\n"
        f"{_SECTION_TOOL_STATUS}\n"
        f"{tool_status}"
    )


# ═══════════════════════════════════════════════════════════
#  LangChain Agent 组件（Week06 核心 — 在 Week05 基础上新增）
# ═══════════════════════════════════════════════════════════

# ── Agent 系统提示词 ──
# 告诉 Agent：你是谁、有什么工具、何时用工具、如何输出
# 这是 LangChain Agent 的 "指令层"，类似于 Week05 的 OUTPUT_CONTRACT

AGENT_SYSTEM_PROMPT = (
    "你是一个简单 AI Agent，负责根据用户问题决定是否查阅本地知识资料，并给出回答。\n"
    "\n"
    "## 可用工具\n"
    "你可以使用 `read_local_knowledge` 工具读取本地知识文件 week06/local_knowledge.md，"
    "该文件包含以下主题的资料：\n"
    "- 简单 Agent 的定义与核心组成\n"
    "- Agent 的四步工作流程\n"
    "- 工具调用条件（关键词匹配规则）\n"
    "- 输出格式要求\n"
    "- 工具失败处理方式\n"
    "- Agent 功能边界\n"
    "\n"
    "## 工具调用规则\n"
    "当用户问题涉及上述主题，或者明确要求查阅\"资料\"\"知识库\"\"文档\"\"本地资料\""
    "\"Agent\"\"工具\"\"流程\"\"依据\"\"week06\"时，应调用 `read_local_knowledge` 获取资料。\n"
    "对于一般性问题（如编程、数学、天气、通用知识），不需要调用工具，直接回答即可。\n"
    "\n"
    "## 输出格式要求（非常重要，必须严格遵守）\n"
    "你的最终输出必须严格遵循以下三段格式，不要添加额外的解释或前缀：\n"
    "\n"
    "结论：\n"
    "<面向用户的最终回答>\n"
    "\n"
    "依据：\n"
    "<说明回答依据，包括是否使用了本地资料>\n"
    "\n"
    "是否调用工具：\n"
    "<填写以下三者之一：是 / 否 / 是，但失败>\n"
    "\n"
    "说明：\n"
    "- 如果调用了工具且成功获取资料 → 填\"是\"\n"
    "- 如果无需调用工具 → 填\"否\"，并在依据中说明\"未调用本地工具，基于当前能力回答\"\n"
    "- 如果调用了工具但读取失败 → 填\"是，但失败\"，在结论中说明\"未获取到资料\"，"
    "并基于自身能力给出有限回答\n"
)


def _create_read_local_knowledge_tool():
    """创建 LangChain Tool 对象，供 Agent 使用。

    使用 @tool 装饰器将底层 read_local_knowledge 包装为 LangChain 可发现的工具。
    这是 Week06 在 Week05 聊天机器人基础上新增的"工具层"——
    Week05 只有"模型调用"，Week06 加上"工具调用"。

    工具描述会告知 Agent 何时应使用该工具，
    Agent 根据描述 + AGENT_SYSTEM_PROMPT 自主决策是否调用。

    Returns:
        BaseTool: LangChain 工具实例。
    """
    from langchain_core.tools import tool  # type: ignore

    @tool
    def read_local_knowledge_tool(query: str = "") -> str:
        """读取本地知识文件 week06/local_knowledge.md。

        该文件包含以下主题的资料：简单Agent定义、Agent工作流程、工具调用条件、
        输出格式要求、工具失败处理方式、Agent功能边界。

        当用户问题明确涉及"资料""知识库""文档""本地资料""Agent""工具""流程"
        "依据""week06"等关键词时，应使用此工具获取本地资料后再回答。
        对于一般性问题（编程、数学、天气等），不需要调用此工具。
        """
        result = _read_knowledge_file()
        if result["ok"]:
            return result["content"]
        else:
            return f"[工具读取失败] {result['error']}"

    return read_local_knowledge_tool


# ── 输出格式校验辅助函数 ──

def _is_likely_tool_echo(output: str) -> bool:
    """检测模型输出是否是工具返回结果的简单回显（而非基于工具结果的合成回答）。

    小模型（如 llama3.2:1b）有时会把工具返回的资料原样输出，
    而不是基于资料生成三段式回答。检测方法：输出开头与知识文件开头一致。
    """
    knowledge = _read_knowledge_file()
    if not knowledge["ok"] or not output:
        return False
    first_line = knowledge["content"].strip().split("\n")[0]
    return output.strip().startswith(first_line)


def _has_three_sections(text: str) -> bool:
    """检查文本是否包含三段标题（支持全角和半角冒号两种写法）。

    要求每个标题作为独立段落开头出现（行首或紧跟换行符），
    避免模型输出中引用 "结论：" 等词（如代码示例中的 key）造成误判。
    """
    if not text:
        return False
    import re

    markers_sets = [
        ("结论：", "依据：", "是否调用工具："),
        ("结论:", "依据:", "是否调用工具:"),
    ]
    for m1, m2, m3 in markers_sets:
        # 每个标题必须作为行首出现（允许前面有空白）
        if (re.search(rf"(^|\n)\s*{re.escape(m1)}", text)
                and re.search(rf"(^|\n)\s*{re.escape(m2)}", text)
                and re.search(rf"(^|\n)\s*{re.escape(m3)}", text)):
            return True
    return False


def _extract_tool_status_from_agent_output(output: str) -> Optional[str]:
    """尝试从 Agent 输出中提取 tool_status 值。

    在"是否调用工具："行之后查找 "是，但失败" / "是" / "否"。
    用于 Agent 输出格式不完整时的兜底推断。
    """
    for marker in ("是否调用工具：", "是否调用工具:"):
        if marker in output:
            after = output.split(marker, 1)[1].strip()
            # 按优先级匹配（"是，但失败" 必须在 "是" 之前检查）
            for candidate in ("是，但失败", "是", "否"):
                if candidate in after:
                    return candidate
            return after.split("\n")[0].strip() or "否"
    return None


def _infer_tool_info_from_messages(messages: List[Any]) -> tuple:
    """从 Agent 返回的消息列表中推断工具调用状态。

    LangChain 1.3+ 的 create_agent 返回 CompiledStateGraph，
    调用 .invoke() 后得到 {"messages": [...]}。
    从消息列表中检测 ToolMessage 来判断工具是否被调用及是否成功。

    Args:
        messages: Agent 返回的 messages 列表。

    Returns:
        (tool_used: bool, tool_ok: bool)
        - tool_used: Agent 至少调用了一次工具
        - tool_ok: 所有工具调用均成功（无 "[工具读取失败]" 标记）
    """
    from langchain_core.messages import ToolMessage  # type: ignore

    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    if not tool_msgs:
        return False, False

    tool_ok = not any(
        str(m.content).startswith("[工具读取失败]") for m in tool_msgs
    )
    return True, tool_ok


def _empty_usage(source: str = "none") -> Dict[str, Any]:
    """构造空 token 用量结构。"""
    return {"prompt": 0, "completion": 0, "total": 0, "source": source}


def _set_last_token_usage(usage: Optional[Dict[str, Any]], source: str) -> None:
    """记录最近一轮 Agent 调用的 token 用量。"""
    global _LAST_TOKEN_USAGE
    data = usage or {}
    _LAST_TOKEN_USAGE = {
        "prompt": int(data.get("prompt", 0) or 0),
        "completion": int(data.get("completion", 0) or 0),
        "total": int(data.get("total", 0) or 0),
        "source": source,
    }


def get_last_token_usage() -> Dict[str, Any]:
    """返回最近一次 run_agent()/stream_agent() 的 token 用量。"""
    return dict(_LAST_TOKEN_USAGE)


def _extract_response_usage(response: Any, provider: str) -> Dict[str, Any]:
    """复用 Week04 的 token 提取逻辑。"""
    try:
        from week04.qa_assistant_lc import _extract_usage  # type: ignore
        return _extract_usage(response, provider)
    except Exception:
        return _empty_usage("unavailable")


def _extract_messages_usage(messages: List[Any], provider: str) -> Dict[str, Any]:
    """汇总 LangChain Agent 消息列表中的模型 token 用量。"""
    total = _empty_usage("agent")
    found_usage = False

    for message in messages:
        if not (
            getattr(message, "usage_metadata", None)
            or getattr(message, "response_metadata", None)
        ):
            continue
        usage = _extract_response_usage(message, provider)
        if usage.get("prompt") or usage.get("completion") or usage.get("total"):
            found_usage = True
            total["prompt"] += int(usage.get("prompt", 0) or 0)
            total["completion"] += int(usage.get("completion", 0) or 0)
            total["total"] += int(usage.get("total", 0) or 0)

    if not found_usage:
        return _empty_usage("unavailable")
    return total


def _format_token_usage(usage: Optional[Dict[str, Any]] = None) -> str:
    """格式化每轮 token 用量显示行。"""
    data = usage or get_last_token_usage()
    source = data.get("source", "none")
    if source in ("week05", "agent"):
        source_label = "真实用量"
    elif source == "unavailable":
        source_label = "模型未返回用量"
    else:
        source_label = "未调用模型"

    return (
        f"--- Token: prompt={int(data.get('prompt', 0) or 0)} "
        f"completion={int(data.get('completion', 0) or 0)} "
        f"total={int(data.get('total', 0) or 0)} | {source_label} ---"
    )


def _build_local_knowledge_fallback_answer(
    user_input: str,
    tool_result: Dict[str, Any],
    error: str = "",
) -> str:
    """基于已读取的 local_knowledge.md 生成确定性的工具路径降级回答。"""
    evidence = (
        f"依据来自 {tool_result.get('source', 'week06/local_knowledge.md')}。"
        "本地资料说明了简单 Agent 的固定流程、工具调用条件、工具边界和输出格式。"
    )
    if error:
        evidence += f" 模型/Agent 调用未完成：{error}"

    return format_final_answer(
        conclusion=(
            "这个简单 Agent 的工作流程是：接收用户输入，使用本地关键词规则判断是否需要工具，"
            "命中“资料、知识库、文档、根据本地、本地资料、week06、Agent、工具、流程、依据”等关键词时，"
            "调用唯一的本地工具 read_local_knowledge 读取 week06/local_knowledge.md，"
            "然后根据用户问题和工具结果组织固定三段式回答。"
        ),
        evidence=evidence,
        tool_status="是",
    )


def _build_tool_failure_fallback_answer(
    user_input: str,
    error: str,
    model_error: str = "",
) -> str:
    """生成工具读取失败时的三段式降级回答。"""
    evidence = f"本地工具读取失败：{error}"
    if model_error:
        evidence += f"。模型/Agent 调用未完成：{model_error}"

    return format_final_answer(
        conclusion=(
            "未获取到资料。以下是当前可回答部分：该问题命中了本地资料工具调用条件，"
            "Agent 已尝试读取 week06/local_knowledge.md，但读取失败；"
            "因此只能确认流程已完成工具判断，并按失败规则返回降级结果。"
        ),
        evidence=evidence,
        tool_status="是，但失败",
    )


def _generate_with_week05_chain(
    user_input: str,
    tool_result: Optional[Dict[str, Any]],
    need_tool: bool,
    provider: str,
    device: str,
    temperature: float,
    max_tokens: int,
) -> Optional[str]:
    """复用 Week05 结构化聊天能力生成三段式 Agent 回答。

    返回 None 表示 Week05 chain 不可用、模型调用失败或输出不满足三段式格式，
    调用方继续走 LangChain Agent 或本地 fallback。
    """
    try:
        from week05.qa_assistant_structured import (  # type: ignore
            build_chat_chain,
            parse_structured_response,
        )
    except Exception:
        return None

    system = (
        "你是 Week06 Simple Agent 的回答生成器。"
        "请根据用户问题、工具调用状态和工具结果生成最终回答。"
        "最终给用户看的 answer 字段必须严格使用三段式：结论、依据、是否调用工具。"
        "是否调用工具只能填写：是、否、是，但失败。"
    )
    prompt = build_agent_prompt(user_input, tool_result, need_tool)

    try:
        chain = build_chat_chain(
            provider=provider,
            role="default",
            temperature=temperature,
            max_tokens=max_tokens,
            device=device,
            custom_system=system,
        )
        response = chain.invoke({"history": [], "question": prompt})
        content = response.content if hasattr(response, "content") else str(response)
        parsed = parse_structured_response(content)
        output = parsed["answer"] if parsed.get("answer") else content
        usage = _extract_response_usage(response, provider)
    except Exception:
        return None

    if output and _has_three_sections(output):
        _set_last_token_usage(usage, "week05")
        return output
    return None


# ═══════════════════════════════════════════════════════════
#  降级 / Fallback 函数
# ═══════════════════════════════════════════════════════════

def _agent_failure_fallback(
    user_input: str,
    error: str,
    messages: Optional[List[Any]] = None,
) -> str:
    """LangChain Agent 执行失败时的降级处理。

    如果 Agent 在调用工具后出错，可以从 messages 中推断工具状态。
    否则统一按"模型服务不可用"处理。

    Args:
        user_input: 用户原始输入。
        error: Agent 执行错误信息。
        messages: 出错前已生成的消息列表（可选）。

    Returns:
        str: 三段格式的降级回答。
    """
    _set_last_token_usage(_empty_usage(), "none")
    # 如果 Agent 已经调用了工具（能从 messages 推断）
    if messages:
        tool_used, tool_ok = _infer_tool_info_from_messages(messages)
        if tool_used:
            if tool_ok:
                tool_result = read_local_knowledge()
                if tool_result["ok"]:
                    return _build_local_knowledge_fallback_answer(
                        user_input, tool_result, error
                    )
                return _build_tool_failure_fallback_answer(
                    user_input, tool_result["error"], error
                )
            else:
                return _build_tool_failure_fallback_answer(
                    user_input, "Agent 工具调用失败", error
                )

    if should_call_tool(user_input):
        tool_result = read_local_knowledge()
        if tool_result["ok"]:
            return _build_local_knowledge_fallback_answer(
                user_input, tool_result, error
            )
        return _build_tool_failure_fallback_answer(
            user_input, tool_result["error"], error
        )

    # Agent 未调用工具就已出错
    return format_final_answer(
        conclusion=(
            "Agent 执行失败，未能生成回答。"
            "请检查模型服务（Ollama / DeepSeek）是否可用。"
        ),
        evidence=f"Agent 执行错误：{error}",
        tool_status="否",
    )


def _rule_based_fallback(user_input: str, error: str = "") -> str:
    """纯规则 fallback：不依赖 LangChain Agent 或模型服务。

    在以下情况使用：
      - LangChain 未安装（ImportError）
      - 模型初始化失败
      - 网络不通

    基于 should_call_tool() 本地规则判断 + read_local_knowledge() 文件读取，
    生成三段格式降级回答。

    Args:
        user_input: 用户输入。
        error: 导入或初始化错误信息（可选）。

    Returns:
        str: 三段格式的降级回答。
    """
    _set_last_token_usage(_empty_usage(), "none")
    need_tool = should_call_tool(user_input)

    if not need_tool:
        return format_final_answer(
            conclusion=(
                "这是一个通用问题。当前 Agent 无法调用模型"
                "（LangChain Agent 依赖未安装或模型服务不可用），"
                "请检查依赖安装和模型服务状态。"
            ),
            evidence=(
                "未调用本地工具，基于规则判断直接回答。"
                + (f" 错误详情：{error}" if error else "")
            ),
            tool_status="否",
        )

    # 需要工具 → 尝试读取本地文件
    tool_result = read_local_knowledge()

    if tool_result["ok"]:
        return _build_local_knowledge_fallback_answer(
            user_input, tool_result, error
        )

    return _build_tool_failure_fallback_answer(
        user_input, tool_result["error"], error
    )


# ═══════════════════════════════════════════════════════════
#  run_agent — LangChain Agent 完整流程（Week06 核心入口）
# ═══════════════════════════════════════════════════════════

def run_agent(
    user_input: str,
    provider: str = _DEFAULT_PROVIDER,
    device: str = _DEFAULT_DEVICE,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
) -> str:
    """执行完整 Agent 流程并返回最终回答。

    流程（LangChain 1.3+ create_agent API — ReAct 循环）：
      1. 输入验证
      2. 导入 LangChain Agent 依赖 + Week04 LLM
      3. 构建 Agent：
         a. get_model() → LLM 实例          （与 Week05 相同的基础层）
         b. @tool → read_local_knowledge     （Week06 新增工具层）
         c. create_agent(llm, [tool], system_prompt=...)
                                            （Week06 新增 Agent 层）
      4. agent.invoke({"messages": [...]})  → Agent 自主决策 → 调工具 → 组织回答
      5. 从返回的消息列表中提取最终回答 + 工具调用状态
      6. 输出格式校验，必要时兜底格式化
      7. 任何异常都有 fallback，不崩溃

    该函数不会抛出异常，所有错误均在返回值中体现。

    Args:
        user_input: 用户输入的自然语言问题。
        provider: 模型后端，"local" (Ollama) 或 "cloud" (DeepSeek)。
        device: 本地推理设备，"gpu" 或 "cpu"。
        temperature: 生成温度 0~2。
        max_tokens: 最大输出 token 数。

    Returns:
        str: 固定三段格式的最终回答。
    """
    _set_last_token_usage(_empty_usage(), "none")
    # ── Step 1: 输入验证 ──
    if not user_input or not user_input.strip():
        return format_final_answer(
            conclusion="请输入有效问题或任务说明。",
            evidence="输入为空，未执行工具调用。",
            tool_status="否",
        )

    user_input = user_input.strip()

    # ── Step 2: 延迟导入 LangChain Agent 依赖 ─
    # LangChain 1.3+：create_agent 替代了 create_tool_calling_agent + AgentExecutor
    # 新 API 直接接受 system_prompt 字符串，不再需要 ChatPromptTemplate
    try:
        from langchain.agents import create_agent  # type: ignore
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage  # type: ignore
        from week04.qa_assistant_lc import get_model  # type: ignore
    except ImportError as e:
        return _rule_based_fallback(user_input, str(e))

    # ── Step 3: 创建 LLM 实例（与 Week05 相同的基础层）──
    try:
        llm = get_model(provider, temperature, max_tokens, device)
    except Exception as e:
        return _rule_based_fallback(user_input, f"模型初始化失败：{e}")

    # ── Step 4: 创建 LangChain 工具（Week06 新增：工具层）──
    tool = _create_read_local_knowledge_tool()

    # ── Step 5: 创建 Agent（LangChain 1.3+ 新 API）──
    # 混合模式：先用本地规则 should_call_tool() 预判 + 预读取资料，
    # 将资料直接注入 system prompt。这样小模型无需自己调工具就能生成回答，
    # 同时保留工具供大模型做额外查询。
    _preloaded = False       # 是否已通过本地规则预加载资料
    _preload_ok = False      # 预加载的资料是否读取成功
    _augmented_prompt = AGENT_SYSTEM_PROMPT
    need_tool = should_call_tool(user_input)
    tool_result: Optional[Dict[str, Any]] = None

    if need_tool:
        _preloaded = True
        tool_result = read_local_knowledge()
        if tool_result["ok"]:
            _preload_ok = True
            _augmented_prompt += (
                f"\n\n"
                f"【已自动调用 read_local_knowledge 工具，获取到以下本地资料】\n"
                f"来源：{tool_result['source']}\n"
                f"---\n{tool_result['content']}\n---\n"
                f"请基于以上资料直接生成三段式回答。\n"
                f"重要：是否调用工具 这一项必须填\"是\"（因为系统已自动调用了工具）。"
            )
        else:
            _augmented_prompt += (
                f"\n\n"
                f"【系统尝试调用 read_local_knowledge 工具但失败：{tool_result['error']}】\n"
                f"请在结论中说明未获取到资料，并基于自身能力给出有限回答。\n"
                f"重要：是否调用工具 这一项必须填\"是，但失败\"。"
            )

    # 优先复用 Week05 结构化聊天能力生成最终回答；失败时继续走现有 Agent。
    if need_tool:
        week05_output = _generate_with_week05_chain(
            user_input=user_input,
            tool_result=tool_result,
            need_tool=need_tool,
            provider=provider,
            device=device,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if week05_output:
            return week05_output

    agent = create_agent(
        model=llm,
        tools=[tool],
        system_prompt=_augmented_prompt,
    )

    # ── Step 6: 执行 Agent（ReAct 循环）──
    # LangChain 1.3+：传入 {"messages": [...]} 格式
    # recursion_limit 控制最大迭代次数（代替旧的 max_iterations）
    try:
        result = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"recursion_limit": 10},
        )
    except Exception as e:
        return _agent_failure_fallback(user_input, str(e))

    # ── Step 7: 从消息列表提取最终回答 ──
    # result["messages"] 包含完整的对话历史：
    #   HumanMessage → AIMessage(tool_calls=[...]) → ToolMessage → AIMessage → ...
    # 最后一条无 tool_calls 的 AIMessage 即为 Agent 最终回答
    messages = result.get("messages", [])
    agent_usage = _extract_messages_usage(messages, provider)
    _set_last_token_usage(agent_usage, agent_usage.get("source", "agent"))
    output = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            # 跳过带有 tool_calls 的 AIMessage（那是工具调用请求，不是最终回答）
            if hasattr(m, "tool_calls") and m.tool_calls:
                continue
            output = str(m.content)
            break

    # 推断 Agent 实际是否调用了工具（从消息中的 ToolMessage 判断）
    agent_called_tool, agent_tool_ok = _infer_tool_info_from_messages(messages)

    # ── Step 8: 输出格式校验 ──
    if output and _is_likely_tool_echo(output):
        output = ""

    if output and _has_three_sections(output):
        return output

    # ── Step 9: 输出格式不完整 → 兜底格式化 ──
    # 工具状态：预加载优先（本地规则已决定 + 已读取资料），
    # 其次取 Agent 实际调用结果，最后为 "否"
    if _preloaded:
        tool_status = "是" if _preload_ok else "是，但失败"
    elif agent_called_tool:
        tool_status = "是" if agent_tool_ok else "是，但失败"
    else:
        tool_status = "否"

    extracted = _extract_tool_status_from_agent_output(output)
    if extracted and extracted in _VALID_TOOL_STATUSES:
        tool_status = extracted

    week05_output = _generate_with_week05_chain(
        user_input=user_input,
        tool_result=tool_result,
        need_tool=need_tool,
        provider=provider,
        device=device,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if week05_output:
        return week05_output

    if _preloaded:
        if _preload_ok and tool_result is not None:
            return _build_local_knowledge_fallback_answer(
                user_input,
                tool_result,
                "Agent 输出未按固定格式",
            )
        if tool_result is not None:
            return _build_tool_failure_fallback_answer(
                user_input,
                tool_result.get("error", "未知错误"),
                "Agent 输出未按固定格式",
            )

    return format_final_answer(
        conclusion=output or "Agent 未生成有效回答。",
        evidence=(
            "Agent 输出未按固定格式，已自动格式化。"
            + (" Agent 调用了本地工具。" if (_preloaded or agent_called_tool)
               else " Agent 未调用本地工具。")
        ),
        tool_status=tool_status,
    )


def stream_agent(
    user_input: str,
    provider: str = _DEFAULT_PROVIDER,
    device: str = _DEFAULT_DEVICE,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    chunk_size: int = 12,
) -> Iterator[str]:
    """以文本块形式流式返回 Agent 最终回答。

    当前实现保持 run_agent() 的稳定三段式契约，先生成最终回答，
    再按固定大小切片输出，适合作为 CLI 流式显示入口。
    """
    answer = run_agent(
        user_input=user_input,
        provider=provider,
        device=device,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    safe_chunk_size = max(1, chunk_size)
    for i in range(0, len(answer), safe_chunk_size):
        yield answer[i:i + safe_chunk_size]


def _print_stream(chunks: Iterator[str]) -> None:
    """打印流式文本块，并确保最后换行。"""
    last_chunk = ""
    for chunk in chunks:
        last_chunk = chunk
        print(chunk, end="", flush=True)
    if not last_chunk.endswith("\n"):
        print()


# ═══════════════════════════════════════════════════════════
#  CLI — 交互模式 UI（仿 Week04/Week05 风格）
# ═══════════════════════════════════════════════════════════

# ── Banner（启动时显示一次，/help 可重新显示）──

_BANNER = """
╔══════════════════════════════════════════════════════════╗
║   Week06 Simple Agent (LangChain Agent 版)                ║
║   基于 Week05 聊天机器人 + LangChain Agent 框架            ║
║                                                          ║
║   直接输入问题即可启动 Agent，模型会自主判断是否调用工具。  ║
║   Agent 拥有 1 个工具：read_local_knowledge               ║
║   输出固定三段格式：结论 / 依据 / 是否调用工具              ║
║                                                          ║
║   会话指令:                                                ║
║     /help       显示帮助     /provider   切换后端          ║
║     /device     切换设备     /stream     切换流式输出      ║
║     /stats      查看当前配置                              ║
║     q/quit/exit  退出对话                                 ║
╚══════════════════════════════════════════════════════════╝"""

# ── 加载 Provider 信息（与 Week04 共享，失败时使用本地 fallback）──

def _load_provider_info() -> Dict[str, Dict[str, str]]:
    """尝试从 Week04 导入 PROVIDER_INFO，失败时使用内置映射。"""
    try:
        from week04.qa_assistant_lc import PROVIDER_INFO  # type: ignore
        return PROVIDER_INFO  # type: ignore[return-value]
    except ImportError:
        return {
            "local": {"label": "Ollama (本地)", "model": "?"},
            "cloud": {"label": "DeepSeek (云端)", "model": "deepseek-chat"},
        }


# ── 状态打印 ──

def _print_agent_status(
    provider: str,
    device: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
    provider_info: Dict[str, Dict[str, str]],
) -> None:
    """打印当前 Agent 会话状态（仿 Week04 [INFO] 行）。"""
    info = provider_info.get(provider, {})
    device_label = "GPU" if device == "gpu" else "CPU"
    stream_label = "开启" if stream else "关闭"
    print(
        f"[INFO] 后端: {info.get('label', provider)}"
        f" ({info.get('model', '?')})  |  "
        f"设备: {device_label}  |  "
        f"流式: {stream_label}  |  "
        f"Temperature: {temperature}  |  "
        f"Max Tokens: {max_tokens}"
    )


# ── 会话指令处理 ──

def _handle_agent_command(
    cmd_line: str,
    provider: str,
    device: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
    provider_info: Dict[str, Dict[str, str]],
) -> dict:
    """解析并执行 / 开头的会话指令。

    Args:
        cmd_line: 用户输入的完整指令行（含 / 前缀）。
        provider: 当前后端。
        device: 当前设备。
        temperature: 当前温度。
        max_tokens: 当前最大 token 数。
        stream: 当前是否启用流式输出。
        provider_info: 后端信息字典。

    Returns:
        dict: 包含更新后的配置值，action="skip" 表示不执行 Agent 调用。
    """
    parts = cmd_line.strip().split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/help", "/h", "/?"):
        print(_BANNER)
        return {"action": "skip"}

    if cmd == "/provider":
        new_provider = arg.strip().lower()
        if new_provider in provider_info:
            provider = new_provider
            info = provider_info[provider]
            print(f"[OK] 后端已切换为: {info['label']} ({info['model']})")
        elif not new_provider:
            current = provider_info.get(provider, {})
            print(f"[INFO] 当前后端: {current.get('label', provider)} "
                  f"({current.get('model', '?')})")
            print(f"[HELP] 可用后端: cloud (DeepSeek), local (Ollama)")
        else:
            print(f"[ERROR] 未知后端 '{new_provider}'。可用: cloud, local")
        return {"action": "skip", "provider": provider}

    if cmd == "/device":
        new_device = arg.strip().lower()
        if new_device in ("gpu", "cpu"):
            device = new_device
            label = "GPU 加速" if device == "gpu" else "纯 CPU"
            print(f"[OK] 本地推理设备已切换为: {label}")
        elif not new_device:
            label = "GPU 加速" if device == "gpu" else "纯 CPU"
            print(f"[INFO] 当前设备: {label}")
            print(f"[HELP] 可用设备: gpu, cpu (仅 local 后端有效)")
        else:
            print(f"[ERROR] 未知设备 '{new_device}'。可用: gpu, cpu")
        return {"action": "skip", "device": device}

    if cmd == "/stats":
        _print_agent_status(
            provider, device, temperature, max_tokens, stream, provider_info
        )
        return {"action": "skip"}

    if cmd == "/stream":
        if not arg:
            stream = not stream
        else:
            value = arg.strip().lower()
            if value in ("on", "true", "1", "yes", "开", "开启"):
                stream = True
            elif value in ("off", "false", "0", "no", "关", "关闭"):
                stream = False
            else:
                print("[ERROR] /stream 只支持 on/off，或不带参数切换状态。")
                return {"action": "skip"}
        print(f"[OK] 流式输出已{'开启' if stream else '关闭'}")
        return {"action": "skip", "stream": stream}

    # 未知指令
    print(f"[HELP] 未知指令 '{cmd}'。输入 /help 查看可用指令。")
    return {"action": "skip"}


# ── 交互式会话 ──

def _interactive_session() -> None:
    """交互模式主循环（仿 Week04 interactive_session_lc 风格）。

    - 显示 banner + 状态信息
    - 支持 / 会话指令
    - 每轮显示 [N] [Q] / [N] [A]
    - 退出时打印摘要
    """
    provider = _DEFAULT_PROVIDER
    device = _DEFAULT_DEVICE
    temperature = _DEFAULT_TEMPERATURE
    max_tokens = _DEFAULT_MAX_TOKENS
    stream = _DEFAULT_STREAM

    provider_info = _load_provider_info()

    # ── 启动 Banner + 状态 ──
    print(_BANNER)
    _print_agent_status(
        provider, device, temperature, max_tokens, stream, provider_info
    )

    turn = 0
    total_tokens = 0
    while True:
        # ── 获取输入（仿 Week04: 用 turn+1 显示下一轮编号）──
        try:
            user_input = input(f"\n[{turn + 1}] [Q] ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in ("q", "quit", "exit"):
            print("再见！")
            break

        # ── 会话指令（不占轮次）──
        if user_input.startswith("/"):
            result = _handle_agent_command(
                user_input, provider, device, temperature, max_tokens,
                stream, provider_info,
            )
            provider = result.get("provider", provider)
            device = result.get("device", device)
            stream = result.get("stream", stream)
            continue

        # ── 调用 Agent（仿 Week04: 成功调用后才计轮次）──
        turn += 1

        # ── 显示回答（仿 Week04: [N] [A] + 分隔线）──
        print(f"\n[{turn}] [A]")
        print("-" * 40)
        if stream:
            _print_stream(stream_agent(
                user_input,
                provider=provider,
                device=device,
                temperature=temperature,
                max_tokens=max_tokens,
            ))
        else:
            result = run_agent(
                user_input,
                provider=provider,
                device=device,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            print(result)
        usage = get_last_token_usage()
        total_tokens += int(usage.get("total", 0) or 0)
        print(_format_token_usage(usage))

    # ── 退出摘要 ──
    info = provider_info.get(provider, {})
    print(f"\n[SUMMARY] 后端: {info.get('label', provider)}  |  "
          f"对话 {turn} 轮  |  累计 Token: {total_tokens}")


# ═══════════════════════════════════════════════════════════
#  main — CLI 入口
# ═══════════════════════════════════════════════════════════

def main() -> None:
    """CLI 入口。

    支持两种模式：
      1. 单次模式：python week06/simple_agent.py [-p cloud|local] [--no-stream] "问题文本"
      2. 交互模式：python week06/simple_agent.py（无参数启动）
         仿 Week04/Week05 风格：banner + [N] [Q]/[A] + 会话指令
         交互模式中用 /provider cloud 切换 DeepSeek，/provider local 切回 Ollama
    """
    args = sys.argv[1:]

    # 解析 -p / --provider 参数
    provider = _DEFAULT_PROVIDER
    stream = _DEFAULT_STREAM
    question_parts = []
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in ("-p", "--provider") and i + 1 < len(args):
            provider = args[i + 1]
            skip_next = True
        elif arg.startswith("--provider="):
            provider = arg.split("=", 1)[1]
        elif arg in ("--stream", "-s"):
            stream = True
        elif arg == "--no-stream":
            stream = False
        else:
            question_parts.append(arg)

    user_input = " ".join(question_parts)

    if not user_input:
        _interactive_session()
    else:
        if stream:
            _print_stream(stream_agent(user_input, provider=provider))
        else:
            result = run_agent(user_input, provider=provider)
            print(result)
        print(_format_token_usage())


if __name__ == "__main__":
    main()
