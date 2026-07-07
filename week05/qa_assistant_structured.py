"""
CLI 中文问答助手 (结构化输出版) — 在 Week04 LangChain 基础上增强

Week05 增强:
  - 系统提示词自动附加结构化 JSON 输出约定 (OUTPUT_CONTRACT)
  - 4 级降级输出解析器 (parse_structured_response)
  - 单轮/多轮对话自动解析，CLI 展示干净的 answer 文本
  - 历史消息存储解析后的纯文本，不污染后续轮次

本模块导入 Week04 的常量与工具函数，仅重写需要增强的部分。
Week04 原始文件不做任何修改。

使用方式:
  # 单次问答
  python qa_assistant_structured.py "什么是 Prompt Template？"
  python qa_assistant_structured.py -p cloud --stream "解释量子计算"

  # 连续对话
  python qa_assistant_structured.py
  python qa_assistant_structured.py -p cloud --role coder
"""

import sys
import os
import time
import json
import re
import argparse
from typing import List, Optional, Dict
from datetime import datetime

# ── Windows 控制台 UTF-8 编码修复 ──────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

# ── LangChain 核心组件 ──────────────────────────────────
from langchain_core.messages import (
    SystemMessage, HumanMessage, AIMessage, AIMessageChunk, BaseMessage,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ── 从 Week04 导入不变的常量与工具函数 ─────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from week04.qa_assistant_lc import (  # type: ignore
    # 常量
    API_KEY,
    ROLE_PROMPTS,
    PROVIDER_INFO,
    CLOUD_API_BASE, CLOUD_MODEL,
    LOCAL_API_BASE, LOCAL_MODEL,
    DEFAULT_DEVICE, DEVICE_NUM_GPU,
    DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, DEFAULT_PROVIDER,
    # 纯工具函数 (不需要修改)
    get_model,
    _extract_usage,
    _extract_finish_reason,
    _format_stats,
    save_result,
    _handle_langchain_error,
    get_user_input,
    _print_status_line,
)

# ── Week04 的 _INTERACTIVE_BANNER 是模块级变量，直接引用 ──
from week04.qa_assistant_lc import _INTERACTIVE_BANNER  # type: ignore


# ═══════════════════════════════════════════════════════════
#  结构化输出格式约定 (Week05 新增)
# ═══════════════════════════════════════════════════════════

OUTPUT_CONTRACT = (
    "\n\n"
    "【输出格式要求】\n"
    "请严格按照以下 JSON 格式输出回答，不要添加额外的解释或前缀：\n"
    "```json\n"
    "{\n"
    '  "answer": "面向用户的主要回答内容",\n'
    '  "summary": "一句话摘要",\n'
    '  "intent": "qa|coding|translation|medical_advice|unknown",\n'
    '  "follow_up": "如需要追问则给出下一步问题，否则为空字符串",\n'
    '  "confidence": 0.8\n'
    "}\n"
    "```\n"
    "注意：\n"
    "- answer 字段必须包含完整的回答内容，使用中文。\n"
    "- confidence 取值为 0.0 到 1.0 之间的数字。\n"
    "- 如果无法判断用户意图，intent 填 \"unknown\"。\n"
    "- 请确保输出是合法 JSON，字符串使用双引号。"
)


# ═══════════════════════════════════════════════════════════
#  结构化输出解析器 (Week05 新增)
# ═══════════════════════════════════════════════════════════

_VALID_INTENTS = frozenset({"qa", "coding", "translation", "medical_advice", "unknown"})


def _extract_json_object(text: str) -> Optional[str]:
    """从文本中提取第一个完整的 JSON 对象字符串。

    使用手工状态机做括号匹配，正确处理字符串内的转义引号和嵌套大括号。
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _safe_str(value) -> str:
    """将任意值安全转为字符串。"""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def _safe_confidence(value) -> float:
    """将置信度值安全转为 [0.0, 1.0] 范围内的 float。"""
    try:
        v = float(value)
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v
    except (TypeError, ValueError):
        return 0.5


def _build_parsed_result(data: dict, raw_text: str, parse_ok: bool) -> dict:
    """从原始 JSON dict 构建标准化结果字典。

    始终返回 7 个字段：answer, summary, intent, follow_up,
    confidence, raw_text, parse_ok。
    """
    if parse_ok and isinstance(data, dict):
        answer = _safe_str(data.get("answer", ""))
        summary = _safe_str(data.get("summary", ""))
        intent_raw = _safe_str(data.get("intent", "unknown"))
        intent = intent_raw if intent_raw in _VALID_INTENTS else "unknown"
        follow_up = _safe_str(data.get("follow_up", ""))
        confidence = _safe_confidence(data.get("confidence", 0.5))
    else:
        answer = raw_text
        summary = ""
        intent = "unknown"
        follow_up = ""
        confidence = 0.5

    return {
        "answer": answer,
        "summary": summary,
        "intent": intent,
        "follow_up": follow_up,
        "confidence": confidence,
        "raw_text": raw_text,
        "parse_ok": parse_ok,
    }


def parse_structured_response(raw_text: str) -> dict:
    """将模型原始输出解析为统一的结构化字典。

    解析策略（按优先级）：
    1. 整个文本就是合法 JSON 对象。
    2. 文本中包含 ```json ... ``` 围栏代码块。
    3. 文本中嵌入了 JSON 对象（通过括号匹配提取）。
    4. 以上均失败 → fallback，保留原始文本作为 answer。

    Args:
        raw_text: 模型原始输出文本。

    Returns:
        dict: 始终包含 answer, summary, intent, follow_up,
              confidence, raw_text, parse_ok 七个字段。
    """
    # ── 空输入保护 ──
    text = (raw_text or "").strip()
    if not text:
        return {
            "answer": "",
            "summary": "",
            "intent": "unknown",
            "follow_up": "",
            "confidence": 0.0,
            "raw_text": raw_text or "",
            "parse_ok": False,
        }

    # ── 策略 1: 直接 JSON 解析 ──
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return _build_parsed_result(data, raw_text, parse_ok=True)
    except (json.JSONDecodeError, ValueError):
        pass

    # ── 策略 2: ```json ... ``` 围栏代码块 ──
    m_fence = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m_fence:
        try:
            data = json.loads(m_fence.group(1).strip())
            if isinstance(data, dict):
                return _build_parsed_result(data, raw_text, parse_ok=True)
        except (json.JSONDecodeError, ValueError):
            pass

    # ── 策略 3: 括号匹配提取 JSON 对象 ──
    json_candidate = _extract_json_object(text)
    if json_candidate:
        try:
            data = json.loads(json_candidate)
            if isinstance(data, dict):
                return _build_parsed_result(data, raw_text, parse_ok=True)
        except (json.JSONDecodeError, ValueError):
            pass

    # ── 策略 4: fallback ──
    return _build_parsed_result({}, raw_text, parse_ok=False)


# ═══════════════════════════════════════════════════════════
#  Prompt Template 构建 (Week05 增强版)
# ═══════════════════════════════════════════════════════════

def build_system_text(role: str, custom_system: Optional[str] = None) -> str:
    """根据角色或自定义内容构建 system prompt 文本。

    与 Week04 的区别：始终在末尾附加 OUTPUT_CONTRACT，
    即使使用 --system 自定义提示词也不会移除。
    """
    base = custom_system if custom_system else ROLE_PROMPTS.get(role, ROLE_PROMPTS["default"])
    return base + "\n" + OUTPUT_CONTRACT


def build_prompt_template(role: str, custom_system: Optional[str] = None) -> ChatPromptTemplate:
    """单轮 Prompt Template — 无历史消息。"""
    system_text = build_system_text(role, custom_system)
    # 转义 system_text 中的花括号，避免被 LangChain 误解析为模板变量
    system_text_escaped = system_text.replace("{", "{{").replace("}", "}}")
    return ChatPromptTemplate.from_messages([
        ("system", system_text_escaped),
        ("human", "{question}"),
    ])


def build_chat_prompt_template(role: str, custom_system: Optional[str] = None) -> ChatPromptTemplate:
    """多轮 Prompt Template — 带 MessagesPlaceholder 承载对话历史。"""
    system_text = build_system_text(role, custom_system)
    # 转义 system_text 中的花括号，避免被 LangChain 误解析为模板变量
    system_text_escaped = system_text.replace("{", "{{").replace("}", "}}")
    return ChatPromptTemplate.from_messages([
        ("system", system_text_escaped),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])


def build_single_shot_chain(provider: str, role: str, temperature: float,
                            max_tokens: int, device: str = "gpu",
                            custom_system: Optional[str] = None):
    """构建单次问答 Chain: prompt | model。"""
    prompt = build_prompt_template(role, custom_system)
    model = get_model(provider, temperature, max_tokens, device)
    return prompt | model


def build_chat_chain(provider: str, role: str, temperature: float,
                     max_tokens: int, device: str = "gpu",
                     custom_system: Optional[str] = None):
    """构建多轮对话 Chain: prompt | model。"""
    prompt = build_chat_prompt_template(role, custom_system)
    model = get_model(provider, temperature, max_tokens, device)
    return prompt | model


# ═══════════════════════════════════════════════════════════
#  CLI 参数解析 (与 Week04 保持一致)
# ═══════════════════════════════════════════════════════════

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数 — 与 Week04 保持一致的 CLI 接口。"""
    parser = argparse.ArgumentParser(
        description="CLI 中文问答助手 (结构化输出版) — 支持本地 Ollama + 云端 DeepSeek 双后端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 本地 Ollama (默认, GPU加速)
  python qa_assistant_structured.py "什么是机器学习？"
  python qa_assistant_structured.py --stream --role teacher "解释量子计算"

  # 本地 Ollama (CPU 推理)
  python qa_assistant_structured.py -d cpu "什么是机器学习？"

  # 云端 DeepSeek
  python qa_assistant_structured.py -p cloud "什么是机器学习？"
  python qa_assistant_structured.py -p cloud --stream "写一首关于春天的诗"

  # 交互模式
  python qa_assistant_structured.py
  python qa_assistant_structured.py -p cloud --role coder
        """,
    )
    parser.add_argument(
        "question", type=str, nargs="?", default=None,
        help="要提问的问题 (不提供则进入交互模式)",
    )
    parser.add_argument(
        "--provider", "-p", type=str, default=DEFAULT_PROVIDER,
        choices=["cloud", "local"],
        help=f"模型后端: cloud=DeepSeek API, local=Ollama 本地 (默认: {DEFAULT_PROVIDER})",
    )
    parser.add_argument(
        "--device", "-d", type=str, default=DEFAULT_DEVICE,
        choices=["cpu", "gpu"],
        help=f"本地推理设备: cpu=CPU推理, gpu=GPU加速 (仅 local 后端有效, 默认: {DEFAULT_DEVICE})",
    )
    parser.add_argument(
        "--role", "-r", type=str, default="default",
        choices=list(ROLE_PROMPTS.keys()),
        help="助手角色预设 (default: 通用助手)",
    )
    parser.add_argument(
        "--system", "-s", type=str, default=None,
        help="自定义系统提示词 (覆盖 --role)",
    )
    parser.add_argument(
        "--temperature", "-t", type=float, default=DEFAULT_TEMPERATURE,
        help=f"生成温度 0~2 (默认: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--max-tokens", "-m", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"最大输出 token 数 (默认: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--stream", action="store_true", default=False,
        help="启用流式输出 (逐 token 打印)",
    )
    parser.add_argument(
        "--save", type=str, default=None,
        help="将回答保存到指定文件 (仅单次模式)",
    )
    parser.add_argument(
        "--no-echo", action="store_true", default=False,
        help="不显示统计数据行",
    )
    return parser.parse_args(argv)


# ═══════════════════════════════════════════════════════════
#  会话指令处理 (Week05 增强版)
# ═══════════════════════════════════════════════════════════

def handle_session_command(
    user_input: str,
    *,
    role: str,
    system: Optional[str],
    stream: bool,
    provider: str,
    device: str,
    history: List[BaseMessage],
    session_stats: dict,
) -> dict:
    """[交互模式] 解析并执行 / 开头的会话指令。

    与 Week04 的区别：/role 和 /system 更新 history[0] 时
    会自动附加 OUTPUT_CONTRACT。
    """
    cmd_line = user_input.strip()
    parts = cmd_line.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1] if len(parts) > 1 else ""

    if cmd in ("/help", "/h", "/?"):
        print(_INTERACTIVE_BANNER)
        return {"action": "skip"}

    if cmd == "/provider":
        new_provider = arg.strip().lower()
        if new_provider in PROVIDER_INFO:
            provider = new_provider
            info = PROVIDER_INFO[provider]
            print(f"[OK] 后端已切换为: {info['label']} ({info['model']})")
        elif not new_provider:
            current = PROVIDER_INFO.get(provider, {})
            print(f"[INFO] 当前后端: {current.get('label', provider)} "
                  f"({current.get('model', '?')})")
            print(f"[HELP] 可用后端: cloud (DeepSeek), local (Ollama)")
        else:
            print(f"[ERROR] 未知后端 '{new_provider}'。可用: cloud, local")
        return {"action": "skip", "provider": provider}

    if cmd == "/role":
        new_role = arg.strip()
        if new_role in ROLE_PROMPTS:
            role = new_role
            system = None
            print(f"[OK] 角色已切换为: {new_role}")
            # ★ Week05 增强：更新 history 时包含 OUTPUT_CONTRACT
            if history and isinstance(history[0], SystemMessage):
                history[0] = SystemMessage(
                    content=ROLE_PROMPTS[new_role] + "\n" + OUTPUT_CONTRACT)
        elif not new_role:
            print(f"[HELP] 可用角色: {', '.join(ROLE_PROMPTS.keys())}")
            print(f"[HELP] 当前角色: {role}")
        else:
            print(f"[ERROR] 未知角色 '{new_role}'。"
                  f"可用: {', '.join(ROLE_PROMPTS.keys())}")
        return {"action": "skip", "role": role, "system": system, "history": history}

    if cmd == "/system":
        new_system = arg.strip()
        if new_system:
            system = new_system
            role = "default"
            print(f"[OK] 自定义 system prompt 已设置")
            # ★ Week05 增强：更新 history 时包含 OUTPUT_CONTRACT
            if history and isinstance(history[0], SystemMessage):
                history[0] = SystemMessage(
                    content=new_system + "\n" + OUTPUT_CONTRACT)
        else:
            if system:
                print(f"[INFO] 当前自定义 system prompt: {system[:80]}...")
            else:
                print(f"[INFO] 当前使用角色预设: {role}")
        return {"action": "skip", "role": role, "system": system, "history": history}

    if cmd == "/stream":
        stream = not stream
        print(f"[OK] 流式输出已切换为: {'ON' if stream else 'OFF'}")
        return {"action": "skip", "stream": stream}

    if cmd == "/device":
        new_device = arg.strip().lower()
        if new_device in DEVICE_NUM_GPU:
            device = new_device
            label = "GPU 加速" if device == "gpu" else "纯 CPU"
            print(f"[OK] 本地推理设备已切换为: {label}")
        elif not new_device:
            current_label = "GPU 加速" if device == "gpu" else "纯 CPU"
            print(f"[INFO] 当前设备: {current_label}")
            print(f"[HELP] 可用设备: cpu, gpu (仅 local 后端有效)")
        else:
            print(f"[ERROR] 未知设备 '{new_device}'。可用: cpu, gpu")
        return {"action": "skip", "device": device}

    if cmd == "/clear":
        sys_msg = history[0] if history and isinstance(history[0], SystemMessage) else None
        history = [sys_msg] if sys_msg else []
        session_stats["turns"] = 0
        session_stats["total_tokens"] = 0
        print("[OK] 对话上下文已清空")
        return {"action": "skip", "history": history}

    if cmd == "/save":
        filepath = arg.strip() or f"qa_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                info = PROVIDER_INFO.get(provider, {})
                f.write(f"# 中文问答助手 (结构化输出版) — 会话记录\n")
                f.write(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 后端: {info.get('label', provider)} ({info.get('model', '?')})\n")
                f.write(f"# 角色: {role}\n")
                f.write(f"# 对话轮次: {session_stats.get('turns', 0)}\n\n")
                for msg in history:
                    if isinstance(msg, SystemMessage):
                        f.write(f"## System Prompt\n\n{msg.content}\n\n")
                    elif isinstance(msg, HumanMessage):
                        f.write(f"### [Q]\n\n{msg.content}\n\n")
                    elif isinstance(msg, AIMessage):
                        f.write(f"### [A]\n\n{msg.content}\n\n")
            print(f"[SAVED] 对话记录已保存到: {os.path.abspath(filepath)}")
        except OSError as e:
            print(f"[ERROR] 保存失败: {e}", file=sys.stderr)
        return {"action": "skip"}

    if cmd == "/stats":
        info = PROVIDER_INFO.get(provider, {})
        turns = session_stats.get("turns", 0)
        total_tokens = session_stats.get("total_tokens", 0)
        msg_count = len([m for m in history if not isinstance(m, SystemMessage)])
        print(f"[STATS] 后端: {info.get('label', provider)}  |  "
              f"角色: {role}  |  轮次: {turns}  |  "
              f"消息数: {msg_count}  |  累计 Token: {total_tokens}")
        return {"action": "skip"}

    # 未知指令
    print(f"[HELP] 未知指令 '{cmd}'。输入 /help 查看可用指令。")
    return {"action": "skip"}


# ═══════════════════════════════════════════════════════════
#  单次问答 (Week05 增强版 — 结构化解析)
# ═══════════════════════════════════════════════════════════

def single_shot_lc(
    question: str,
    provider: str = DEFAULT_PROVIDER,
    role: str = "default",
    system: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stream: bool = False,
    save: Optional[str] = None,
    no_echo: bool = False,
    device: str = DEFAULT_DEVICE,
) -> int:
    """[单次模式] 一问一答 — 含结构化解析。

    与 Week04 的区别：
    - 非流式模式解析响应并展示干净的 answer 字段
    - 流式模式先实时显示，结束后再解析
    """
    chain = build_single_shot_chain(provider, role, temperature, max_tokens,
                                    device, system)
    info = PROVIDER_INFO.get(provider, {})

    t0 = time.perf_counter()

    try:
        if stream:
            # ── 流式输出 ──
            print("[A] ", end="")
            full_text = ""
            last_chunk = None
            for chunk in chain.stream({"question": question}):
                text = chunk.content if isinstance(chunk.content, str) else ""
                if text:
                    print(text, end="", flush=True)
                full_text += text
                last_chunk = chunk
            print()
            elapsed = time.perf_counter() - t0
            content = full_text
            # 流式结束后解析（不影响已显示内容）
            parsed = parse_structured_response(content)
            usage = _extract_usage(last_chunk, provider) if last_chunk else \
                {"prompt": 0, "completion": 0, "total": 0}
            finish = _extract_finish_reason(last_chunk, provider) if last_chunk else "stop"
        else:
            # ── 非流式输出 ──
            response = chain.invoke({"question": question})
            elapsed = time.perf_counter() - t0
            content = response.content if hasattr(response, "content") else str(response)
            usage = _extract_usage(response, provider)
            finish = _extract_finish_reason(response, provider)

            # ★ Week05 增强：解析后展示 answer 字段
            parsed = parse_structured_response(content)
            display_text = parsed["answer"] if parsed["parse_ok"] else content

            print("\n" + "=" * 60)
            print("  [回答]")
            print("=" * 60)
            print(display_text)

        if not no_echo:
            print()
            print("-" * 60)
            print(_format_stats(info.get("model", "?"), elapsed, usage, finish))
            print("-" * 60)

        if save:
            save_result(content, save, question, role, provider)

        return 0

    except Exception as e:
        _handle_langchain_error(e, provider)
        return 1


# ═══════════════════════════════════════════════════════════
#  连续对话 (Week05 增强版 — 结构化解析 + 干净历史)
# ═══════════════════════════════════════════════════════════

def interactive_session_lc(
    provider: str = DEFAULT_PROVIDER,
    role: str = "default",
    system: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stream: bool = False,
    no_echo: bool = False,
    device: str = DEFAULT_DEVICE,
) -> int:
    """[交互模式] 连续对话循环 — 含结构化解析。

    与 Week04 的区别：
    - 模型响应经过解析，CLI 展示干净的 answer 文本
    - 历史中存储解析后的 answer（而非原始 JSON），避免污染上下文
    - 流式模式先实时显示，结束后解析再存入历史
    """
    system_text = build_system_text(role, system)
    history: List[BaseMessage] = [
        SystemMessage(content=system_text)
    ]
    session_stats = {"turns": 0, "total_tokens": 0}
    info = PROVIDER_INFO.get(provider, {})

    print(_INTERACTIVE_BANNER)
    device_label = "GPU" if device == "gpu" else "CPU"
    print(f"[INFO] 后端: {info.get('label', provider)} ({info.get('model', '?')})  |  "
          f"设备: {device_label}")
    print(f"[INFO] 角色: {'自定义' if system else role}  |  "
          f"流式: {'ON' if stream else 'OFF'}  |  "
          f"Temperature: {temperature}  |  Max Tokens: {max_tokens}")
    print(f"[INFO] 结构化输出: 已启用 (Week05)")

    while True:
        # --- 获取输入 ---
        user_input = get_user_input(f"\n[{session_stats['turns'] + 1}] [Q] ")
        if user_input is None:
            break
        if not user_input:
            continue

        # --- 会话指令 ---
        if user_input.startswith("/"):
            result = handle_session_command(
                user_input,
                role=role, system=system, stream=stream, provider=provider,
                device=device, history=history, session_stats=session_stats,
            )
            role = result.get("role", role)
            system = result.get("system", system)
            stream = result.get("stream", stream)
            provider = result.get("provider", provider)
            device = result.get("device", device)
            history = result.get("history", history)
            session_stats = result.get("session_stats", session_stats)
            if result.get("action") == "break":
                break
            if "provider" in result or "device" in result:
                info = PROVIDER_INFO.get(provider, {})
            _print_status_line(provider, role, system, stream, device,
                               temperature, max_tokens)
            continue

        # --- 重建 chain (反映当前的 provider/device/role) ---
        chain = build_chat_chain(provider, role, temperature, max_tokens,
                                 device, system)

        # --- Stage 2: 模型调用 ---
        t0 = time.perf_counter()
        try:
            if stream:
                # ── 流式输出 ──
                print(f"[{session_stats['turns'] + 1}] [A] ", end="")
                full_text = ""
                last_chunk = None
                for chunk in chain.stream({
                    "history": history,
                    "question": user_input,
                }):
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if text:
                        print(text, end="", flush=True)
                    full_text += text
                    last_chunk = chunk
                print()
                elapsed = time.perf_counter() - t0
                content = full_text
                usage = _extract_usage(last_chunk, provider) if last_chunk else \
                    {"prompt": 0, "completion": 0, "total": 0}
                finish = _extract_finish_reason(last_chunk, provider) if last_chunk else "stop"

                # ★ Week05 增强：流式结束后解析，提取干净的 answer
                parsed = parse_structured_response(content)
                answer_text = parsed["answer"] if parsed["answer"] else content
            else:
                # ── 非流式输出 ──
                response = chain.invoke({
                    "history": history,
                    "question": user_input,
                })
                elapsed = time.perf_counter() - t0
                content = response.content if hasattr(response, "content") else str(response)
                usage = _extract_usage(response, provider)
                finish = _extract_finish_reason(response, provider)

                # ★ Week05 增强：解析后展示 answer 字段
                parsed = parse_structured_response(content)
                display_text = parsed["answer"] if parsed["parse_ok"] else content
                answer_text = display_text

                print(f"\n[{session_stats['turns'] + 1}] [A]")
                print("-" * 40)
                print(display_text)

            # --- 模型调用成功后再写入本轮历史 ---
            # ★ Week05 增强：历史存储解析后的 answer 文本，而非原始 JSON
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=answer_text))

            if not no_echo:
                print(_format_stats(info.get("model", "?"), elapsed, usage, finish))

        except Exception as e:
            _handle_langchain_error(e, provider)
            continue

        session_stats["turns"] += 1

    print(f"\n[SUMMARY] 后端: {info.get('label', provider)}  |  "
          f"对话 {session_stats['turns']} 轮")
    return 0


# ═══════════════════════════════════════════════════════════
#  统一入口
# ═══════════════════════════════════════════════════════════

def run(question: Optional[str] = None,
        provider: str = DEFAULT_PROVIDER,
        role: str = "default",
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        stream: bool = False,
        save: Optional[str] = None,
        no_echo: bool = False,
        device: str = DEFAULT_DEVICE) -> int:
    """统一入口 — 按是否有 CLI 问题参数分流。"""
    if question and question.strip():
        return single_shot_lc(
            question=question.strip(), provider=provider, role=role, system=system,
            temperature=temperature, max_tokens=max_tokens,
            stream=stream, save=save, no_echo=no_echo, device=device,
        )
    else:
        return interactive_session_lc(
            provider=provider, role=role, system=system,
            temperature=temperature, max_tokens=max_tokens,
            stream=stream, no_echo=no_echo, device=device,
        )


def main():
    """CLI 入口。"""
    args = parse_args()
    return run(
        question=args.question,
        provider=args.provider,
        role=args.role,
        system=args.system,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stream=args.stream,
        save=args.save,
        no_echo=args.no_echo,
        device=args.device,
    )


if __name__ == "__main__":
    sys.exit(main())
