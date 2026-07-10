"""
Week07 中文资料问答 Agent — 大模型入门课程资料助手

架构层次：
  课程问答路径：
    输入路由 → 本地关键词检索与重排 → 上下文组装 → 模型生成结论
    → 本地补充引用/状态 → JSONL 运行记录
  固定回复路径：
    问候、帮助、范围外和工具读取失败不调用模型。

课程问答默认使用所选 provider 的模型服务；资料检索、引用和日志始终由本地逻辑完成。

使用方式:
  # 直接回答
  python week07/simple_agent.py --no-stream "你能帮助我做什么？"

  # 本地资料检索
  python week07/simple_agent.py --no-stream "根据课程资料，提示词模板由哪些部分组成？"

  # 模拟工具失败
  python week07/simple_agent.py --no-stream --simulate-tool-failure "根据课程资料，什么是 RAG？"

  # 交互模式
  python week07/simple_agent.py
"""

import sys
import os
import re
import json
import uuid
import shutil
import textwrap
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Iterator, Tuple

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

# 项目根路径（供延迟导入 week04 使用）
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ═══════════════════════════════════════════════════════════
#  常量
# ═══════════════════════════════════════════════════════════

_KNOWLEDGE_FILENAME = "local_knowledge.md"
_RUNS_DIR_NAME = "runs"
_RUNS_FILENAME = "agent_runs.jsonl"

_SECTION_CONCLUSION = "结论："
_SECTION_EVIDENCE = "依据："
_SECTION_TOOL_STATUS = "是否调用工具："
_SECTION_RUN_RECORD = "运行记录："

_VALID_TOOL_STATUSES = frozenset({"是", "否", "是，但失败"})

# 时区：北京时间 UTC+8
_TZ_BEIJING = timezone(timedelta(hours=8))

# ── 路由关键词 ──

# 直接回答：问候、帮助、使用方式等
_DIRECT_ANSWER_PATTERNS: List[str] = [
    "你能帮助我做什么", "你能做什么", "你可以做什么", "你会做什么",
    "你能帮我做什么", "你能干什么", "你可以干什么",
    "你好", "嗨", "hello", "hi",
    "帮助", "使用方式", "怎么用", "如何使用", "怎么使用",
    "介绍一下你自己", "你是谁", "你是什么", "你叫什么",
    "有什么功能", "功能有哪些", "你能干什么",
    "你的能力", "能力范围", "使用说明",
]

# 本地检索：课程概念问题或显式资料请求
_LOCAL_RETRIEVAL_PATTERNS: List[str] = [
    "根据资料", "知识库", "文档", "课程资料", "本地资料",
    "提示词模板", "提示词", "prompt template",
    "上下文窗口", "上下文",
    "rag", "RAG", "检索增强", "本地检索",
    "agent", "Agent", "工具调用", "工具",
    "使用边界", "功能边界", "能力边界",
    "大模型入门", "llm入门",
    "token", "Token", "TOKEN",
    "embedding", "向量", "嵌入",
    "资料",
]

# 本地检索中的弱匹配词（单独出现不触发检索，但与其他词组合时触发）
_WEAK_RETRIEVAL_PATTERNS: List[str] = [
    "微调", "fine-tuning", "预训练", "生成式",
    "大模型", "llm", "LLM",
    "检索", "生成",
]

# 课程资料所有二级标题关键词（用于路由判断）
_COURSE_TOPIC_KEYWORDS: List[str] = [
    "提示词模板", "上下文窗口", "rag", "本地检索",
    "agent", "工具调用", "使用边界",
    "prompt", "token", "embedding",
]


# ═══════════════════════════════════════════════════════════
#  工具路径解析
# ═══════════════════════════════════════════════════════════

def _resolve_knowledge_path(path: Optional[str] = None) -> Path:
    """解析 local_knowledge.md 的绝对路径。"""
    if path:
        return Path(path).resolve()
    return (Path(__file__).parent / _KNOWLEDGE_FILENAME).resolve()


def _resolve_runs_path() -> Path:
    """解析 runs 目录的绝对路径。"""
    return Path(__file__).parent / _RUNS_DIR_NAME


def _resolve_runs_file_path() -> Path:
    """解析 JSONL 日志文件的绝对路径。"""
    return _resolve_runs_path() / _RUNS_FILENAME


def _relative_source_path(source_path: str) -> str:
    """返回不会暴露本机路径的相对来源标识。"""
    try:
        return Path(source_path).resolve().relative_to(Path(__file__).parent.parent).as_posix()
    except (ValueError, OSError):
        if Path(source_path).name == _KNOWLEDGE_FILENAME:
            return f"week07/{_KNOWLEDGE_FILENAME}"
        return Path(source_path).name or "week07/未知来源"


def _sanitize_error_message(error: str, fallback: str) -> str:
    """保留可读错误原因，同时移除 Windows/Unix 绝对路径。"""
    if not error:
        return fallback
    sanitized = re.sub(r"(?:[A-Za-z]:[\\/]|/)[^\s，。；;]+", "[已隐藏路径]", str(error))
    return sanitized or fallback


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
                "error": "本地资料文件不存在"}
    except PermissionError:
        return {"ok": False, "content": "", "source": str(file_path),
                "error": "无权限读取本地资料文件"}
    except OSError:
        return {"ok": False, "content": "", "source": str(file_path),
                "error": "本地资料文件读取失败"}
    except Exception:
        return {"ok": False, "content": "", "source": str(file_path),
                "error": "本地资料文件读取发生未知错误"}


def read_local_knowledge(path: Optional[str] = None) -> Dict[str, Any]:
    """读取本地知识文件 week07/local_knowledge.md。

    这是 Agent 唯一允许调用的工具：
      - 只读文件，不产生副作用
      - 不访问网络，不读取其他目录
      - 失败时返回错误 dict，不抛出异常

    Args:
        path: 可选的知识文件路径，默认 week07/local_knowledge.md。

    Returns:
        dict: 始终包含 ok, content, source, error 四个字段。
    """
    return _read_knowledge_file(path)


# ═══════════════════════════════════════════════════════════
#  输入路由 — 确定性规则（默认路径，不依赖 LLM）
# ═══════════════════════════════════════════════════════════

def _route_input(user_input: str) -> str:
    """根据用户输入确定路由类型（纯规则，不依赖模型）。

    Args:
        user_input: 用户输入文本。

    Returns:
        str: 'direct_answer' | 'local_retrieval' | 'out_of_scope'
    """
    if not user_input or not user_input.strip():
        return "direct_answer"  # 空输入按直接回答处理，提示输入有效问题

    text = user_input.strip()
    text_lower = text.lower()

    # ── 检查直接回答模式 ──
    for pattern in _DIRECT_ANSWER_PATTERNS:
        if pattern in text_lower:
            return "direct_answer"

    # ── 检查本地检索模式 ──
    for pattern in _LOCAL_RETRIEVAL_PATTERNS:
        if pattern.lower() in text_lower:
            return "local_retrieval"

    # ── 弱匹配：课程主题词需要至少命中 2 个才触发检索 ──
    weak_hits = sum(1 for p in _WEAK_RETRIEVAL_PATTERNS if p.lower() in text_lower)
    if weak_hits >= 2:
        return "local_retrieval"

    # ── 默认：范围外 ──
    return "out_of_scope"


# ═══════════════════════════════════════════════════════════
#  should_call_tool — 保留兼容，基于路由结果
# ═══════════════════════════════════════════════════════════

def should_call_tool(user_input: str) -> bool:
    """根据用户输入判断是否需要调用本地知识工具（纯规则匹配）。

    保留以兼容 Week06 的调用方式。Week07 内部使用 _route_input()。

    Args:
        user_input: 用户输入。

    Returns:
        True 表示需要调用工具，False 表示不需要。
    """
    if not user_input:
        return False
    return _route_input(user_input) == "local_retrieval"


# ═══════════════════════════════════════════════════════════
#  资料切分与检索
# ═══════════════════════════════════════════════════════════

def _split_sections(content: str, source_path: str) -> List[Dict[str, Any]]:
    """按 Markdown 二级标题切分资料内容。

    第一个 ## 之前的内容作为"前言"片段（序号 0）。
    每个 ## 标题开始到下一个 ## 标题之前为一个片段。

    Args:
        content: Markdown 全文。
        source_path: 来源文件路径（用于生成相对引用）。

    Returns:
        list[dict]: 每个元素包含 index, heading, body, source 字段。
    """
    sections: List[Dict[str, Any]] = []
    # 按 ## 标题切分（行首的 ## ）
    parts = re.split(r'\n(?=## )', content)

    # 计算相对来源路径
    try:
        rel_source = str(Path(source_path).relative_to(Path(__file__).parent.parent))
    except ValueError:
        rel_source = Path(source_path).name

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        lines = part.split("\n", 1)
        first_line = lines[0].strip()
        if first_line.startswith("## "):
            heading = first_line[3:].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
        else:
            heading = "前言"
            body = part

        sections.append({
            "index": i,
            "heading": heading,
            "body": body,
            "source": rel_source,
        })

    return sections


def _clean_query(query: str) -> str:
    """清洗查询文本，去除标点、空白和常见停用词，用于关键词匹配。

    Args:
        query: 原始查询文本。

    Returns:
        str: 清洗后的纯中文/英文关键词文本。
    """
    # 去除常见标点和空白
    cleaned = re.sub(
        r'[，。！？、；：""''（）()\[\]【】\-—…\.\,\!\?\:\;\"\'\/\\\s\n\r\t]+',
        '',
        query
    )
    # 去除常见中文停用词
    stop_words = {
        '的', '是', '了', '在', '和', '与', '或', '等',
        '什么', '怎么', '如何', '哪些', '哪个', '哪',
        '请', '根据', '帮助', '我', '你', '您', '它', '她', '他',
        '一个', '一下', '可以', '能', '会', '有', '这', '那',
        '也', '就', '都', '要', '把', '被', '从', '到', '对', '为',
        '以', '及', '而', '但', '中', '个', '里', '上', '下',
        '去', '来', '做', '让', '给', '向', '由', '其',
        '着', '过', '得', '地', '么', '吗', '呢', '吧', '啊',
        '用', '使', '该', '此', '不', '很', '太', '更', '最',
        '非常', '比较', '特别', '稍微', '几乎', '大约', '左右',
        '我们', '你们', '他们', '她们', '它们',
        '说明', '组成', '部分', '作用', '查询', '问题', '回答',
        '当前', '本次', '是否', '应该', '需要',
        '哪些', '其他', '一些', '所有', '任何',
        '课程', '资料', '本地', '知识库', '文档',
    }
    result = []
    i = 0
    while i < len(cleaned):
        # 尝试匹配双字停用词
        if i + 1 < len(cleaned) and cleaned[i:i+2] in stop_words:
            i += 2
            continue
        # 单字停用词
        if cleaned[i] in stop_words:
            i += 1
            continue
        result.append(cleaned[i])
        i += 1
    return ''.join(result)


def _search_sections(
    query: str,
    sections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """使用中文关键词匹配搜索最相关的资料片段。

    规则：
      - 标题命中优先（权重 ×3）
      - 正文命中次数其次（权重 ×1）
      - 返回分数最高的 1—2 个片段
      - 无命中时返回空列表

    Args:
        query: 清洗后的查询文本。
        sections: _split_sections 返回的片段列表。

    Returns:
        list[dict]: 包含 score 和 matched_terms 的片段列表，按分数降序。
    """
    cleaned = _clean_query(query)
    if not cleaned:
        return []

    # 提取查询中的 bigram 用于匹配
    bigrams: List[str] = []
    for i in range(len(cleaned) - 1):
        bg = cleaned[i:i+2]
        if bg not in bigrams:
            bigrams.append(bg)
    # 也加入单字
    singles = list(cleaned)

    scored: List[Dict[str, Any]] = []
    for sec in sections:
        heading = sec["heading"]
        body = sec["body"]
        score = 0
        matched_terms: List[str] = []

        # 标题匹配：bigram 权重 3
        for bg in bigrams:
            if bg in heading:
                score += 3
                if bg not in matched_terms:
                    matched_terms.append(bg)
        # 标题匹配：单字权重 1
        for ch in singles:
            if ch in heading:
                score += 1

        # 正文匹配：bigram 权重 1（按出现次数）
        for bg in bigrams:
            count = body.count(bg)
            if count > 0:
                score += count
                if bg not in matched_terms:
                    matched_terms.append(bg)

        if score > 0:
            scored.append({**sec, "score": score, "matched_terms": matched_terms})

    # 按分数降序排列，过滤低分噪音（单一弱匹配不算有效命中）
    # 阈值 3：单个 bigram 命中标题 → 3 分，或 ≥3 次正文命中
    scored.sort(key=lambda x: x["score"], reverse=True)
    return [s for s in scored if s["score"] >= 3][:2]


# ═══════════════════════════════════════════════════════════
#  运行记录（JSONL 日志）
# ═══════════════════════════════════════════════════════════

# 进程级计数器，确保 run_id 在同一进程内不重复
_run_counter: int = 0


def _generate_run_id() -> str:
    """生成唯一 run_id。

    格式：run_YYYYMMDD_HHMMSS_<uuid8>_<counter>
    使用北京时间戳 + UUID 前 8 位 + 进程内计数器。

    Returns:
        str: 唯一运行标识符。
    """
    global _run_counter
    _run_counter += 1
    now = datetime.now(_TZ_BEIJING)
    ts = now.strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"run_{ts}_{short_uuid}_{_run_counter:04d}"


def _write_run_log(record: Dict[str, Any]) -> Dict[str, Any]:
    """向 week07/runs/agent_runs.jsonl 追加一行 JSON 运行记录。

    日志写入失败不影响主回答；返回写入状态。

    Args:
        record: 运行记录字典，至少包含时间、run_id、用户输入等字段。

    Returns:
        dict: {"ok": True/False, "path": str, "error": str}
    """
    runs_dir = _resolve_runs_path()
    log_path = _resolve_runs_file_path()

    # 尝试创建目录
    try:
        runs_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return {"ok": False, "path": str(log_path), "error": f"无法创建日志目录: {e}"}

    # 追加写入一行 JSON
    try:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
        return {"ok": True, "path": str(log_path), "error": ""}
    except OSError as e:
        return {"ok": False, "path": str(log_path), "error": f"日志写入失败: {e}"}
    except Exception as e:
        return {"ok": False, "path": str(log_path), "error": f"日志写入异常: {e}"}


def _build_run_record(
    run_id: str,
    user_input: str,
    route: str,
    tool_called: bool,
    tool_status: str,
    source_info: str,
    sections_hit: str,
    matched_terms: str,
    error_info: str,
    answer_summary: str,
    log_ok: bool,
) -> Dict[str, Any]:
    """构建一条完整的运行记录字典。

    Args:
        run_id: 唯一运行 ID。
        user_input: 用户输入。
        route: 路由类型。
        tool_called: 是否尝试调用工具。
        tool_status: 工具状态（是/否/是，但失败）。
        source_info: 来源信息。
        sections_hit: 命中的章节。
        matched_terms: 命中词。
        error_info: 错误信息。
        answer_summary: 回答摘要（前 200 字）。
        log_ok: 日志写入是否成功。

    Returns:
        dict: 运行记录。
    """
    now = datetime.now(_TZ_BEIJING)
    return {
        "时间": now.strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "用户输入": user_input[:500],
        "路由": route,
        "是否调用工具": tool_called,
        "工具状态": tool_status,
        "来源": source_info,
        "章节": sections_hit,
        "命中词": matched_terms,
        "错误信息": error_info,
        "回答摘要": answer_summary[:200],
        "日志写入状态": "成功" if log_ok else "失败",
    }


def _finalize_answer(
    *,
    run_id: str,
    user_input: str,
    route: str,
    conclusion: str,
    evidence: str,
    tool_status: str,
    source_info: str = "",
    sections_hit: str = "",
    matched_terms: str = "",
    error_info: str = "",
) -> str:
    """写入一次运行日志，并生成带真实写入状态的四段式回答。"""
    safe_source = _relative_source_path(source_info) if source_info else ""
    safe_error = _sanitize_error_message(error_info, "")
    record = _build_run_record(
        run_id=run_id,
        user_input=user_input,
        route=route,
        tool_called=tool_status != "否",
        tool_status=tool_status,
        source_info=safe_source,
        sections_hit=sections_hit,
        matched_terms=matched_terms,
        error_info=safe_error,
        answer_summary=conclusion,
        log_ok=True,
    )
    log_result = _write_run_log(record)
    if log_result.get("ok"):
        run_record = f"run_id={run_id}，日志已写入 week07/runs/{_RUNS_FILENAME}"
    else:
        error = _sanitize_error_message(log_result.get("error", ""), "日志写入失败")
        run_record = f"run_id={run_id}，日志写入失败：{error}"
    return format_final_answer(conclusion, evidence, tool_status, run_record)


# ═══════════════════════════════════════════════════════════
#  回答构建器（确定性，不依赖 LLM）
# ═══════════════════════════════════════════════════════════

def _make_source_ref(source_path: str, heading: str) -> str:
    """构建可读的来源引用字符串。

    Args:
        source_path: 相对来源路径。
        heading: 章节标题。

    Returns:
        str: 如 "week07/local_knowledge.md#提示词模板"
    """
    return f"{source_path}#{heading}"


def _build_direct_answer(
    user_input: str,
    run_id: str,
    log_write_result: Dict[str, Any],
) -> str:
    """构建直接回答（问候、帮助、能力说明）。

    Args:
        user_input: 用户输入。
        run_id: 运行 ID。
        log_write_result: 日志写入结果。

    Returns:
        str: 四段式完整回答。
    """
    text_lower = user_input.strip().lower()

    # 判断具体类型
    is_greeting = any(g in text_lower for g in ["你好", "嗨", "hello", "hi"])
    is_help = any(h in text_lower for h in [
        "帮助", "你能做什么", "你可以做什么", "你会做什么",
        "你能帮我", "你能帮助", "有什么功能", "功能有哪些",
        "使用方式", "怎么用", "如何使用", "你的能力", "能力范围",
        "介绍一下你自己", "你是谁", "你是什么",
    ])
    is_empty = not user_input or not user_input.strip()

    if is_empty:
        conclusion = (
            "请输入有效问题或任务说明。你可以询问大模型入门课程资料中的概念，"
            "例如：提示词模板、上下文窗口、RAG与本地检索、Agent与工具调用等。"
        )
        evidence = "输入为空，未调用本地资料工具，直接返回输入提示。"
    elif is_greeting or is_help:
        conclusion = (
            "你好！我是 Week07 中文资料问答 Agent，一个专注于大模型入门课程资料的本地助手。\n\n"
            "我能帮助你：\n"
            "1. 回答大模型入门课程资料中的概念问题，如提示词模板、上下文窗口、RAG 与本地检索、Agent 与工具调用等。\n"
            "2. 在回答时提供资料来源和依据（文件、章节、命中词），你可以追溯到具体资料。\n"
            "3. 在资料未覆盖时诚实说明范围限制，不虚构内容。\n\n"
            "我不能：联网搜索、读取其他文件、执行命令、进行复杂工具规划。\n\n"
            "使用方法：直接输入课程资料相关的问题即可，例如「根据课程资料，提示词模板由哪些部分组成？」。"
        )
        evidence = "未调用本地资料工具。这是一条帮助/能力说明类问题，由确定性路由规则直接回答。"
    else:
        conclusion = (
            "你好！我是 Week07 中文资料问答 Agent。"
            "我可以回答大模型入门课程资料中的概念问题。"
            "请直接提出你的问题，或者输入「帮助」查看具体使用说明。"
        )
        evidence = "未调用本地资料工具。输入被路由为直接回答，由确定性规则生成回复。"

    log_ok = log_write_result.get("ok", False)
    log_path = log_write_result.get("path", "")
    log_error = log_write_result.get("error", "")

    if log_ok:
        run_record_text = f"run_id={run_id}，日志已写入 {Path(log_path).name}"
    else:
        run_record_text = f"run_id={run_id}，日志写入失败：{log_error}"

    return format_final_answer(
        conclusion=conclusion,
        evidence=evidence,
        tool_status="否",
        run_record=run_record_text,
    )


def _build_retrieval_answer(
    user_input: str,
    sections: List[Dict[str, Any]],
    search_results: List[Dict[str, Any]],
    source_path: str,
    run_id: str,
    log_write_result: Dict[str, Any],
) -> str:
    """构建基于检索结果的回答。

    Args:
        user_input: 用户输入。
        sections: 所有资料片段。
        search_results: 检索命中的片段。
        source_path: 资料文件路径。
        run_id: 运行 ID。
        log_write_result: 日志写入结果。

    Returns:
        str: 四段式完整回答。
    """
    # 计算相对来源路径
    try:
        rel_source = str(Path(source_path).relative_to(Path(__file__).parent.parent))
    except ValueError:
        rel_source = Path(source_path).name

    log_ok = log_write_result.get("ok", False)
    log_path = log_write_result.get("path", "")

    if not search_results:
        # 资料可读但无命中
        available_topics = "、".join([s["heading"] for s in sections if s["heading"] != "前言"])
        conclusion = (
            f"当前资料未覆盖该问题。本资料库目前包含以下主题：{available_topics}。"
            f"请尝试询问上述主题相关的问题，或查阅「资料来源说明」中的外部链接进行延伸学习。"
        )
        evidence = (
            f"已读取本地资料 {rel_source}，共 {len(sections)} 个片段；"
            f"使用本地关键词检索，未找到与问题相关的章节。"
        )
        tool_status = "是"
        source_info = rel_source
        sections_hit = ""
        matched_terms_str = ""
        error_info = ""
        answer_summary = "资料未覆盖"

        run_record_text = (
            f"run_id={run_id}，日志已写入 {Path(log_path).name}"
            if log_ok else
            f"run_id={run_id}，日志写入失败：{log_write_result.get('error', '')}"
        )

        # 构建命中词信息
        cleaned = _clean_query(user_input)
        matched_terms_str = "无命中"
    else:
        # 有命中结果
        top_hits = search_results[:2]
        source_refs = []
        all_matched: List[str] = []
        conclusion_parts: List[str] = []

        for i, hit in enumerate(top_hits):
            ref = _make_source_ref(rel_source, hit["heading"])
            source_refs.append(ref)
            matched = hit.get("matched_terms", [])
            all_matched.extend(matched)

            # 从正文截取摘要（前 300 字）
            body_summary = hit["body"][:300]
            if len(hit["body"]) > 300:
                body_summary += "..."

            conclusion_parts.append(
                f"【{hit['heading']}】\n{body_summary}"
            )

        conclusion = "\n\n".join(conclusion_parts)
        matched_unique = list(dict.fromkeys(all_matched))  # 去重保序
        matched_terms_str = "、".join(matched_unique[:10])

        evidence = (
            f"本地来源：{'、'.join(source_refs)}；"
            f"检索方式：本地关键词检索（标题命中优先 ×3，正文命中次数 ×1）；"
            f"命中词：{matched_terms_str}。"
        )
        tool_status = "是"
        source_info = rel_source
        sections_hit = "、".join([h["heading"] for h in top_hits])
        error_info = ""
        answer_summary = conclusion[:200]
        matched_terms_str = matched_terms_str

        run_record_text = (
            f"run_id={run_id}，日志已写入 {Path(log_path).name}"
            if log_ok else
            f"run_id={run_id}，日志写入失败：{log_write_result.get('error', '')}"
        )

    return format_final_answer(
        conclusion=conclusion,
        evidence=evidence,
        tool_status=tool_status,
        run_record=run_record_text,
    )


def _build_tool_failure_answer(
    user_input: str,
    error: str,
    run_id: str,
    log_write_result: Dict[str, Any],
) -> str:
    """构建工具失败时的降级回答。

    Args:
        user_input: 用户输入。
        error: 工具失败错误信息。
        run_id: 运行 ID。
        log_write_result: 日志写入结果。

    Returns:
        str: 四段式完整回答。
    """
    log_ok = log_write_result.get("ok", False)
    log_path = log_write_result.get("path", "")

    conclusion = (
        f"未获取到资料。以下是当前可回答部分：该问题命中了本地资料检索条件，"
        f"但读取本地知识文件时发生错误。请检查 week07/local_knowledge.md 是否存在且可读。"
    )
    evidence = f"本地工具读取失败：{error}。Agent 按失败规则返回降级结果，不虚构资料来源。"
    tool_status = "是，但失败"

    if log_ok:
        run_record_text = f"run_id={run_id}，日志已写入 {Path(log_path).name}"
    else:
        run_record_text = f"run_id={run_id}，日志写入失败：{log_write_result.get('error', '')}"

    return format_final_answer(
        conclusion=conclusion,
        evidence=evidence,
        tool_status=tool_status,
        run_record=run_record_text,
    )


def _build_out_of_scope_answer(
    user_input: str,
    run_id: str,
    log_write_result: Dict[str, Any],
) -> str:
    """构建范围外回答。

    Args:
        user_input: 用户输入。
        run_id: 运行 ID。
        log_write_result: 日志写入结果。

    Returns:
        str: 四段式完整回答。
    """
    log_ok = log_write_result.get("ok", False)
    log_path = log_write_result.get("path", "")

    conclusion = (
        "抱歉，这个问题超出了我的资料范围。\n\n"
        "我是 Week07 中文资料问答 Agent，只回答与大模型入门课程资料相关的问题，"
        "包括：提示词模板、上下文窗口、RAG 与本地检索、Agent 与工具调用等主题。\n\n"
        "我不能联网搜索，不能回答课程资料之外的问题。"
        "如果你有课程资料相关的疑问，请尝试换个方式提问。"
    )
    evidence = (
        "未调用本地资料工具。该问题被路由为「范围外」，"
        "不在大模型入门课程资料覆盖范围内。确定性规则直接返回边界说明。"
    )
    tool_status = "否"

    if log_ok:
        run_record_text = f"run_id={run_id}，日志已写入 {Path(log_path).name}"
    else:
        run_record_text = f"run_id={run_id}，日志写入失败：{log_write_result.get('error', '')}"

    return format_final_answer(
        conclusion=conclusion,
        evidence=evidence,
        tool_status=tool_status,
        run_record=run_record_text,
    )


# ═══════════════════════════════════════════════════════════
#  format_final_answer — 统一格式化（四段式）
# ═══════════════════════════════════════════════════════════

def format_final_answer(
    conclusion: str,
    evidence: str,
    tool_status: str,
    run_record: str = "",
) -> str:
    """对最终结果做格式化，确保输出始终包含固定四段标题。

    Args:
        conclusion: 结论内容。
        evidence: 依据内容。
        tool_status: 工具调用状态，"是" / "否" / "是，但失败"，
                     非法值会被修正为 "否"。
        run_record: 运行记录内容，默认为空字符串。

    Returns:
        str: 固定四段格式的回答。
    """
    if tool_status not in _VALID_TOOL_STATUSES:
        tool_status = "否"

    if not run_record:
        run_record = "无运行记录"

    return (
        f"{_SECTION_CONCLUSION}\n"
        f"{conclusion}\n\n"
        f"{_SECTION_EVIDENCE}\n"
        f"{evidence}\n\n"
        f"{_SECTION_TOOL_STATUS}\n"
        f"{tool_status}\n\n"
        f"{_SECTION_RUN_RECORD}\n"
        f"{run_record}"
    )


# ═══════════════════════════════════════════════════════════
#  兼容层：build_agent_prompt（保留供外部调用）
# ═══════════════════════════════════════════════════════════

def build_agent_prompt(
    user_input: str,
    tool_result: Optional[Dict[str, Any]],
    need_tool: bool,
) -> str:
    """将用户问题、工具调用状态和工具结果组织成 prompt 文本。

    保留以兼容 Week06 的调用方式。Week07 默认路径不使用此函数。

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
        "请严格按照以下四段格式输出，不要添加额外的解释或前缀：\n\n"
        "结论：\n"
        "<面向用户的最终回答>\n\n"
        "依据：\n"
        "<说明回答依据，包括是否使用了本地资料>\n\n"
        "是否调用工具：\n"
        "<是 / 否 / 是，但失败>\n\n"
        "运行记录：\n"
        "<运行记录信息>\n"
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
#  run_agent — 主执行入口（课程问答默认模型路径）
# ═══════════════════════════════════════════════════════════

def run_agent(
    user_input: str,
    provider: str = _DEFAULT_PROVIDER,
    device: str = _DEFAULT_DEVICE,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    simulate_tool_failure: bool = False,
) -> str:
    """执行 Agent 流程并返回最终回答。

    课程问答路径：
      1. 输入验证与意图识别
      2. 本地资料检索、重排和上下文组装
      3. 调用所选模型生成结论正文
      4. 本地补充引用、工具状态和运行记录

    问候、帮助和范围外说明使用固定回复；模型故障返回明确的四段式失败结果。

    该函数不会抛出异常，所有错误均在返回值中体现。

    Args:
        user_input: 用户输入的自然语言问题。
        provider: 模型后端，"local" (Ollama) 或 "cloud" (DeepSeek)。
        device: 本地推理设备，"gpu" 或 "cpu"。
        temperature: 生成温度 0~2。
        max_tokens: 最大输出 token 数。
        simulate_tool_failure: 模拟工具读取失败（仅本次调用，不改写真实文件）。

    Returns:
        str: 固定四段格式的最终回答。
    """
    _set_last_token_usage(_empty_usage(), "none")
    global _run_counter

    # ── Step 1: 生成 run_id ──
    run_id = _generate_run_id()

    # ── Step 2: 输入验证 ──
    if not user_input or not user_input.strip():
        return _finalize_answer(
            run_id=run_id, user_input=user_input or "", route="direct_answer",
            conclusion="请输入有效问题或任务说明。你可以询问提示词模板、上下文窗口、RAG 或 Agent 工具调用等课程主题。",
            evidence="输入为空，未调用本地资料工具。", tool_status="否",
        )

    # ── Step 3: 路由判断（确定性）──
    route = _route_input(user_input)

    # ── Step 4: 直接回答路径 ──
    if route == "direct_answer":
        return _finalize_answer(
            run_id=run_id, user_input=user_input, route="direct_answer",
            conclusion=(
                "你好！我是 Week07 中文资料问答 Agent。我可以回答提示词模板、上下文窗口、"
                "RAG 与本地检索、Agent 与工具调用等课程问题。"
            ),
            evidence="未调用本地资料工具。这是一条问候、帮助或能力说明类问题。",
            tool_status="否",
        )

    # ── Step 5: 范围外路径 ──
    if route == "out_of_scope":
        return _finalize_answer(
            run_id=run_id, user_input=user_input, route="out_of_scope",
            conclusion="抱歉，这个问题超出了我的课程资料范围。我不能联网查询或核验课程资料之外的信息。",
            evidence="未调用本地资料工具。该问题被确定性规则识别为范围外。",
            tool_status="否",
        )

    # ── Step 6: 本地检索路径 ──
    # Step 6a: 模拟工具失败
    if simulate_tool_failure:
        error_msg = (
            "[模拟工具失败] 根据 --simulate-tool-failure 开关，"
            "本次调用模拟本地知识文件读取失败。真实文件未被修改。"
        )
        return _finalize_answer(
            run_id=run_id, user_input=user_input, route="local_retrieval",
            conclusion="未获取到资料。本次调用按参数要求模拟本地资料读取失败。",
            evidence=f"本地工具读取失败：{error_msg}", tool_status="是，但失败",
            error_info=error_msg,
        )

    # Step 6b: 真实读取资料文件
    tool_result = read_local_knowledge()

    if not tool_result["ok"]:
        # 真实读取失败
        error_msg = _sanitize_error_message(tool_result.get("error", ""), "本地资料读取失败")
        return _finalize_answer(
            run_id=run_id, user_input=user_input, route="local_retrieval",
            conclusion="未获取到资料。请检查本地课程资料是否存在且可读。",
            evidence=f"本地工具读取失败：{error_msg}", tool_status="是，但失败",
            source_info=str(tool_result.get("source", "")), error_info=error_msg,
        )

    # Step 6c: 资料读取成功，切分并检索
    content = tool_result["content"]
    source_path = tool_result["source"]

    # 处理空资料
    if not content or not content.strip():
        return _finalize_answer(
            run_id=run_id, user_input=user_input, route="local_retrieval",
            conclusion="当前资料文件为空，无法进行检索。",
            evidence="已读取本地课程资料，但文件内容为空。",
            tool_status="是", source_info=str(source_path), error_info="资料文件为空",
        )

    sections = _split_sections(content, source_path)

    # 处理无有效片段
    if not sections:
        return _finalize_answer(
            run_id=run_id, user_input=user_input, route="local_retrieval",
            conclusion="当前资料文件可读，但无法从中提取有效内容片段。",
            evidence="已读取本地课程资料，但按 Markdown 二级标题切分后未获得有效片段。",
            tool_status="是", source_info=str(source_path), error_info="无法从资料中提取有效片段",
        )

    # 关键词检索
    search_results = _search_sections(user_input, sections)

    sections_hit = "、".join([h["heading"] for h in search_results]) if search_results else ""
    matched_terms_list: List[str] = []
    for sr in search_results:
        matched_terms_list.extend(sr.get("matched_terms", []))
    matched_terms_str = "、".join(list(dict.fromkeys(matched_terms_list))[:10])

    if not search_results:
        return _finalize_answer(
            run_id=run_id, user_input=user_input, route="local_retrieval",
            conclusion="当前资料未覆盖该问题，无法组织可供模型生成答案的课程上下文。",
            evidence=(
                f"已读取本地资料 {_relative_source_path(str(source_path))}；"
                "使用本地关键词检索，但未找到相关章节。"
            ),
            tool_status="是", source_info=str(source_path),
            error_info="资料未命中", matched_terms="无命中",
        )

    model_result = _generate_model_answer(
        user_input=user_input,
        search_results=search_results,
        provider=provider,
        device=device,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    source_refs = "、".join(
        _make_source_ref(_relative_source_path(str(source_path)), hit["heading"])
        for hit in search_results
    )
    evidence = (
        f"本地来源：{source_refs}；"
        "检索方式：本地关键词检索（标题命中优先 ×3，正文命中次数 ×1）；"
        f"命中词：{matched_terms_str}。"
    )

    if model_result["ok"]:
        return _finalize_answer(
            run_id=run_id, user_input=user_input, route="local_retrieval",
            conclusion=model_result["answer"], evidence=evidence, tool_status="是",
            source_info=str(source_path), sections_hit=sections_hit,
            matched_terms=matched_terms_str,
        )

    model_error = _sanitize_error_message(model_result.get("error", ""), "模型服务不可用")
    return _finalize_answer(
        run_id=run_id, user_input=user_input, route="local_retrieval",
        conclusion="模型服务不可用，未生成最终答案。请检查 API Key、模型服务或 provider 配置后重试。",
        evidence=f"{evidence} 模型调用失败：{model_error}。",
        tool_status="是", source_info=str(source_path), sections_hit=sections_hit,
        matched_terms=matched_terms_str, error_info=model_error,
    )


def _generate_model_answer(
    user_input: str,
    search_results: List[Dict[str, Any]],
    provider: str,
    device: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Any]:
    """使用模型基于重排后的课程片段生成“结论”正文。

    本函数只负责模型生成；引用、工具状态和运行记录由调用方统一补充。

    Args:
        user_input: 用户原始输入。
        search_results: 检索命中的片段列表。
        provider: 模型后端。
        device: 推理设备。
        temperature: 生成温度。
        max_tokens: 最大输出 token 数。

    Returns:
        dict: ``ok``、``answer`` 和 ``error`` 字段。
    """
    try:
        from week05.qa_assistant_structured import (  # type: ignore
            build_chat_chain,
            parse_structured_response,
        )
    except Exception:
        return {"ok": False, "answer": "", "error": "模型依赖不可用"}

    if not search_results:
        return {"ok": False, "answer": "", "error": "没有可供模型生成的资料片段"}

    # 构建上下文
    context_parts = []
    for hit in search_results[:2]:
        context_parts.append(f"【{hit['heading']}】\n{hit['body'][:500]}")
    context = "\n\n".join(context_parts)

    system = (
        "你是 Week07 中文资料问答 Agent。"
        "只能依据提供的课程资料片段回答，不补充片段之外的事实。"
        "只输出面向用户的中文答案正文；不要输出结论、依据、引用、工具状态、运行记录或 Markdown 标题。"
    )

    prompt = (
        f"用户问题：{user_input}\n\n"
        "以下是经过本地检索和重排的课程资料片段：\n"
        f"--- 资料开始 ---\n{context}\n--- 资料结束 ---\n\n"
        "请只输出基于上述资料的答案正文。"
    )

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
        raw_output = str(response.content if hasattr(response, "content") else response).strip()
        parsed = parse_structured_response(raw_output)
        output = str(parsed.get("answer") or raw_output).strip()
        if output:
            _set_last_token_usage(
                _extract_response_usage(response, provider), "week05"
            )
            return {"ok": True, "answer": output, "error": ""}
    except Exception:
        return {"ok": False, "answer": "", "error": "模型服务调用失败"}

    return {"ok": False, "answer": "", "error": "模型未返回有效答案"}


# ═══════════════════════════════════════════════════════════
#  辅助函数
# ═══════════════════════════════════════════════════════════

def _has_four_sections(text: str) -> bool:
    """检查文本是否包含四段标题。"""
    if not text:
        return False
    markers = [_SECTION_CONCLUSION, _SECTION_EVIDENCE,
               _SECTION_TOOL_STATUS, _SECTION_RUN_RECORD]
    for m in markers:
        if not re.search(rf"(^|\n)\s*{re.escape(m)}", text):
            return False
    return True


def _extract_section(text: str, start_marker: str, end_marker: str) -> str:
    """从四段式文本中提取指定段落的内容。

    Args:
        text: 四段式完整文本。
        start_marker: 段落起始标题（如"依据："）。
        end_marker: 下一段落标题（如"是否调用工具："）。

    Returns:
        str: 段落内容，未找到时返回空字符串。
    """
    pattern = rf"{re.escape(start_marker)}\s*\n?(.*?)\n?\s*{re.escape(end_marker)}"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def _strip_run_record_section(text: str) -> str:
    """去掉文本末尾的「运行记录：」段落，避免重复。

    Args:
        text: 可能包含运行记录段的文本。

    Returns:
        str: 去掉运行记录段后的文本。
    """
    pattern = rf"\n*\s*{re.escape(_SECTION_RUN_RECORD)}\s*.*$"
    return re.sub(pattern, "", text, flags=re.DOTALL).rstrip()


def _empty_usage(source: str = "none") -> Dict[str, Any]:
    """构造空 token 用量结构。"""
    return {"prompt": 0, "completion": 0, "total": 0, "source": source}


def _set_last_token_usage(usage: Optional[Dict[str, Any]], source: str) -> None:
    """记录最近一轮的 token 用量。"""
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


def _format_terminal_text(text: str, width: Optional[int] = None) -> str:
    """仅为 CLI 展示按终端宽度折行，不改变 Agent 的原始返回值。"""
    effective_width = width if width is not None else max(
        60, shutil.get_terminal_size(fallback=(88, 24)).columns
    )
    section_titles = {
        _SECTION_CONCLUSION,
        _SECTION_EVIDENCE,
        _SECTION_TOOL_STATUS,
        _SECTION_RUN_RECORD,
    }
    output_lines: List[str] = []
    for line in text.splitlines():
        if not line or line in section_titles:
            output_lines.append(line)
            continue
        output_lines.extend(
            textwrap.wrap(
                line,
                width=effective_width,
                break_long_words=True,
                break_on_hyphens=False,
            ) or [""]
        )
    return "\n".join(output_lines)


# ═══════════════════════════════════════════════════════════
#  stream_agent — 流式输出
# ═══════════════════════════════════════════════════════════

def stream_agent(
    user_input: str,
    provider: str = _DEFAULT_PROVIDER,
    device: str = _DEFAULT_DEVICE,
    temperature: float = _DEFAULT_TEMPERATURE,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    chunk_size: int = 12,
    simulate_tool_failure: bool = False,
) -> Iterator[str]:
    """以文本块形式流式返回 Agent 最终回答。

    当前实现保持 run_agent() 的稳定四段式契约，先生成最终回答，
    再按固定大小切片输出，适合作为 CLI 流式显示入口。

    Args:
        user_input: 用户输入。
        provider: 模型后端。
        device: 推理设备。
        temperature: 生成温度。
        max_tokens: 最大输出 token 数。
        chunk_size: 每块字符数。
        simulate_tool_failure: 模拟工具失败。

    Yields:
        str: 文本块。
    """
    answer = run_agent(
        user_input=user_input,
        provider=provider,
        device=device,
        temperature=temperature,
        max_tokens=max_tokens,
        simulate_tool_failure=simulate_tool_failure,
    )
    safe_chunk_size = max(1, chunk_size)
    for i in range(0, len(answer), safe_chunk_size):
        yield answer[i:i + safe_chunk_size]


def _print_stream(chunks: Iterator[str]) -> None:
    """打印流式回答，并统一按终端宽度折行。"""
    answer = "".join(chunks)
    print(_format_terminal_text(answer))


# ═══════════════════════════════════════════════════════════
#  CLI — 交互模式 UI（兼容 Week06 风格）
# ═══════════════════════════════════════════════════════════

_BANNER = """
╔══════════════════════════════════════════════════════════╗
║   Week07 中文资料问答 Agent                               ║
║   大模型入门课程资料助手（课程问答使用模型服务）             ║
║                                                          ║
║   课程问题会检索资料并调用所选模型生成回答。                ║
║   Agent 拥有 1 个工具：read_local_knowledge               ║
║   输出固定四段格式：                                      ║
║     结论 / 依据 / 是否调用工具 / 运行记录                  ║
║                                                          ║
║   会话指令:                                                ║
║     /help       显示帮助     /provider   切换后端          ║
║     /device     切换设备     /stream     切换流式输出      ║
║     /stats      查看当前配置                              ║
║     q/quit/exit  退出对话                                 ║
╚══════════════════════════════════════════════════════════╝"""


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


def _print_agent_status(
    provider: str,
    device: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
    provider_info: Dict[str, Dict[str, str]],
) -> None:
    """打印当前 Agent 会话状态。"""
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


def _handle_agent_command(
    cmd_line: str,
    provider: str,
    device: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
    provider_info: Dict[str, Dict[str, str]],
) -> dict:
    """解析并执行 / 开头的会话指令。"""
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

    print(f"[HELP] 未知指令 '{cmd}'。输入 /help 查看可用指令。")
    return {"action": "skip"}


def _interactive_session() -> None:
    """交互模式主循环。"""
    provider = _DEFAULT_PROVIDER
    device = _DEFAULT_DEVICE
    temperature = _DEFAULT_TEMPERATURE
    max_tokens = _DEFAULT_MAX_TOKENS
    stream = _DEFAULT_STREAM

    provider_info = _load_provider_info()

    print(_BANNER)
    _print_agent_status(
        provider, device, temperature, max_tokens, stream, provider_info
    )

    turn = 0
    total_tokens = 0
    while True:
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

        if user_input.startswith("/"):
            result = _handle_agent_command(
                user_input, provider, device, temperature, max_tokens,
                stream, provider_info,
            )
            provider = result.get("provider", provider)
            device = result.get("device", device)
            stream = result.get("stream", stream)
            continue

        turn += 1

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
            print(_format_terminal_text(result))
        usage = get_last_token_usage()
        total_tokens += int(usage.get("total", 0) or 0)
        print(_format_token_usage(usage))

    info = provider_info.get(provider, {})
    print(f"\n[SUMMARY] 后端: {info.get('label', provider)}  |  "
          f"对话 {turn} 轮  |  累计 Token: {total_tokens}")


# ═══════════════════════════════════════════════════════════
#  main — CLI 入口
# ═══════════════════════════════════════════════════════════

def main() -> None:
    """CLI 入口。

    支持两种模式：
      1. 单次模式：
         python week07/simple_agent.py [--no-stream] [--simulate-tool-failure] "问题文本"
      2. 交互模式：
         python week07/simple_agent.py（无参数启动）

    新增参数：
      --simulate-tool-failure  模拟工具读取失败（仅本次调用，不改写真实文件）
    """
    args = sys.argv[1:]

    provider = _DEFAULT_PROVIDER
    stream = _DEFAULT_STREAM
    simulate_tool_failure = False
    question_parts: List[str] = []
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
        elif arg == "--simulate-tool-failure":
            simulate_tool_failure = True
        else:
            question_parts.append(arg)

    user_input = " ".join(question_parts)

    if not user_input:
        _interactive_session()
    else:
        if stream:
            _print_stream(stream_agent(
                user_input,
                provider=provider,
                simulate_tool_failure=simulate_tool_failure,
            ))
        else:
            result = run_agent(
                user_input,
                provider=provider,
                simulate_tool_failure=simulate_tool_failure,
            )
            print(_format_terminal_text(result))
        print(_format_token_usage())


if __name__ == "__main__":
    main()
