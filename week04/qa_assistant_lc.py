"""
CLI 中文问答助手 (LangChain 版) — 支持本地 Ollama (CPU/GPU) + 云端 DeepSeek 双后端

Week04 重构: 用 LangChain 核心组件替代手写 HTTP 请求和消息拼接。

原始版本: week03/qa_assistant.py
重构映射:
  手写 messages         → ChatPromptTemplate
  手写 HTTP 请求        → ChatOllama / ChatDeepSeek
  手写 call_api         → model.invoke()
  手写 call_api_stream  → model.stream()
  手写多轮 messages 累积 → history + MessagesPlaceholder

特性:
  - 单次问答: 命令行直接传入问题
  - 连续对话: 不传问题参数则进入交互模式, 保持多轮对话上下文
  - 双后端: 默认使用本地 Ollama (GPU), --provider cloud 切换 DeepSeek API
  - 本地双设备: --device gpu 启用GPU加速, --device cpu 纯CPU推理
  - 会话指令: /role /clear /stream /provider /device /help 等

使用方式:
  # 本地 Ollama (默认, GPU加速)
  python qa_assistant_lc.py "什么是机器学习？"
  python qa_assistant_lc.py --stream --role teacher "解释 REST API"

  # 本地 Ollama (CPU 推理)
  python qa_assistant_lc.py -d cpu "什么是机器学习？"

  # 云端 DeepSeek
  python qa_assistant_lc.py -p cloud "什么是机器学习？"
  python qa_assistant_lc.py -p cloud --stream --role teacher "解释量子计算"

  # 连续对话
  python qa_assistant_lc.py
  python qa_assistant_lc.py -p cloud --role coder
"""

import sys
import os
import time
import argparse
from typing import List, Optional, Dict
from datetime import datetime

# ── Windows 控制台 UTF-8 编码修复 ──────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

# ── LangChain 核心组件 ──────────────────────────────────
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, AIMessageChunk, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek

# ── 复用项目级 config 中的 API Key ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import API_KEY


# ═══════════════════════════════════════════════════════════
#  配置 (Config) — 所有可调参数集中在此
# ═══════════════════════════════════════════════════════════

# ── 云端 DeepSeek ──
CLOUD_API_BASE = "https://api.deepseek.com/v1"
CLOUD_MODEL = "deepseek-chat"

# ── 本地 Ollama ──
LOCAL_API_BASE = "http://localhost:11434"
LOCAL_MODEL = "llama3.2:1b"
DEFAULT_DEVICE = "gpu"                  # cpu | gpu
DEVICE_NUM_GPU = {                      # device → Ollama num_gpu 映射
    "cpu": 0,                           #   0 = 不加载任何 GPU 层 → 纯 CPU
    "gpu": -1,                          #  -1 = 自动加载全部 GPU 层
}

# ── 通用 ──
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_PROVIDER = "local"             # cloud | local

# ── 后端元信息 ──
PROVIDER_INFO = {
    "cloud": {"label": "DeepSeek (云端)", "model": CLOUD_MODEL, "base_url": CLOUD_API_BASE},
    "local": {"label": "Ollama (本地)",   "model": LOCAL_MODEL, "base_url": LOCAL_API_BASE},
}

# ── 角色提示词 (从 Week03 迁移, 保持完全一致) ──────────
ROLE_PROMPTS: Dict[str, str] = {
    "default": (
        "你是一个智能问答助手，擅长用中文清晰、准确地回答问题。"
        "如果问题涉及复杂概念，请用通俗易懂的语言解释。"
    ),
    "teacher": (
        "你是一位经验丰富的中文教师。请用通俗易懂的语言讲解概念，"
        "配合生活化的例子和类比，帮助学生真正理解。"
        "回答控制在 300 字以内。"
    ),
    "coder": (
        "你是一位资深软件工程师。请用中文回答技术问题，"
        "提供清晰的代码示例，并解释关键逻辑。"
        "代码块使用 Markdown 格式。"
    ),
    "doctor": (
        "你是一位医学顾问。请用中文提供专业的健康建议，"
        "引用权威医学知识，并始终提醒用户必要时咨询医生。"
        "回答应清晰、准确、有依据。"
    ),
    "translator": (
        "你是一位专业的中英翻译。请准确翻译用户提供的内容，"
        "保持原文风格和专业术语的一致性。"
        "如果是中文翻译成英文，输出英文；反之输出中文。"
    ),
}


# ═══════════════════════════════════════════════════════════
#  Token 用量提取 (Token Usage Extraction)
# ═══════════════════════════════════════════════════════════

def _extract_usage(response, provider: str) -> dict:
    """从 AIMessage / AIMessageChunk 中提取 token 用量。

    Ollama 将用量放在 ``response_metadata`` 中 (prompt_eval_count / eval_count)。
    DeepSeek (OpenAI 兼容) 放在 ``response_metadata.token_usage`` 或 ``usage_metadata`` 中。
    返回值: {"prompt": int, "completion": int, "total": int}，提取失败时返回全 0。
    """
    # 优先尝试 usage_metadata (LangChain 标准字段, 较新版本)
    um = getattr(response, "usage_metadata", None) or {}
    if um:
        return {
            "prompt":    um.get("input_tokens", 0),
            "completion": um.get("output_tokens", 0),
            "total":     um.get("total_tokens", 0),
        }

    # 回退到 response_metadata (各 provider 原始字段)
    rm = getattr(response, "response_metadata", None) or {}

    if provider == "local":
        prompt_tok = rm.get("prompt_eval_count", 0)
        comp_tok = rm.get("eval_count", 0)
        return {
            "prompt": prompt_tok,
            "completion": comp_tok,
            "total": prompt_tok + comp_tok,
        }

    # cloud (DeepSeek / OpenAI 兼容)
    tu = rm.get("token_usage", {})
    return {
        "prompt":    tu.get("prompt_tokens", 0),
        "completion": tu.get("completion_tokens", 0),
        "total":     tu.get("total_tokens", 0),
    }


def _extract_finish_reason(response, provider: str) -> str:
    """从 AIMessage 中提取结束原因, 提取失败时返回 "stop"。

    Ollama: response_metadata["done_reason"]
    DeepSeek: response_metadata["finish_reason"]
    """
    rm = getattr(response, "response_metadata", None) or {}
    if provider == "local":
        return rm.get("done_reason", "stop")
    return rm.get("finish_reason", "stop")


def _format_stats(model_name: str, elapsed: float, usage: dict,
                  finish_reason: str) -> str:
    """格式化底部统计行。"""
    return (f"--- {model_name} | {elapsed:.1f}s | "
            f"Token: prompt={usage['prompt']} completion={usage['completion']} "
            f"total={usage['total']} | 结束: {finish_reason} ---")


# ═══════════════════════════════════════════════════════════
#  LangChain 核心: Model 工厂
# ═══════════════════════════════════════════════════════════

def get_model(provider: str, temperature: float, max_tokens: int,
              device: str = "gpu") -> ChatOllama | ChatDeepSeek:
    """模型工厂 — 按 provider 返回对应的 LangChain ChatModel 实例

    LangChain 等价于 Week03 的 build_payload() + call_api() 中的 provider 分发。
    """
    if provider == "local":
        num_gpu = DEVICE_NUM_GPU.get(device, 0)
        return ChatOllama(
            model=LOCAL_MODEL,
            temperature=temperature,
            num_predict=max_tokens,
            num_gpu=num_gpu,
            base_url=LOCAL_API_BASE,
        )

    if provider == "cloud":
        return ChatDeepSeek(
            model_name=CLOUD_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=API_KEY,
            openai_api_base=CLOUD_API_BASE,
        )

    raise ValueError(f"未知 provider: {provider}")


# ═══════════════════════════════════════════════════════════
#  LangChain 核心: Prompt Template
# ═══════════════════════════════════════════════════════════

def build_system_text(role: str, custom_system: Optional[str] = None) -> str:
    """根据角色或自定义内容构建 system prompt 文本"""
    if custom_system:
        return custom_system
    return ROLE_PROMPTS.get(role, ROLE_PROMPTS["default"])


def build_prompt_template(role: str, custom_system: Optional[str] = None) -> ChatPromptTemplate:
    """单轮 Prompt Template — 无历史消息

    LangChain 等价于 Week03 的 build_messages()。
    """
    system_text = build_system_text(role, custom_system)
    return ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "{question}"),
    ])


def build_chat_prompt_template(role: str, custom_system: Optional[str] = None) -> ChatPromptTemplate:
    """多轮 Prompt Template — 带 MessagesPlaceholder 承载对话历史

    LangChain 等价于 Week03 中 interactive_session() 的 messages 列表累积。
    """
    system_text = build_system_text(role, custom_system)
    return ChatPromptTemplate.from_messages([
        ("system", system_text),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])


# ═══════════════════════════════════════════════════════════
#  LangChain 核心: Chain 构建 (LCEL)
# ═══════════════════════════════════════════════════════════

def build_single_shot_chain(provider: str, role: str, temperature: float,
                            max_tokens: int, device: str = "gpu",
                            custom_system: Optional[str] = None):
    """构建单次问答 Chain: prompt | model

    返回 AIMessage (含 response_metadata / usage_metadata)，调用方自行提取 content 和 token 用量。

    LangChain 等价于 Week03 的 single_shot() 完整流程。
    """
    prompt = build_prompt_template(role, custom_system)
    model = get_model(provider, temperature, max_tokens, device)
    return prompt | model


def build_chat_chain(provider: str, role: str, temperature: float,
                     max_tokens: int, device: str = "gpu",
                     custom_system: Optional[str] = None):
    """构建多轮对话 Chain: prompt | model

    返回 AIMessage (含 response_metadata / usage_metadata)，调用方自行提取 content 和 token 用量。
    Prompt 中包含 MessagesPlaceholder("history") 用于承载对话历史。
    """
    prompt = build_chat_prompt_template(role, custom_system)
    model = get_model(provider, temperature, max_tokens, device)
    return prompt | model


# ═══════════════════════════════════════════════════════════
#  Stage 1: 输入处理 (Input Processing)
# ═══════════════════════════════════════════════════════════

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数 — 与 Week03 保持一致的 CLI 接口"""
    parser = argparse.ArgumentParser(
        description="CLI 中文问答助手 (LangChain 版) — 支持本地 Ollama + 云端 DeepSeek 双后端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 本地 Ollama (默认, GPU加速)
  python qa_assistant_lc.py "什么是机器学习？"
  python qa_assistant_lc.py --stream --role teacher "解释量子计算"

  # 本地 Ollama (CPU 推理)
  python qa_assistant_lc.py -d cpu "什么是机器学习？"

  # 云端 DeepSeek
  python qa_assistant_lc.py -p cloud "什么是机器学习？"
  python qa_assistant_lc.py -p cloud --stream "写一首关于春天的诗"

  # 交互模式
  python qa_assistant_lc.py
  python qa_assistant_lc.py -p cloud --role coder
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
#  交互模式 UI
# ═══════════════════════════════════════════════════════════

_INTERACTIVE_BANNER = """
╔══════════════════════════════════════════════════════════╗
║   CLI 中文问答助手 (LangChain 版)                          ║
║   本地 Ollama CPU/GPU + 云端 DeepSeek 双后端              ║
║                                                        ║
║   直接输入问题即可开始对话，模型会记住上下文。               ║
║   会话指令:                                              ║
║     /help      显示帮助     /role <name>  切换角色        ║
║     /stream    切换流式     /clear        清空上下文       ║
║     /provider  切换后端     /device       切换CPU/GPU     ║
║     /save      保存记录     /stats        查看统计         ║
║     q/quit/exit  退出对话                                ║
╚══════════════════════════════════════════════════════════╝"""


def _print_status_line(provider: str, role: str, system: Optional[str],
                       stream: bool, device: str, temperature: float,
                       max_tokens: int) -> None:
    """打印当前会话状态的紧凑摘要行"""
    info = PROVIDER_INFO.get(provider, {})
    device_label = "GPU" if device == "gpu" else "CPU"
    role_label = f"自定义:{system[:20]}..." if system else role
    stream_label = "ON" if stream else "OFF"
    print(f"  [STATUS] 后端: {info.get('label', provider)} | "
          f"设备: {device_label} | 角色: {role_label} | "
          f"流式: {stream_label} | T: {temperature} | Max: {max_tokens}")


def get_user_input(prompt: str = "[?] 请输入: ") -> Optional[str]:
    """[交互模式] 获取一轮用户输入, 返回 None 表示退出"""
    try:
        user_input = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[BYE] 再见!")
        return None

    if not user_input:
        return ""
    if user_input.lower() in ("q", "quit", "exit"):
        print("[BYE] 再见!")
        return None
    return user_input


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
    """[交互模式] 解析并执行 / 开头的会话指令

    返回 dict, 调用方用 .get() 提取变更的字段。
    history 使用 LangChain 的 BaseMessage 列表, 而非原生 dict。
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
            # 更新 history 中的 system message
            if history and isinstance(history[0], SystemMessage):
                history[0] = SystemMessage(content=ROLE_PROMPTS[new_role])
        elif not new_role:
            print(f"[HELP] 可用角色: {', '.join(ROLE_PROMPTS.keys())}")
            print(f"[HELP] 当前角色: {role}")
        else:
            print(f"[ERROR] 未知角色 '{new_role}'。可用: {', '.join(ROLE_PROMPTS.keys())}")
        return {"action": "skip", "role": role, "system": system, "history": history}

    if cmd == "/system":
        new_system = arg.strip()
        if new_system:
            system = new_system
            role = "default"
            print(f"[OK] 自定义 system prompt 已设置")
            if history and isinstance(history[0], SystemMessage):
                history[0] = SystemMessage(content=new_system)
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
        # 保留 system message, 清空对话历史
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
                f.write(f"# 中文问答助手 (LangChain 版) — 会话记录\n")
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
#  Stage 3: 输出处理 — 保存结果
# ═══════════════════════════════════════════════════════════

def save_result(content: str, filepath: str, question: str,
                role: str, provider: str = DEFAULT_PROVIDER):
    """将问答结果保存到文件 (与 Week03 兼容)"""
    try:
        info = PROVIDER_INFO.get(provider, {})
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# 中文问答助手 (LangChain 版) — 问答记录\n")
            f.write(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 后端: {info.get('label', provider)} ({info.get('model', '?')})\n")
            f.write(f"# 角色: {role}\n")
            f.write(f"\n## 问题\n\n{question}\n\n")
            f.write(f"## 回答\n\n{content}\n")
        print(f"\n[SAVED] 回答已保存到: {os.path.abspath(filepath)}")
    except OSError as e:
        print(f"\n[ERROR] 保存文件失败: {e}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════
#  单次问答 (Single-shot) — LangChain 版
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
    """
    [单次模式] 一问一答 — LangChain LCEL 版

    等价于 Week03 的 single_shot(), 但用 chain.invoke() / chain.stream()
    替代手写 HTTP 请求和 payload 构建。
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
            last_usage_chunk = None  # ★ 追踪最后一个携带 usage_metadata 的 chunk
            for chunk in chain.stream({"question": question}):
                text = chunk.content if isinstance(chunk.content, str) else ""
                if text:
                    print(text, end="", flush=True)
                full_text += text
                last_chunk = chunk
                if getattr(chunk, "usage_metadata", None):
                    last_usage_chunk = chunk
            print()
            elapsed = time.perf_counter() - t0
            content = full_text
            # 优先使用携带 token 用量的 chunk (流式模式下可能在倒数第二个 chunk)
            stats_chunk = last_usage_chunk or last_chunk
            usage = _extract_usage(stats_chunk, provider) if stats_chunk else \
                {"prompt": 0, "completion": 0, "total": 0}
            finish = _extract_finish_reason(stats_chunk, provider) if stats_chunk else "stop"
        else:
            # ── 非流式输出 ──
            response = chain.invoke({"question": question})
            elapsed = time.perf_counter() - t0
            content = response.content if hasattr(response, "content") else str(response)
            usage = _extract_usage(response, provider)
            finish = _extract_finish_reason(response, provider)

            print("\n" + "=" * 60)
            print("  [回答]")
            print("=" * 60)
            print(content)

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
#  连续对话 (Interactive Session) — LangChain 版
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
    """
    [交互模式] 连续对话循环 — LangChain 版

    使用 MessagesPlaceholder + 手动维护 history 列表的方式承载上下文。
    每次调用 chain 时传入完整的 history。

    等价于 Week03 的 interactive_session(), 但:
      - 用 chain.invoke({"history": [...], "question": ...}) 替代手写 API 调用
      - history 使用 LangChain 的 BaseMessage 类型 (SystemMessage/HumanMessage/AIMessage)
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
                last_usage_chunk = None  # ★ 追踪最后一个携带 usage_metadata 的 chunk
                for chunk in chain.stream({
                    "history": history,
                    "question": user_input,
                }):
                    text = chunk.content if isinstance(chunk.content, str) else ""
                    if text:
                        print(text, end="", flush=True)
                    full_text += text
                    last_chunk = chunk
                    if getattr(chunk, "usage_metadata", None):
                        last_usage_chunk = chunk
                print()
                elapsed = time.perf_counter() - t0
                content = full_text
                # 优先使用携带 token 用量的 chunk (流式模式下可能在倒数第二个 chunk)
                stats_chunk = last_usage_chunk or last_chunk
                usage = _extract_usage(stats_chunk, provider) if stats_chunk else \
                    {"prompt": 0, "completion": 0, "total": 0}
                finish = _extract_finish_reason(stats_chunk, provider) if stats_chunk else "stop"
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

                print(f"\n[{session_stats['turns'] + 1}] [A]")
                print("-" * 40)
                print(content)

            # --- 模型调用成功后再写入本轮历史，避免当前问题在 prompt 中重复出现 ---
            history.append(HumanMessage(content=user_input))
            history.append(AIMessage(content=content))

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
#  错误处理
# ═══════════════════════════════════════════════════════════

def _handle_langchain_error(e: Exception, provider: str):
    """统一错误处理 — LangChain 版

    捕获 LangChain 底层抛出的异常 (通常是 requests 异常或 API 错误),
    输出与 Week03 一致的中文错误信息。
    """
    # 尝试获取底层异常
    cause = e.__cause__ or e

    error_type = type(cause).__name__
    error_msg = str(cause)

    # ── 连接错误 ──
    if error_type == "ConnectionError" or "Connection" in error_msg:
        label = PROVIDER_INFO.get(provider, {}).get("label", provider)
        hint = ("请确认 Ollama 已启动: ollama serve"
                if provider == "local" else "请检查网络连接")
        print(f"[ERROR] 无法连接到 {label}。{hint}", file=sys.stderr)
        return

    # ── 超时 ──
    if error_type == "Timeout" or "timeout" in error_msg.lower():
        print(f"[ERROR] 请求超时。", file=sys.stderr)
        return

    # ── HTTP 错误 ──
    if hasattr(cause, 'response'):
        status = getattr(cause.response, 'status_code', 0)
        if provider == "local":
            if status == 404:
                print(f"[ERROR] 模型 '{LOCAL_MODEL}' 未找到。运行 'ollama list' 查看。",
                      file=sys.stderr)
            else:
                print(f"[ERROR] Ollama HTTP {status}: {error_msg}", file=sys.stderr)
        else:
            error_map = {
                401: "鉴权失败 (401): API Key 无效或已过期。",
                402: "余额不足 (402)。",
                429: f"请求频率超限 (429): {error_msg}",
                500: f"服务端内部错误 (500): {error_msg}",
            }
            print(f"[ERROR] {error_map.get(status, f'HTTP {status}: {error_msg}')}",
                  file=sys.stderr)
        return

    # ── 通用错误 ──
    print(f"[ERROR] {error_type}: {error_msg}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════
#  主入口 (Main Entry)
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
    """
    统一入口 — 按是否有 CLI 问题参数分流:

        question 非空  →  single_shot_lc()
        question 为空  →  interactive_session_lc()
    """
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
    """CLI 入口"""
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
