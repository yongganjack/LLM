"""
CLI 中文问答助手 — 支持本地 (Ollama, CPU/GPU) + 云端 (DeepSeek) 双后端

特性:
  - 单次问答: 命令行直接传入问题
  - 连续对话: 不传问题参数则进入交互模式, 保持多轮对话上下文
  - 双后端: 默认使用本地 Ollama (GPU), --provider cloud 切换 DeepSeek API
  - 本地双设备: --device gpu 启用GPU加速, --device cpu 纯CPU推理
  - 会话指令: /role /clear /save /stream /provider /device /help 等

三阶段流水线 (为 LangChain 迁移而设计):
  Stage 1 — 输入处理:  命令行参数解析 → 输入校验 → 提示词模板构建
  Stage 2 — 模型调用:  统一接口 → 按 provider 分发到 Ollama 或 DeepSeek
  Stage 3 — 输出处理:  格式化打印 + token 统计 + 可选存档

使用方式:
  # 本地 Ollama (默认, GPU加速)
  python qa_assistant.py "什么是机器学习？"
  python qa_assistant.py --stream --role teacher "解释 REST API"

  # 本地 Ollama (CPU 推理)
  python qa_assistant.py -d cpu "什么是机器学习？"
  python qa_assistant.py -d cpu --stream "写一首关于春天的诗"

  # 云端 DeepSeek
  python qa_assistant.py -p cloud "什么是机器学习？"
  python qa_assistant.py -p cloud --stream --role teacher "解释 REST API"

  # 连续对话
  python qa_assistant.py
  python qa_assistant.py -p cloud --role coder
"""

import sys
import os
import time
import argparse
import json
from typing import List, Dict, Optional
from datetime import datetime

import requests

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
LOCAL_MODEL = "llama3.2:1b"            # 与 ollama list 一致
LOCAL_NUM_THREAD = None                 # None = 自动; 设为数字可指定 CPU 线程数
DEFAULT_DEVICE = "gpu"                  # cpu | gpu  (本地推理设备, 默认GPU)
DEVICE_NUM_GPU = {                      # device → Ollama num_gpu 映射
    "cpu": 0,                           #   0 = 不加载任何 GPU 层 → 纯 CPU
    "gpu": -1,                          #  -1 = 自动加载全部 GPU 层
}

# ── 通用 ──
REQUEST_TIMEOUT = 60
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_PROVIDER = "local"             # cloud | local (默认本地 GPU 推理)

# ── 后端元信息 ──
PROVIDER_INFO = {
    "cloud": {"label": "DeepSeek (云端)", "model": CLOUD_MODEL, "base_url": CLOUD_API_BASE},
    "local": {"label": "Ollama (本地)",   "model": LOCAL_MODEL, "base_url": LOCAL_API_BASE},
}

# 预设角色 → system prompt 映射 (可直接复用到 LangChain 的 ChatPromptTemplate)
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
#  Stage 1: 输入处理 (Input Processing)
# ═══════════════════════════════════════════════════════════

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数

    LangChain 等价物: 自定义输入 Schema / Pydantic model
    """
    parser = argparse.ArgumentParser(
        description="CLI 中文问答助手 — 支持本地 Ollama + 云端 DeepSeek 双后端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 本地 Ollama (默认, GPU加速)
  python qa_assistant.py "什么是机器学习？"
  python qa_assistant.py --stream --role teacher "解释量子计算"

  # 本地 Ollama (CPU 推理)
  python qa_assistant.py -d cpu "什么是机器学习？"
  python qa_assistant.py -d cpu --stream "写一首关于春天的诗"

  # 云端 DeepSeek
  python qa_assistant.py -p cloud "什么是机器学习？"
  python qa_assistant.py -p cloud --stream "写一首关于春天的诗"

  # 交互模式
  python qa_assistant.py
  python qa_assistant.py -p cloud --role coder
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


def validate_input(question: Optional[str]) -> str:
    """[单次模式] 校验 CLI 传入的问题"""
    if question and question.strip():
        return question.strip()
    return ""


# ── 交互模式 banner ──
_INTERACTIVE_BANNER = """
╔══════════════════════════════════════════════════════════╗
║     CLI 中文问答助手 (本地 Ollama CPU/GPU + 云端 DeepSeek) ║
║                                                        ║
║   直接输入问题即可开始对话，模型会记住上下文。               ║
║   默认使用本地模型 (GPU加速), -d cpu 切换 CPU 推理。       ║
║   会话指令:                                              ║
║     /help      显示帮助     /role <name>  切换角色        ║
║     /stream    切换流式     /clear        清空上下文       ║
║     /provider  切换后端     /device       切换CPU/GPU     ║
║     /save      保存记录     /stats        查看统计         ║
║     q/quit/exit  退出对话                                ║
╚══════════════════════════════════════════════════════════╝"""


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
    messages: List[Dict[str, str]],
    session_stats: dict,
) -> dict:
    """[交互模式] 解析并执行 / 开头的会话指令

    返回 dict, 调用方用 .get() 提取变更的字段
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
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = ROLE_PROMPTS[new_role]
        elif not new_role:
            print(f"[HELP] 可用角色: {', '.join(ROLE_PROMPTS.keys())}")
            print(f"[HELP] 当前角色: {role}")
        else:
            print(f"[ERROR] 未知角色 '{new_role}'。可用: {', '.join(ROLE_PROMPTS.keys())}")
        return {"action": "skip", "role": role, "system": system, "messages": messages}

    if cmd == "/system":
        new_system = arg.strip()
        if new_system:
            system = new_system
            role = "default"
            print(f"[OK] 自定义 system prompt 已设置")
            if messages and messages[0]["role"] == "system":
                messages[0]["content"] = new_system
        else:
            if system:
                print(f"[INFO] 当前自定义 system prompt: {system[:80]}...")
            else:
                print(f"[INFO] 当前使用角色预设: {role}")
        return {"action": "skip", "role": role, "system": system, "messages": messages}

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
        sys_msg = messages[0] if messages and messages[0]["role"] == "system" else None
        messages = [sys_msg] if sys_msg else []
        session_stats["turns"] = 0
        session_stats["total_tokens"] = 0
        print("[OK] 对话上下文已清空")
        return {"action": "skip", "messages": messages}

    if cmd == "/save":
        filepath = arg.strip() or f"qa_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                info = PROVIDER_INFO.get(provider, {})
                f.write(f"# 中文问答助手 — 会话记录\n")
                f.write(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 后端: {info.get('label', provider)} ({info.get('model', '?')})\n")
                f.write(f"# 角色: {role}\n")
                f.write(f"# 对话轮次: {session_stats.get('turns', 0)}\n\n")
                for i, msg in enumerate(messages, 1):
                    if msg["role"] == "system":
                        f.write(f"## System Prompt\n\n{msg['content']}\n\n")
                    else:
                        tag = "[Q]" if msg["role"] == "user" else "[A]"
                        f.write(f"### {tag}\n\n{msg['content']}\n\n")
            print(f"[SAVED] 对话记录已保存到: {os.path.abspath(filepath)}")
        except OSError as e:
            print(f"[ERROR] 保存失败: {e}", file=sys.stderr)
        return {"action": "skip"}

    if cmd == "/stats":
        info = PROVIDER_INFO.get(provider, {})
        turns = session_stats.get("turns", 0)
        total_tokens = session_stats.get("total_tokens", 0)
        msg_count = len([m for m in messages if m["role"] != "system"])
        print(f"[STATS] 后端: {info.get('label', provider)}  |  "
              f"角色: {role}  |  轮次: {turns}  |  "
              f"消息数: {msg_count}  |  累计 Token: {total_tokens}")
        return {"action": "skip"}

    # 未知指令
    print(f"[HELP] 未知指令 '{cmd}'。输入 /help 查看可用指令。")
    return {"action": "skip"}


def build_system_prompt(role: str, custom_system: Optional[str] = None) -> str:
    """根据角色或自定义内容构建系统提示词"""
    if custom_system:
        return custom_system
    return ROLE_PROMPTS.get(role, ROLE_PROMPTS["default"])


def _print_status_line(provider: str, role: str, system: Optional[str],
                       stream: bool, device: str, temperature: float,
                       max_tokens: int) -> None:
    """打印当前会话状态的紧凑摘要行 — 每次 / 指令切换后调用"""
    info = PROVIDER_INFO.get(provider, {})
    device_label = "GPU" if device == "gpu" else "CPU"
    role_label = f"自定义:{system[:20]}..." if system else role
    stream_label = "ON" if stream else "OFF"
    print(f"  [STATUS] 后端: {info.get('label', provider)} | "
          f"设备: {device_label} | 角色: {role_label} | "
          f"流式: {stream_label} | T: {temperature} | Max: {max_tokens}")


def build_messages(question: str, system_prompt: str) -> List[Dict[str, str]]:
    """构建标准的 OpenAI 兼容消息列表 (单次模式用)"""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


# ═══════════════════════════════════════════════════════════
#  Stage 2: 模型调用 — 统一分发层
# ═══════════════════════════════════════════════════════════
#  所有 call_* 函数返回统一格式:
#    {"content": str, "model": str, "finish_reason": str,
#     "usage": {"prompt_tokens": int, "completion_tokens": int, "total_tokens": int},
#     "elapsed": float}
#  流式返回: {"content": str, "elapsed": float}

def build_payload(
    messages: List[Dict[str, str]],
    provider: str = DEFAULT_PROVIDER,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    stream: bool = False,
    device: str = DEFAULT_DEVICE,
) -> dict:
    """构建请求体 — 按 provider 生成对应格式

    LangChain 等价物: model.bind(temperature=..., max_tokens=...)
    """
    if provider == "local":
        return _build_local_payload(messages, temperature, max_tokens, stream, device)
    else:
        return _build_cloud_payload(messages, temperature, max_tokens, stream)


def call_api(payload: dict, provider: str = DEFAULT_PROVIDER) -> dict:
    """非流式调用 — 按 provider 分发"""
    if provider == "local":
        return _call_local_api(payload)
    else:
        return _call_cloud_api(payload)


def call_api_stream(payload: dict, provider: str = DEFAULT_PROVIDER) -> dict:
    """流式调用 — 按 provider 分发"""
    if provider == "local":
        return _call_local_api_stream(payload)
    else:
        return _call_cloud_api_stream(payload)


# ── 云端 DeepSeek 实现 ──────────────────────────────────

def _build_cloud_payload(
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> dict:
    return {
        "model": CLOUD_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }


def _call_cloud_api(payload: dict) -> dict:
    """DeepSeek 非流式调用"""
    url = f"{CLOUD_API_BASE}/chat/completions"
    t0 = time.perf_counter()
    resp = requests.post(
        url, json=payload, timeout=REQUEST_TIMEOUT,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    resp.raise_for_status()
    elapsed = time.perf_counter() - t0
    body = resp.json()
    choice = body.get("choices", [{}])[0]
    return {
        "content": choice.get("message", {}).get("content", ""),
        "model": body.get("model", "?"),
        "finish_reason": choice.get("finish_reason", "?"),
        "usage": body.get("usage", {}),
        "elapsed": elapsed,
    }


def _call_cloud_api_stream(payload: dict) -> dict:
    """DeepSeek SSE 流式调用"""
    url = f"{CLOUD_API_BASE}/chat/completions"
    payload["stream"] = True
    full_text = ""

    t0 = time.perf_counter()
    with requests.post(
        url, json=payload, timeout=REQUEST_TIMEOUT, stream=True,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content", "")
            if token:
                print(token, end="", flush=True)
                full_text += token
    elapsed = time.perf_counter() - t0
    print()
    return {"content": full_text, "elapsed": elapsed}


# ── 本地 Ollama 实现 ────────────────────────────────────

def _build_local_payload(
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    stream: bool,
    device: str = DEFAULT_DEVICE,
) -> dict:
    """Ollama /api/chat 请求体 (OpenAI 兼容 messages 格式)

    device: "cpu" → num_gpu=0 (纯CPU); "gpu" → num_gpu=-1 (自动加载全部GPU层)
    """
    num_gpu = DEVICE_NUM_GPU.get(device, 0)
    options = {
        "temperature": temperature,
        "num_predict": max_tokens,
        "num_gpu": num_gpu,
    }
    if LOCAL_NUM_THREAD is not None:
        options["num_thread"] = LOCAL_NUM_THREAD
    return {
        "model": LOCAL_MODEL,
        "messages": messages,
        "stream": stream,
        "options": options,
    }


def _call_local_api(payload: dict) -> dict:
    """Ollama 非流式调用 /api/chat"""
    url = f"{LOCAL_API_BASE}/api/chat"
    t0 = time.perf_counter()
    resp = requests.post(
        url, json=payload, timeout=REQUEST_TIMEOUT,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    elapsed = time.perf_counter() - t0
    body = resp.json()
    msg = body.get("message", {})
    return {
        "content": msg.get("content", ""),
        "model": body.get("model", LOCAL_MODEL),
        "finish_reason": "stop" if body.get("done") else "length",
        "usage": {
            "prompt_tokens": body.get("prompt_eval_count", 0),
            "completion_tokens": body.get("eval_count", 0),
            "total_tokens": body.get("prompt_eval_count", 0) + body.get("eval_count", 0),
        },
        "elapsed": elapsed,
    }


def _call_local_api_stream(payload: dict) -> dict:
    """Ollama 流式调用 /api/chat — 逐行 JSON"""
    url = f"{LOCAL_API_BASE}/api/chat"
    payload["stream"] = True
    full_text = ""

    t0 = time.perf_counter()
    with requests.post(
        url, json=payload, timeout=REQUEST_TIMEOUT, stream=True,
        headers={"Content-Type": "application/json"},
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            token = chunk.get("message", {}).get("content", "")
            if token:
                print(token, end="", flush=True)
                full_text += token
            if chunk.get("done"):
                break
    elapsed = time.perf_counter() - t0
    print()
    return {"content": full_text, "elapsed": elapsed}


# ═══════════════════════════════════════════════════════════
#  错误处理包装层
# ═══════════════════════════════════════════════════════════

def safe_invoke_nonstream_only(payload: dict, provider: str = DEFAULT_PROVIDER) -> Optional[dict]:
    """非流式调用 + 错误处理"""
    try:
        return call_api(payload, provider)
    except requests.ConnectionError:
        label = PROVIDER_INFO.get(provider, {}).get("label", provider)
        hint = ("请确认 Ollama 已启动: ollama serve"
                if provider == "local" else "请检查网络连接")
        print(f"[ERROR] 无法连接到 {label}。{hint}", file=sys.stderr)
        return None
    except requests.Timeout:
        print(f"[ERROR] 请求超时 (>{REQUEST_TIMEOUT}s)。", file=sys.stderr)
        return None
    except requests.HTTPError as e:
        _print_http_error(e, provider)
        return None
    except KeyboardInterrupt:
        print("\n[WARN] 用户中断。", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return None


def safe_invoke_stream_only(payload: dict, provider: str = DEFAULT_PROVIDER) -> Optional[dict]:
    """流式调用 + 错误处理"""
    try:
        return call_api_stream(payload, provider)
    except requests.ConnectionError:
        label = PROVIDER_INFO.get(provider, {}).get("label", provider)
        hint = ("请确认 Ollama 已启动: ollama serve"
                if provider == "local" else "请检查网络连接")
        print(f"\n[ERROR] 无法连接到 {label}。{hint}", file=sys.stderr)
        return None
    except requests.Timeout:
        print(f"\n[ERROR] 请求超时 (>{REQUEST_TIMEOUT}s)。", file=sys.stderr)
        return None
    except requests.HTTPError as e:
        _print_http_error(e, provider)
        return None
    except KeyboardInterrupt:
        print("\n[WARN] 用户中断。", file=sys.stderr)
        return None
    except Exception as e:
        print(f"\n[ERROR] {type(e).__name__}: {e}", file=sys.stderr)
        return None


def _print_http_error(e: requests.HTTPError, provider: str):
    """统一 HTTP 错误信息"""
    status = e.response.status_code
    detail = ""
    try:
        body = e.response.json()
        if provider == "local":
            detail = body.get("error", "")
        else:
            detail = body.get("error", {}).get("message", "")
    except Exception:
        detail = e.response.text[:200]

    if provider == "local":
        if status == 404:
            print(f"[ERROR] 模型 '{LOCAL_MODEL}' 未找到。运行 'ollama list' 查看。",
                  file=sys.stderr)
        else:
            print(f"[ERROR] Ollama HTTP {status}: {detail}", file=sys.stderr)
    else:
        error_map = {
            401: "鉴权失败 (401): API Key 无效或已过期。",
            402: "余额不足 (402)。",
            429: f"请求频率超限 (429): {detail}",
            500: f"服务端内部错误 (500): {detail}",
        }
        print(f"[ERROR] {error_map.get(status, f'HTTP {status}: {detail}')}",
              file=sys.stderr)


def save_result(content: str, filepath: str, question: str,
                role: str, provider: str = DEFAULT_PROVIDER):
    """将问答结果保存到文件"""
    try:
        info = PROVIDER_INFO.get(provider, {})
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# 中文问答助手 — 问答记录\n")
            f.write(f"# 时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 后端: {info.get('label', provider)} ({info.get('model', '?')})\n")
            f.write(f"# 角色: {role}\n")
            f.write(f"\n## 问题\n\n{question}\n\n")
            f.write(f"## 回答\n\n{content}\n")
        print(f"\n[SAVED] 回答已保存到: {os.path.abspath(filepath)}")
    except OSError as e:
        print(f"\n[ERROR] 保存文件失败: {e}", file=sys.stderr)


# ═══════════════════════════════════════════════════════════
#  交互对话循环 (Interactive Conversation Loop)
# ═══════════════════════════════════════════════════════════

def interactive_session(
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
    [交互模式] 连续对话循环

    维护 messages 列表作为对话历史，每次 API 调用携带完整历史。

    LangChain 等价物:
        chain = prompt | model | output_parser
        chain_with_history = RunnableWithMessageHistory(chain, ...)
    """
    system_prompt = build_system_prompt(role, system)
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": system_prompt}
    ]
    session_stats = {"turns": 0, "total_tokens": 0}
    info = PROVIDER_INFO.get(provider, {})

    print(_INTERACTIVE_BANNER)
    device_label = f"GPU" if device == "gpu" else "CPU"
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
                device=device, messages=messages, session_stats=session_stats,
            )
            role = result.get("role", role)
            system = result.get("system", system)
            stream = result.get("stream", stream)
            provider = result.get("provider", provider)
            device = result.get("device", device)
            messages = result.get("messages", messages)
            session_stats = result.get("session_stats", session_stats)
            if result.get("action") == "break":
                break
            # 切换后端或设备时更新 info
            if "provider" in result or "device" in result:
                info = PROVIDER_INFO.get(provider, {})
            _print_status_line(provider, role, system, stream, device,
                               temperature, max_tokens)
            continue

        # --- 追加用户消息 ---
        messages.append({"role": "user", "content": user_input})

        # --- Stage 2: 模型调用 ---
        payload = build_payload(messages, provider, temperature, max_tokens, stream, device)

        if stream:
            print(f"[{session_stats['turns'] + 1}] [A] ", end="")
            result = safe_invoke_stream_only(payload, provider)
            if result is None:
                messages.pop()
                continue
            content = result["content"]
            elapsed = result["elapsed"]
            messages.append({"role": "assistant", "content": content})
            if not no_echo:
                print(f"--- 耗时: {elapsed:.1f}s | 字符数: {len(content)} ---")
        else:
            result = safe_invoke_nonstream_only(payload, provider)
            if result is None:
                messages.pop()
                continue
            content = result["content"]
            elapsed = result["elapsed"]
            messages.append({"role": "assistant", "content": content})

            print(f"\n[{session_stats['turns'] + 1}] [A]")
            print("-" * 40)
            print(content)
            if not no_echo:
                usage = result.get("usage", {})
                pt = usage.get("prompt_tokens", "?")
                ct = usage.get("completion_tokens", "?")
                tt = usage.get("total_tokens", "?")
                finish = result.get("finish_reason", "?")
                print(f"--- {result.get('model', '?')} | {elapsed:.1f}s | "
                      f"Token: prompt={pt} completion={ct} total={tt} | "
                      f"结束: {finish} ---")

        session_stats["turns"] += 1
        session_stats["total_tokens"] += result.get("usage", {}).get("total_tokens", 0)

    print(f"\n[SUMMARY] 后端: {info.get('label', provider)}  |  "
          f"对话 {session_stats['turns']} 轮  |  "
          f"累计 Token: {session_stats['total_tokens']}")
    return 0


# ═══════════════════════════════════════════════════════════
#  单次问答入口 (Single-shot)
# ═══════════════════════════════════════════════════════════

def single_shot(
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
    [单次模式] 一问一答

    LangChain 等价物: chain.invoke({"question": ...})
    """
    system_prompt = build_system_prompt(role, system)
    messages = build_messages(question, system_prompt)
    payload = build_payload(messages, provider, temperature, max_tokens, stream, device)

    info = PROVIDER_INFO.get(provider, {})

    if stream:
        print("[A] ", end="")
        result = safe_invoke_stream_only(payload, provider)
    else:
        result = safe_invoke_nonstream_only(payload, provider)

    if result is None:
        return 1

    content = result["content"]

    if not stream:
        print("\n" + "=" * 60)
        print("  [回答]")
        print("=" * 60)
        print(content)

    if not no_echo:
        print()
        print("-" * 60)
        elapsed = result.get("elapsed", 0)
        usage = result.get("usage", {})
        if usage:
            pt = usage.get("prompt_tokens", "?")
            ct = usage.get("completion_tokens", "?")
            tt = usage.get("total_tokens", "?")
            print(f"  后端: {info.get('label', provider)}  |  "
                  f"模型: {result.get('model', '?')}  |  "
                  f"耗时: {elapsed:.1f}s  |  "
                  f"Token: prompt={pt} completion={ct} total={tt}")
        else:
            print(f"  后端: {info.get('label', provider)}  |  "
                  f"模式: 流式  |  耗时: {elapsed:.1f}s  |  字符数: {len(content)}")
        finish = result.get("finish_reason", "")
        if finish:
            finish_labels = {"stop": "正常结束", "length": "达到长度上限",
                             "content_filter": "被内容过滤"}
            print(f"  结束原因: {finish} ({finish_labels.get(finish, finish)})")
        print("-" * 60)

    if save:
        save_result(content, save, question, role, provider)

    return 0


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

        question 非空  →  single_shot()
        question 为空  →  interactive_session()
    """
    question = validate_input(question)

    if question:
        return single_shot(
            question=question, provider=provider, role=role, system=system,
            temperature=temperature, max_tokens=max_tokens,
            stream=stream, save=save, no_echo=no_echo, device=device,
        )
    else:
        return interactive_session(
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
