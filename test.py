"""
调用本地 Ollama 的 Llama 3.2 模型
包含：发送请求、打印结果、错误处理 三部分

注意（GPU 用户）：
  如果遇到 "CUDA error: device kernel image is invalid"，说明 NVIDIA 驱动版本
  低于 Ollama 编译时使用的 CUDA 版本。解决方式：
    1. 更新 NVIDIA 驱动到 555+ 版本（推荐，可正常使用 GPU 加速）
    2. 设置 use_gpu=False（CPU 推理，速度较慢但可用）
  当前默认 use_gpu=False 以兼容低版本驱动。
"""

import requests
import json
import sys
import time
from typing import Optional


# ── 配置 ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
MODEL_NAME = "llama3.2:1b"          # 与 ollama list 中的名称一致
REQUEST_TIMEOUT = 60                 # 请求超时（秒）
USE_GPU = False                      # 设为 True 启用 GPU 加速（需驱动 555+）


# ═══════════════════════════════════════════════════════
#  Part 1: 发送请求
# ═══════════════════════════════════════════════════════

def build_payload(
    prompt: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 512,
    stream: bool = False,
    use_gpu: bool = USE_GPU,
) -> dict:
    """构建 Ollama /api/generate 请求体"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_gpu": 999 if use_gpu else 0,   # 999=自动加载所有层到 GPU; 0=纯 CPU
        },
    }
    if system_prompt:
        payload["system"] = system_prompt
    return payload


def call_ollama(payload: dict) -> dict:
    """
    非流式调用 — 一次性返回完整结果
    Returns: {"response": "...", "eval_count": int, "eval_duration": int, ...}
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    resp = requests.post(
        url,
        json=payload,
        timeout=REQUEST_TIMEOUT,
        headers={"Content-Type": "application/json"},
    )
    resp.raise_for_status()
    return resp.json()


def call_ollama_stream(payload: dict):
    """
    流式调用 — 逐 token 打印，同时累积完整响应
    Returns: 累积后的完整文本
    """
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload["stream"] = True

    full_text = ""
    with requests.post(
        url,
        json=payload,
        timeout=REQUEST_TIMEOUT,
        stream=True,
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

            token = chunk.get("response", "")
            print(token, end="", flush=True)
            full_text += token

            if chunk.get("done"):
                break

    print()  # 换行
    return full_text


# ═══════════════════════════════════════════════════════
#  Part 2: 打印结果
# ═══════════════════════════════════════════════════════

def print_result(result: dict, elapsed: float):
    """格式化打印非流式结果"""
    print("\n" + "=" * 60)
    print("  [模型回复]")
    print("=" * 60)
    print(result.get("response", "[无内容]"))
    print("-" * 60)
    print(f"  耗时:       {elapsed:.2f}s")
    print(f"  Token 数:   {result.get('eval_count', '?')}")
    speed = result.get('eval_count', 0) / max(result.get('eval_duration', 1) / 1e9, 0.001)
    print(f"  生成速度:   {speed:.1f} tokens/s")
    print(f"  上下文:     {result.get('prompt_eval_count', '?')} prompt "
          f"+ {result.get('eval_count', '?')} generated")
    print("=" * 60)


def print_stream_summary(full_text: str, elapsed: float):
    """流式调用结束后的摘要"""
    print("\n" + "-" * 60)
    print(f"  总耗时: {elapsed:.2f}s  |  字符数: {len(full_text)}")
    print("-" * 60)


# ═══════════════════════════════════════════════════════
#  Part 3: 错误处理
# ═══════════════════════════════════════════════════════

def safe_call(prompt: str, use_stream: bool = False, **kwargs):
    """
    带完整错误处理的统一入口
    捕获：连接失败、超时、HTTP 错误、模型不存在、JSON 解析失败 等
    """
    payload = build_payload(prompt, **kwargs)

    try:
        t0 = time.perf_counter()

        if use_stream:
            full_text = call_ollama_stream(payload)
            elapsed = time.perf_counter() - t0
            print_stream_summary(full_text, elapsed)
            return full_text
        else:
            result = call_ollama(payload)
            elapsed = time.perf_counter() - t0
            print_result(result, elapsed)
            return result

    # ── 连接层错误 ──
    except requests.ConnectionError:
        print("[ERROR] 无法连接到 Ollama 服务。", file=sys.stderr)
        print(f"   请确认 Ollama 已在 {OLLAMA_BASE_URL} 启动。", file=sys.stderr)
        print("   终端运行:  ollama serve", file=sys.stderr)
        return None

    except requests.Timeout:
        print(f"[ERROR] 请求超时（>{REQUEST_TIMEOUT}s）。", file=sys.stderr)
        print("   可以增大 REQUEST_TIMEOUT 或简化 prompt。", file=sys.stderr)
        return None

    # ── HTTP 状态码错误 ──
    except requests.HTTPError as e:
        status = e.response.status_code
        detail = ""
        try:
            detail = e.response.json().get("error", "")
        except Exception:
            detail = e.response.text[:200]

        if status == 404:
            print(f"[ERROR] 模型 '{MODEL_NAME}' 未找到。", file=sys.stderr)
            print("   运行 'ollama list' 查看可用模型。", file=sys.stderr)
        elif status == 500:
            print(f"[ERROR] Ollama 服务内部错误。\n   {detail}", file=sys.stderr)
        else:
            print(f"[ERROR] HTTP {status} 错误: {detail}", file=sys.stderr)
        return None

    # ── JSON / 数据层错误 ──
    except json.JSONDecodeError as e:
        print(f"[ERROR] 响应 JSON 解析失败: {e}", file=sys.stderr)
        return None

    except requests.RequestException as e:
        print(f"[ERROR] 未知网络错误: {e}", file=sys.stderr)
        return None

    except KeyboardInterrupt:
        print("\n[WARN] 用户中断。", file=sys.stderr)
        return None

    except Exception as e:
        print(f"[ERROR] 未预期的错误: {type(e).__name__}: {e}", file=sys.stderr)
        return None


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

def main():
    # ── 非流式示例 ──
    print("\n>> [非流式] 调用 Llama 3.2 ...\n")
    safe_call(
        prompt="用一句话介绍深度学习。",
        system_prompt="请用中文回答，简洁明了。",
        temperature=0.7,
        max_tokens=256,
        use_stream=False,
    )

    # ── 流式示例 ──
    print("\n>> [流式] 调用 Llama 3.2 ...\n")
    safe_call(
        prompt="写一首关于编程的五言绝句。",
        system_prompt="请用中文回答。",
        temperature=0.8,
        max_tokens=256,
        use_stream=True,
    )


if __name__ == "__main__":
    main()
