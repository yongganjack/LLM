"""
调用 DeepSeek 云端 API 完成最小对话
DeepSeek API 兼容 OpenAI SDK 格式，使用 requests 直接调用

包含：发送请求、打印结果、错误处理 三部分
"""

import requests
import json
import sys
import time
import os
from typing import List, Dict


# ── 配置 ──────────────────────────────────────────────
# DeepSeek API: https://platform.deepseek.com/api_keys
API_BASE_URL = "https://api.deepseek.com/v1"
from config import API_KEY
MODEL_NAME = "deepseek-chat"         # deepseek-chat 或 deepseek-reasoner
REQUEST_TIMEOUT = 60                  # 请求超时（秒）

if not API_KEY or API_KEY == "sk-your-key-here":
    print("[WARN] 未设置 API Key。请在代码中设置 API_KEY 或设置环境变量。",
          file=sys.stderr)


# ═══════════════════════════════════════════════════════
#  Part 1: 发送请求
# ═══════════════════════════════════════════════════════

def build_messages(user_prompt: str, system_prompt: str = "") -> List[Dict]:
    """构建 OpenAI 兼容的 messages 列表"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def build_payload(
    messages: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 512,
    stream: bool = False,
) -> dict:
    """构建 /v1/chat/completions 请求体"""
    return {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }


def call_api(payload: dict) -> dict:
    """
    非流式调用 — 一次性返回完整结果
    Returns: choices[0].message.content  + usage info
    """
    url = f"{API_BASE_URL}/chat/completions"
    resp = requests.post(
        url,
        json=payload,
        timeout=REQUEST_TIMEOUT,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    )
    resp.raise_for_status()
    return resp.json()


def call_api_stream(payload: dict) -> str:
    """
    流式调用 — 逐 token 打印，同时累积完整响应
    使用 SSE (Server-Sent Events) 协议
    """
    url = f"{API_BASE_URL}/chat/completions"
    payload["stream"] = True

    full_text = ""
    with requests.post(
        url,
        json=payload,
        timeout=REQUEST_TIMEOUT,
        stream=True,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}",
        },
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            # SSE 格式: data: <json>
            if not line.startswith("data: "):
                continue
            data_str = line[6:]  # 去掉 "data: " 前缀
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            # 从 delta 中提取 content
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            token = delta.get("content", "")
            if token:
                print(token, end="", flush=True)
                full_text += token

    print()  # 换行
    return full_text


# ═══════════════════════════════════════════════════════
#  Part 2: 打印结果
# ═══════════════════════════════════════════════════════

def print_result(result: dict, elapsed: float):
    """格式化打印非流式结果"""
    choice = result.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "[无内容]")
    usage = result.get("usage", {})

    print("\n" + "=" * 60)
    print("  [模型回复]")
    print("=" * 60)
    print(content)
    print("-" * 60)
    print(f"  模型:       {result.get('model', '?')}")
    print(f"  耗时:       {elapsed:.2f}s")
    print(f"  Token 用量: prompt={usage.get('prompt_tokens','?')}, "
          f"completion={usage.get('completion_tokens','?')}, "
          f"total={usage.get('total_tokens','?')}")
    finish = choice.get("finish_reason", "?")
    print(f"  结束原因:   {finish}")
    print("=" * 60)


def print_stream_summary(full_text: str, elapsed: float):
    """流式调用结束后的摘要"""
    print("\n" + "-" * 60)
    print(f"  总耗时: {elapsed:.2f}s  |  字符数: {len(full_text)}")
    print("-" * 60)


# ═══════════════════════════════════════════════════════
#  Part 3: 错误处理
# ═══════════════════════════════════════════════════════

def safe_call(user_prompt: str, system_prompt: str = "",
              use_stream: bool = False, **kwargs):
    """
    带完整错误处理的统一入口
    捕获：连接失败、超时、HTTP 错误、鉴权失败、JSON 解析失败 等
    """
    messages = build_messages(user_prompt, system_prompt)
    payload = build_payload(messages, **kwargs)

    try:
        t0 = time.perf_counter()

        if use_stream:
            full_text = call_api_stream(payload)
            elapsed = time.perf_counter() - t0
            print_stream_summary(full_text, elapsed)
            return full_text
        else:
            result = call_api(payload)
            elapsed = time.perf_counter() - t0
            print_result(result, elapsed)
            return result

    # ── 连接层错误 ──
    except requests.ConnectionError:
        print("[ERROR] 无法连接到 API 服务。", file=sys.stderr)
        print(f"   请检查网络连接: {API_BASE_URL}", file=sys.stderr)
        return None

    except requests.Timeout:
        print(f"[ERROR] 请求超时（>{REQUEST_TIMEOUT}s）。", file=sys.stderr)
        return None

    # ── HTTP 状态码错误 ──
    except requests.HTTPError as e:
        status = e.response.status_code
        detail = ""
        try:
            detail = e.response.json().get("error", {}).get("message", "")
        except Exception:
            detail = e.response.text[:200]

        if status == 401:
            print(f"[ERROR] 鉴权失败 (401): API Key 无效或已过期。", file=sys.stderr)
            print("   请在 DeepSeek 控制台获取新的 Key。", file=sys.stderr)
        elif status == 402:
            print(f"[ERROR] 余额不足 (402)。", file=sys.stderr)
        elif status == 429:
            print(f"[ERROR] 请求频率超限 (429): {detail}", file=sys.stderr)
        elif status == 500:
            print(f"[ERROR] 服务端内部错误 (500): {detail}", file=sys.stderr)
        else:
            print(f"[ERROR] HTTP {status}: {detail}", file=sys.stderr)
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
    print("\n>> [非流式] DeepSeek Chat API ...\n")
    safe_call(
        user_prompt="用一句话解释什么是 API。",
        system_prompt="请用中文回答，简洁明了。",
        temperature=0.7,
        max_tokens=256,
        use_stream=False,
    )

    # ── 流式示例 ──
    print("\n>> [流式] DeepSeek Chat API ...\n")
    safe_call(
        user_prompt="用 Python 写一个 Hello World。",
        system_prompt="请用中文回答，只输出代码。",
        temperature=0.5,
        max_tokens=128,
        use_stream=True,
    )


if __name__ == "__main__":
    main()
