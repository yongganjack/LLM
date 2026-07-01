"""
本地模型 vs 云端 API —— 提示词对比实验

使用相同的测试用例，对比 Llama 3.2 (本地 Ollama) 和 DeepSeek (云端) 在不同
提示词工程技术下的表现差异：

  1. Zero-shot  vs  Few-shot          (情感分类)
  2. Standard   vs  Chain-of-Thought  (数学推理)
  3. No-role    vs  Role-based        (技术解释)
  4. Plain      vs  Structured Output (信息提取)
  5. Vague      vs  Detailed System   (创意写作)

输出: 终端 side-by-side 对比 + comparison_results.json
"""

import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import requests

# ── 复用已有模块的调用函数 ──────────────────────────
# 注意: 这两个模块各自维护自己的 config (URL, API Key, Model Name)
import test as local_model
import cloud_api_demo as cloud_model


# ═══════════════════════════════════════════════════════
#  测试用例定义
# ═══════════════════════════════════════════════════════

# 每个测试用例包含: 标题, 技术名称, 两个变体 (simple, engineered)
# 每个变体包含: system_prompt, user_prompt

TEST_CASES = [
    # ── Case 1: Zero-shot vs Few-shot ──
    {
        "id": 1,
        "title": "Zero-shot  vs  Few-shot",
        "task": "情感分类: 判断句子情感倾向 (正面/负面)",
        "variants": {
            "simple": {
                "label": "Zero-shot",
                "system": "你是一个情感分析助手。",
                "user": "判断以下句子的情感倾向（只回答'正面'或'负面'）：'等了两个小时才上菜，但是味道确实惊艳！'",
            },
            "engineered": {
                "label": "Few-shot (3示例)",
                "system": "你是一个情感分析助手。",
                "user": (
                    "请判断句子的情感倾向，只回答'正面'或'负面'。\n\n"
                    "示例1: '快递太慢了，包装还破了。' → 负面\n"
                    "示例2: '性价比很高，会回购。' → 正面\n"
                    "示例3: '虽然有点贵，但品质确实好。' → 正面\n\n"
                    "现在判断: '等了两个小时才上菜，但是味道确实惊艳！' →"
                ),
            },
        },
    },

    # ── Case 2: Standard vs Chain-of-Thought ──
    {
        "id": 2,
        "title": "Standard  vs  Chain-of-Thought",
        "task": "数学推理: 多步骤算术题",
        "variants": {
            "simple": {
                "label": "直接回答",
                "system": "你是一个数学助手，请直接给出答案。",
                "user": "小明有5个苹果，给了小红2个，又买了3个，最后吃掉了1个。请问小明现在有几个苹果？",
            },
            "engineered": {
                "label": "Chain-of-Thought",
                "system": "你是一个数学助手。",
                "user": (
                    "小明有5个苹果，给了小红2个，又买了3个，最后吃掉了1个。"
                    "请问小明现在有几个苹果？\n\n"
                    "请逐步推理，每一步写出计算过程和中间结果，最后给出答案。"
                ),
            },
        },
    },

    # ── Case 3: No-role vs Role-based ──
    {
        "id": 3,
        "title": "No-role  vs  Role-based",
        "task": "技术概念解释: 什么是 REST API",
        "variants": {
            "simple": {
                "label": "无角色设定",
                "system": "请用中文回答。",
                "user": "什么是 REST API？",
            },
            "engineered": {
                "label": "角色: 资深后端工程师",
                "system": (
                    "你是一位有10年经验的资深后端工程师。"
                    "现在你要给刚入职的实习生解释技术概念。"
                    "请用通俗易懂的语言，配合生活化的类比，最多150字。"
                ),
                "user": "什么是 REST API？",
            },
        },
    },

    # ── Case 4: Plain vs Structured Output ──
    {
        "id": 4,
        "title": "Plain  vs  Structured Output",
        "task": "信息提取: 从句子中提取结构化信息",
        "variants": {
            "simple": {
                "label": "普通文本输出",
                "system": "你是一个信息提取助手。",
                "user": "从以下句子中提取姓名和年龄：'张三今年28岁，在北京工作，职位是软件工程师。'",
            },
            "engineered": {
                "label": "JSON 格式约束",
                "system": "你是一个信息提取助手。请始终以 JSON 格式输出。",
                "user": (
                    "从以下句子中提取姓名、年龄、城市和职位信息。"
                    "请用严格的 JSON 格式输出，不要包含其他内容：\n\n"
                    "句子: '张三今年28岁，在北京工作，职位是软件工程师。'\n\n"
                    '输出格式: {"name": "...", "age": ..., "city": "...", "title": "..."}'
                ),
            },
        },
    },

    # ── Case 5: Vague vs Detailed System Prompt ──
    {
        "id": 5,
        "title": "Vague  vs  Detailed System Prompt",
        "task": "创意写作: 写一首关于编程的诗",
        "variants": {
            "simple": {
                "label": "无系统提示",
                "system": "",
                "user": "写一首关于编程的诗。",
            },
            "engineered": {
                "label": "详细约束 (五言绝句+押韵)",
                "system": (
                    "你是一位精通中国古典诗词的诗人。"
                    "请用五言绝句的格式创作（每句5个字，共4句）。"
                    "必须押韵（第二句和第四句末尾字押韵）。"
                    "主题围绕编程。只输出诗，不要解释。"
                ),
                "user": "写一首关于编程的诗。",
            },
        },
    },
]


# ═══════════════════════════════════════════════════════
#  Part 1: 构建请求 & 调用
# ═══════════════════════════════════════════════════════

def call_local(prompt: str, system_prompt: str,
               temperature: float = 0.7, max_tokens: int = 512) -> Dict[str, Any]:
    """调用本地 Ollama 模型，返回 {response, elapsed, tokens, error}"""
    payload = local_model.build_payload(
        prompt=prompt,
        system_prompt=system_prompt or None,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    t0 = time.perf_counter()
    try:
        result = local_model.call_ollama(payload)
        elapsed = time.perf_counter() - t0
        return {
            "response": result.get("response", ""),
            "elapsed": elapsed,
            "eval_count": result.get("eval_count", 0),
            "prompt_eval_count": result.get("prompt_eval_count", 0),
            "error": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "response": "",
            "elapsed": elapsed,
            "eval_count": 0,
            "prompt_eval_count": 0,
            "error": str(e),
        }


def call_cloud(user_prompt: str, system_prompt: str,
               temperature: float = 0.7, max_tokens: int = 512) -> Dict[str, Any]:
    """调用云端 DeepSeek API，返回 {response, elapsed, tokens, error}"""
    messages = cloud_model.build_messages(user_prompt, system_prompt)
    payload = cloud_model.build_payload(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )
    t0 = time.perf_counter()
    try:
        result = cloud_model.call_api(payload)
        elapsed = time.perf_counter() - t0
        usage = result.get("usage", {})
        choice = result.get("choices", [{}])[0]
        return {
            "response": choice.get("message", {}).get("content", ""),
            "elapsed": elapsed,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "model": result.get("model", ""),
            "error": None,
        }
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            "response": "",
            "elapsed": elapsed,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "model": "",
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════
#  Part 2: 侧边对比显示
# ═══════════════════════════════════════════════════════

TERM_WIDTH = 100
COL_WIDTH = (TERM_WIDTH - 3) // 2  # 中间 3 字符分隔符 " | "


def fmt_line(text: str, width: int) -> str:
    """中文友好的等宽截断/填充"""
    # 中文字符按 2 格宽度计算
    visible = 0
    result = []
    for ch in text:
        w = 2 if '一' <= ch <= '鿿' or '　' <= ch <= '〿' else 1
        if visible + w > width:
            break
        result.append(ch)
        visible += w
    pad = width - visible
    return ''.join(result) + ' ' * max(pad, 0)


def print_comparison_header(case_id: int, title: str, task: str):
    """打印测试用例标题"""
    print()
    print("=" * TERM_WIDTH)
    print(f"  Case {case_id}: {title}")
    print(f"  Task: {task}")
    print("=" * TERM_WIDTH)


def print_variant_header(variant_type: str, label: str):
    """打印变体小标题: Simple 或 Engineered"""
    print(f"\n  -- {variant_type}: {label} --")


def print_side_by_side(local_result: Dict, cloud_result: Dict):
    """并排打印两个模型的输出"""
    sep = " | "
    local_resp = local_result.get("response", "").replace("\n", " ").replace("\r", "")
    cloud_resp = cloud_result.get("response", "").replace("\n", " ").replace("\r", "")

    # 如果文本太长，分行显示
    max_lines = max(
        (len(local_resp) + COL_WIDTH - 1) // COL_WIDTH if local_resp else 1,
        (len(cloud_resp) + COL_WIDTH - 1) // COL_WIDTH if cloud_resp else 1,
    )

    print()
    print(f"  {'Llama 3.2 (本地)':^{COL_WIDTH}}{sep}{'DeepSeek (云端)':^{COL_WIDTH}}")
    print(f"  {'-' * COL_WIDTH}{sep}{'-' * COL_WIDTH}")

    for i in range(max(max_lines, 1)):
        lo = local_resp[i * COL_WIDTH:(i + 1) * COL_WIDTH] if local_resp else "[错误]"
        co = cloud_resp[i * COL_WIDTH:(i + 1) * COL_WIDTH] if cloud_resp else "[错误]"
        print(f"  {fmt_line(lo, COL_WIDTH)}{sep}{fmt_line(co, COL_WIDTH)}")

    # 统计行
    print(f"  {'-' * COL_WIDTH}{sep}{'-' * COL_WIDTH}")
    le = local_result.get("error")
    ce = cloud_result.get("error")
    local_stats = f"[ERROR] {le}" if le else (
        f"耗时: {local_result.get('elapsed', 0):.1f}s | "
        f"Tokens: {local_result.get('eval_count', '?')}"
    )
    cloud_stats = f"[ERROR] {ce}" if ce else (
        f"耗时: {cloud_result.get('elapsed', 0):.1f}s | "
        f"Tokens: {cloud_result.get('completion_tokens', '?')}"
    )
    print(f"  {fmt_line(local_stats, COL_WIDTH)}{sep}{fmt_line(cloud_stats, COL_WIDTH)}")


# ═══════════════════════════════════════════════════════
#  Part 3: 对比运行器
# ═══════════════════════════════════════════════════════

def run_comparison(temperature: float = 0.7, max_tokens: int = 512):
    """
    遍历所有测试用例，每个用例的每个变体分别调用本地和云端模型，
    打印 side-by-side 对比，并收集结果到 all_results。
    """
    all_results = []
    start_time = datetime.now()

    print(f"\n{' Prompt 对比实验 ':=^{TERM_WIDTH}}")
    print(f"  本地: {local_model.MODEL_NAME} (Ollama)")
    print(f"  云端: {cloud_model.MODEL_NAME} (DeepSeek API)")
    print(f"  开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Temperature: {temperature}  |  Max Tokens: {max_tokens}")

    for case in TEST_CASES:
        case_id = case["id"]
        title = case["title"]
        task = case["task"]
        variants = case["variants"]

        print_comparison_header(case_id, title, task)
        case_record = {
            "id": case_id,
            "title": title,
            "task": task,
            "variants": {},
        }

        for variant_key in ["simple", "engineered"]:
            v = variants[variant_key]
            label = v["label"]
            system_prompt = v["system"]
            user_prompt = v["user"]

            print_variant_header(
                "Simple  " if variant_key == "simple" else "Eng.    ",
                label,
            )

            # 显示使用的提示词 (截断)
            sys_preview = (system_prompt[:50] + "...") if len(system_prompt) > 50 else system_prompt
            usr_preview = (user_prompt[:60] + "...") if len(user_prompt) > 60 else user_prompt
            print(f"    System: {sys_preview if sys_preview else '(无)'}")
            print(f"    User:   {usr_preview}")

            # 调用两个模型
            local_result = call_local(user_prompt, system_prompt, temperature, max_tokens)
            cloud_result = call_cloud(user_prompt, system_prompt, temperature, max_tokens)

            # 显示对比
            print_side_by_side(local_result, cloud_result)

            # 记录结果
            case_record["variants"][variant_key] = {
                "label": label,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "local": {
                    "response": local_result["response"],
                    "elapsed": local_result["elapsed"],
                    "eval_count": local_result.get("eval_count", 0),
                    "error": local_result.get("error"),
                },
                "cloud": {
                    "response": cloud_result["response"],
                    "elapsed": cloud_result["elapsed"],
                    "prompt_tokens": cloud_result.get("prompt_tokens", 0),
                    "completion_tokens": cloud_result.get("completion_tokens", 0),
                    "model": cloud_result.get("model", ""),
                    "error": cloud_result.get("error"),
                },
            }

        all_results.append(case_record)

    # ── 总结 ──
    end_time = datetime.now()
    total_elapsed = (end_time - start_time).total_seconds()
    print()
    print("=" * TERM_WIDTH)
    print(f"  实验完成。共 {len(TEST_CASES)} 个对比用例，"
          f"总耗时 {total_elapsed:.1f}s")
    local_errors = sum(
        1 for r in all_results
        for v in r["variants"].values()
        if v["local"]["error"]
    )
    cloud_errors = sum(
        1 for r in all_results
        for v in r["variants"].values()
        if v["cloud"]["error"]
    )
    if local_errors or cloud_errors:
        print(f"  本地错误: {local_errors}  云端错误: {cloud_errors}")
    print("=" * TERM_WIDTH)

    return all_results, start_time


# ═══════════════════════════════════════════════════════
#  Part 4: JSON 结果导出
# ═══════════════════════════════════════════════════════

def export_results(results: List[Dict], start_time: datetime,
                   filename: str = "comparison_results.json"):
    """将对比结果导出为 JSON 文件"""
    output = {
        "meta": {
            "timestamp": start_time.isoformat(),
            "local_model": local_model.MODEL_NAME,
            "cloud_model": cloud_model.MODEL_NAME,
            "total_cases": len(results),
        },
        "cases": results,
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已导出到: {filename}")


# ═══════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════

def main():
    results, start_time = run_comparison(temperature=0.7, max_tokens=256)
    export_results(results, start_time)


if __name__ == "__main__":
    main()
