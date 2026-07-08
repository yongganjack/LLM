"""
Week06 Simple Agent 基础测试脚本

测试范围：
  - should_call_tool() 关键词判断
  - read_local_knowledge() 成功 / 失败路径
  - build_agent_prompt() 三种情况
  - format_final_answer() 格式检查
  - run_agent() 空输入 / 工具路径 / 非工具路径

运行方式：
  python week06/test_simple_agent.py

特点：
  - 默认不依赖真实模型服务（Ollama / DeepSeek）
  - 不依赖 pytest，使用标准库 assert
  - 打印 [PASS] / [FAIL] 风格结果
  - 不需要网络连接
"""

import sys
import os
from pathlib import Path

# ── 确保可以导入 week06.simple_agent ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import week06.simple_agent as simple_agent_module

from week06.simple_agent import (
    should_call_tool,
    read_local_knowledge,
    build_agent_prompt,
    format_final_answer,
    run_agent,
    stream_agent,
    get_last_token_usage,
    TOOL_KEYWORDS,
    _SECTION_CONCLUSION,
    _SECTION_EVIDENCE,
    _SECTION_TOOL_STATUS,
)

# ═══════════════════════════════════════════════════════════
#  测试框架
# ═══════════════════════════════════════════════════════════

_passed = 0
_failed = 0


def _run_test(name: str, fn):
    """运行单个测试并记录结果。"""
    global _passed, _failed
    try:
        fn()
        _passed += 1
        print(f"  [PASS] {name}")
    except AssertionError as e:
        _failed += 1
        print(f"  [FAIL] {name}")
        print(f"         {e}")
    except Exception as e:
        _failed += 1
        print(f"  [FAIL] {name} — 未预期的异常: {type(e).__name__}: {e}")


# ═══════════════════════════════════════════════════════════
#  should_call_tool 测试
# ═══════════════════════════════════════════════════════════

def test_keyword_hit():
    """每个工具关键词都应触发 should_call_tool → True"""
    for kw in TOOL_KEYWORDS:
        assert should_call_tool(f"请根据{kw}回答问题") is True, (
            f"关键词 '{kw}' 未被命中"
        )

    # 包含多个关键词的复合输入
    assert should_call_tool("请根据本地资料说明 Agent 的流程和工具") is True


def test_keyword_no_hit():
    """普通问题不应触发工具调用"""
    normal_questions = [
        "什么是 Prompt Template？",
        "你好，请介绍一下自己",
        "Python 中如何实现单例模式？",
        "今天的天气怎么样？",
        "帮我写一个排序算法",
        "1 + 1 等于多少？",
        "",
        "   ",
    ]
    for q in normal_questions:
        assert should_call_tool(q) is False, (
            f"普通问题不应触发工具调用: '{q}'"
        )


def test_keyword_case_sensitive():
    """关键词匹配是大小写敏感的（中文无大小写，英文关键词需精确匹配）"""
    # "Agent" 需精确匹配，小写 "agent" 不应命中
    # 注意：测试句子不能包含任何其他中文关键词（如 "流程"）
    assert should_call_tool("这个 agent 是什么") is False, (
        "小写 'agent' 不应命中关键词 'Agent'"
    )
    assert should_call_tool("这个 Agent 是什么") is True, (
        "大写 'Agent' 应命中关键词"
    )

    # "week06" 需精确匹配
    assert should_call_tool("week6 项目结构") is False, (
        "'week6' 不应命中关键词 'week06'"
    )
    assert should_call_tool("week06 项目结构") is True, (
        "'week06' 应命中关键词"
    )


# ═══════════════════════════════════════════════════════════
#  read_local_knowledge 测试
# ═══════════════════════════════════════════════════════════

def test_read_knowledge_success():
    """读取 week06/local_knowledge.md 应成功"""
    result = read_local_knowledge()
    assert result["ok"] is True, f"读取失败: {result.get('error')}"
    assert len(result["content"]) > 0, "文件内容不应为空"
    assert "Agent" in result["content"], "本地知识文件应包含 Agent 相关描述"
    assert result["error"] == "", "成功时 error 应为空字符串"
    assert result["source"].endswith("local_knowledge.md"), (
        f"source 应以 local_knowledge.md 结尾，实际: {result['source']}"
    )


def test_read_knowledge_file_not_found():
    """指定不存在的文件路径时应返回失败结构，不抛出异常"""
    missing_path = str(Path(__file__).parent / "nonexistent_file_12345.md")
    result = read_local_knowledge(missing_path)
    assert result["ok"] is False, "不存在的文件应返回 ok=False"
    assert result["content"] == "", "失败时 content 应为空字符串"
    assert len(result["error"]) > 0, "失败时 error 不应为空"
    assert "不存在" in result["error"] or "not found" in result["error"].lower(), (
        f"错误信息应说明文件不存在: {result['error']}"
    )


def test_read_knowledge_returns_all_fields():
    """返回值始终包含 ok, content, source, error 四个字段"""
    # 成功路径
    result_ok = read_local_knowledge()
    for field in ("ok", "content", "source", "error"):
        assert field in result_ok, f"成功返回值缺少字段: {field}"

    # 失败路径
    result_fail = read_local_knowledge("missing.md")
    for field in ("ok", "content", "source", "error"):
        assert field in result_fail, f"失败返回值缺少字段: {field}"


# ═══════════════════════════════════════════════════════════
#  build_agent_prompt 测试
# ═══════════════════════════════════════════════════════════

def test_build_prompt_no_tool():
    """未调用工具时 prompt 应包含相应说明"""
    prompt = build_agent_prompt("什么是 Prompt Template？", None, need_tool=False)
    assert "什么是 Prompt Template？" in prompt, "prompt 应包含用户问题"
    assert "未调用本地工具" in prompt, "prompt 应说明未调用工具"


def test_build_prompt_tool_success():
    """工具成功时 prompt 应包含本地资料内容"""
    tool_result = {
        "ok": True,
        "content": "这是本地知识资料内容",
        "source": "week06/local_knowledge.md",
        "error": "",
    }
    prompt = build_agent_prompt("请根据本地资料回答问题", tool_result, need_tool=True)
    assert "请根据本地资料回答问题" in prompt, "prompt 应包含用户问题"
    assert "这是本地知识资料内容" in prompt, "prompt 应包含工具返回的资料"
    assert "week06/local_knowledge.md" in prompt, "prompt 应包含资料来源"


def test_build_prompt_tool_failure():
    """工具失败时 prompt 应包含错误信息"""
    tool_result = {
        "ok": False,
        "content": "",
        "source": "week06/missing.md",
        "error": "文件不存在",
    }
    prompt = build_agent_prompt("请根据本地资料回答问题", tool_result, need_tool=True)
    assert "请根据本地资料回答问题" in prompt, "prompt 应包含用户问题"
    assert "文件不存在" in prompt, "prompt 应包含工具错误信息"
    assert "未获取到资料" in prompt, "prompt 应说明未获取到资料"


def test_build_prompt_tool_result_none():
    """tool_result 为 None 且 need_tool=True 时的边界情况"""
    prompt = build_agent_prompt("测试问题", None, need_tool=True)
    assert "工具未执行" in prompt, "tool_result 为 None 时应说明工具未执行"


def test_build_prompt_contains_format_instruction():
    """所有 prompt 都应包含三段格式指令"""
    for need_tool, tool_result in [
        (False, None),
        (True, {"ok": True, "content": "test", "source": "s", "error": ""}),
        (True, {"ok": False, "content": "", "source": "s", "error": "err"}),
    ]:
        prompt = build_agent_prompt("测试", tool_result, need_tool)
        assert "结论：" in prompt, "prompt 缺少格式指令: 结论"
        assert "依据：" in prompt, "prompt 缺少格式指令: 依据"
        assert "是否调用工具：" in prompt, "prompt 缺少格式指令: 是否调用工具"


# ═══════════════════════════════════════════════════════════
#  format_final_answer 测试
# ═══════════════════════════════════════════════════════════

def test_format_contains_three_sections():
    """format_final_answer 输出必须包含三段标题"""
    result = format_final_answer(
        conclusion="测试结论",
        evidence="测试依据",
        tool_status="是",
    )
    assert _SECTION_CONCLUSION in result, "输出缺少 '结论：'"
    assert _SECTION_EVIDENCE in result, "输出缺少 '依据：'"
    assert _SECTION_TOOL_STATUS in result, "输出缺少 '是否调用工具：'"


def test_format_content_appears():
    """format_final_answer 输出应包含传入的内容"""
    result = format_final_answer(
        conclusion="自定义结论内容",
        evidence="自定义依据说明",
        tool_status="否",
    )
    assert "自定义结论内容" in result, "输出应包含 conclusion 内容"
    assert "自定义依据说明" in result, "输出应包含 evidence 内容"
    assert "否" in result, "输出应包含 tool_status"


def test_format_all_valid_statuses():
    """format_final_answer 应接受所有合法的 tool_status"""
    for status in ("是", "否", "是，但失败"):
        result = format_final_answer("c", "e", status)
        assert status in result, f"工具状态 '{status}' 应出现在输出中"


def test_format_invalid_status_fallback():
    """非法的 tool_status 应被修正为 '否'"""
    result = format_final_answer("c", "e", "invalid_status")
    assert "否" in result, "非法 tool_status 应被修正为 '否'"
    # 不应包含原始非法值
    assert "invalid_status" not in result.split(_SECTION_TOOL_STATUS, 1)[1], (
        "输出不应包含原始的非法 tool_status"
    )


def test_format_empty_content():
    """format_final_answer 接受空字符串内容"""
    result = format_final_answer("", "", "否")
    assert _SECTION_CONCLUSION in result
    assert _SECTION_EVIDENCE in result
    assert _SECTION_TOOL_STATUS in result
    # 不应崩溃，应有完整的结构


def test_format_multiline_content():
    """format_final_answer 正确处理多行内容"""
    result = format_final_answer(
        conclusion="第一行\n第二行\n第三行",
        evidence="依据行1\n依据行2",
        tool_status="是",
    )
    assert "第一行" in result
    assert "第二行" in result
    assert "依据行1" in result
    assert "是" in result


# ═══════════════════════════════════════════════════════════
#  run_agent 测试（不依赖真实模型）
# ═══════════════════════════════════════════════════════════

def test_run_agent_empty_input():
    """空输入应返回错误提示，不应调用模型或工具"""
    for empty in ("", "   ", "\t", "\n"):
        result = run_agent(empty)
        assert _SECTION_CONCLUSION in result, f"空输入 '{repr(empty)}' 应有三段格式"
        assert "请输入有效问题" in result, "空输入应提示输入有效问题"
        assert "否" in result.split(_SECTION_TOOL_STATUS, 1)[1].strip(), (
            "空输入 tool_status 应为 '否'"
        )


def test_run_agent_tool_path_produces_three_sections():
    """工具路径：输出必须包含三段格式（模型可能不可用，走 fallback）"""
    result = run_agent("请根据本地资料说明 Agent 的工作流程")
    assert _SECTION_CONCLUSION in result, "工具路径输出缺少 '结论：'"
    assert _SECTION_EVIDENCE in result, "工具路径输出缺少 '依据：'"
    assert _SECTION_TOOL_STATUS in result, "工具路径输出缺少 '是否调用工具：'"


def test_run_agent_tool_path_uses_local_knowledge():
    """工具路径不应只返回模型不可用提示，而应包含本地资料中的 Agent 流程语义"""
    result = run_agent("请根据本地资料说明 Agent 的工作流程")
    assert "请安装 LangChain" not in result, "工具成功时不应只提示安装 LangChain"
    assert "启动模型服务后重试" not in result, "工具成功时不应只提示启动模型服务"
    assert "local_knowledge.md" in result or "本地资料" in result, (
        "工具路径应说明依据来自本地资料"
    )
    expected_terms = ("接收用户输入", "判断", "读取", "组织", "工作流程")
    hit_count = sum(1 for term in expected_terms if term in result)
    assert hit_count >= 3, (
        "工具路径回答应包含 Agent 工作流程语义：接收、判断、读取、组织"
    )


def test_run_agent_tool_path_status():
    """工具路径：tool_status 应为 '是' 或 '是，但失败'（取决于模型是否可用）"""
    result = run_agent("请根据本地资料说明 Agent 的工作流程")
    status_line = result.split(_SECTION_TOOL_STATUS, 1)[1].strip()
    assert status_line in ("是", "是，但失败"), (
        f"工具路径 tool_status 应为 '是' 或 '是，但失败'，实际: '{status_line}'"
    )


def test_run_agent_no_tool_path_produces_three_sections():
    """非工具路径：输出必须包含三段格式"""
    result = run_agent("什么是 Prompt Template？")
    assert _SECTION_CONCLUSION in result, "非工具路径输出缺少 '结论：'"
    assert _SECTION_EVIDENCE in result, "非工具路径输出缺少 '依据：'"
    assert _SECTION_TOOL_STATUS in result, "非工具路径输出缺少 '是否调用工具：'"


def test_run_agent_no_tool_path_status():
    """非工具路径：tool_status 应为 '否'（模型不可用时 fallback 也是 '否'）"""
    result = run_agent("什么是 Prompt Template？")
    status_line = result.split(_SECTION_TOOL_STATUS, 1)[1].strip()
    # 模型可用时返回 '否'，模型不可用时 fallback 也返回 '否'
    assert status_line == "否", (
        f"非工具路径 tool_status 应为 '否'，实际: '{status_line}'"
    )


def test_run_agent_does_not_crash():
    """run_agent 在任何输入下都不应崩溃"""
    edge_cases = [
        "",
        "   ",
        "请根据本地资料说明 Agent 的工作流程",
        "什么是 Prompt Template？",
        "hello world",
        "a" * 1000,  # 长输入
        "!@#$%^&*()",  # 特殊字符
    ]
    for inp in edge_cases:
        try:
            result = run_agent(inp)
            assert isinstance(result, str), f"返回值应为 str，实际: {type(result)}"
        except Exception as e:
            raise AssertionError(
                f"run_agent('{inp[:50]}...') 崩溃: {type(e).__name__}: {e}"
            )


def test_stream_agent_matches_run_agent_output():
    """stream_agent 拼接后的内容应与 run_agent 完全一致"""
    question = "请根据本地资料说明 Agent 的工作流程"
    expected = run_agent(question, provider="invalid-provider")
    chunks = list(stream_agent(question, provider="invalid-provider", chunk_size=7))
    assert len(chunks) > 1, "chunk_size 较小时应产生多个流式片段"
    assert "".join(chunks) == expected, "流式片段拼接后应等于完整回答"
    assert _SECTION_CONCLUSION in expected
    assert _SECTION_EVIDENCE in expected
    assert _SECTION_TOOL_STATUS in expected


def test_token_usage_available_after_run_agent():
    """每轮 run_agent 后应可读取 token 用量结构"""
    run_agent("请根据本地资料说明 Agent 的工作流程", provider="invalid-provider")
    usage = get_last_token_usage()
    assert set(("prompt", "completion", "total", "source")).issubset(usage), (
        "token 用量应包含 prompt/completion/total/source"
    )
    assert usage["prompt"] == 0, "模型未调用成功时 prompt token 应为 0"
    assert usage["completion"] == 0, "模型未调用成功时 completion token 应为 0"
    assert usage["total"] == 0, "模型未调用成功时 total token 应为 0"


def test_rule_fallback_tool_success_uses_local_knowledge():
    """模型/Agent 不可用但工具成功时，fallback 应基于本地资料回答"""
    result = simple_agent_module._rule_based_fallback(
        "请根据本地资料说明 Agent 的工作流程",
        "No module named 'langchain'",
    )
    assert "请安装 LangChain" not in result, "fallback 不应只提示安装依赖"
    assert "启动模型服务后重试" not in result, "fallback 不应只提示启动模型服务"
    assert "接收用户输入" in result, "fallback 应包含本地资料中的流程步骤"
    assert "read_local_knowledge" in result, "fallback 应说明唯一工具"
    assert result.split(_SECTION_TOOL_STATUS, 1)[1].strip() == "是", (
        "工具成功 fallback 的 tool_status 应为 '是'"
    )


def test_run_agent_tool_failure_final_output():
    """run_agent 层面的工具失败应返回未获取到资料 + 是，但失败"""
    original_read = simple_agent_module.read_local_knowledge
    try:
        simple_agent_module.read_local_knowledge = lambda path=None: {
            "ok": False,
            "content": "",
            "source": "week06/missing.md",
            "error": "模拟文件不存在",
        }
        result = simple_agent_module.run_agent(
            "请根据本地资料说明 Agent 的工作流程",
            provider="invalid-provider",
        )
    finally:
        simple_agent_module.read_local_knowledge = original_read

    assert "未获取到资料" in result, "工具失败时结论应说明未获取到资料"
    assert "模拟文件不存在" in result, "工具失败时依据应包含错误信息"
    assert result.split(_SECTION_TOOL_STATUS, 1)[1].strip() == "是，但失败", (
        "工具失败时 tool_status 应为 '是，但失败'"
    )


# ═══════════════════════════════════════════════════════════
#  已知路径测试（指定不存在的文件模拟工具失败）
# ═══════════════════════════════════════════════════════════

def test_tool_failure_with_nonexistent_path():
    """工具失败路径：手动调用 read_local_knowledge 指定不存在文件"""
    result = read_local_knowledge("definitely_missing_file_xyz.md")
    assert result["ok"] is False, "不存在的文件应返回 ok=False"
    assert len(result["error"]) > 0, "应包含错误信息"


# ═══════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════

def main():
    global _passed, _failed
    _passed = 0
    _failed = 0

    print("=" * 60)
    print("  Week06 Simple Agent — 基础测试")
    print("=" * 60)
    print()
    print("注意：以下测试默认不依赖真实模型服务。")
    print()

    # ── should_call_tool ──
    print("─ should_call_tool ─")
    _run_test("工具关键词命中", test_keyword_hit)
    _run_test("普通问题不触发", test_keyword_no_hit)
    _run_test("大小写敏感性", test_keyword_case_sensitive)

    # ── read_local_knowledge ──
    print("\n─ read_local_knowledge ─")
    _run_test("成功读取本地知识文件", test_read_knowledge_success)
    _run_test("不存在的文件返回失败但不抛异常", test_read_knowledge_file_not_found)
    _run_test("返回值始终包含四个字段", test_read_knowledge_returns_all_fields)

    # ── build_agent_prompt ──
    print("\n─ build_agent_prompt ─")
    _run_test("未调用工具时的 prompt", test_build_prompt_no_tool)
    _run_test("工具成功时的 prompt", test_build_prompt_tool_success)
    _run_test("工具失败时的 prompt", test_build_prompt_tool_failure)
    _run_test("tool_result=None 时的边界处理", test_build_prompt_tool_result_none)
    _run_test("所有 prompt 包含格式指令", test_build_prompt_contains_format_instruction)

    # ── format_final_answer ──
    print("\n─ format_final_answer ─")
    _run_test("输出包含三段标题", test_format_contains_three_sections)
    _run_test("输出包含传入内容", test_format_content_appears)
    _run_test("接受所有合法 tool_status", test_format_all_valid_statuses)
    _run_test("非法 tool_status 修正为 '否'", test_format_invalid_status_fallback)
    _run_test("空字符串内容不崩溃", test_format_empty_content)
    _run_test("多行内容正确处理", test_format_multiline_content)

    # ── run_agent ──
    print("\n─ run_agent (模型可能不可用，走 fallback) ─")
    _run_test("空输入返回错误提示", test_run_agent_empty_input)
    _run_test("工具路径输出三段格式", test_run_agent_tool_path_produces_three_sections)
    _run_test("工具路径包含本地资料语义", test_run_agent_tool_path_uses_local_knowledge)
    _run_test("工具路径 tool_status 正确", test_run_agent_tool_path_status)
    _run_test("非工具路径输出三段格式", test_run_agent_no_tool_path_produces_three_sections)
    _run_test("非工具路径 tool_status 为 '否'", test_run_agent_no_tool_path_status)
    _run_test("各类边界输入不崩溃", test_run_agent_does_not_crash)
    _run_test("流式输出拼接后等于完整回答", test_stream_agent_matches_run_agent_output)
    _run_test("每轮调用后可读取 token 用量", test_token_usage_available_after_run_agent)
    _run_test("模型不可用时 fallback 基于本地资料回答", test_rule_fallback_tool_success_uses_local_knowledge)
    _run_test("run_agent 工具失败最终输出正确", test_run_agent_tool_failure_final_output)

    # ── 工具失败路径 ──
    print("\n─ 工具失败路径 ─")
    _run_test("不存在文件返回 ok=False", test_tool_failure_with_nonexistent_path)

    # ── 汇总 ──
    print()
    print("=" * 60)
    total = _passed + _failed
    print(f"  结果: {_passed}/{total} 通过", end="")
    if _failed > 0:
        print(f", {_failed} 失败")
    else:
        print(" — 全部通过 ✓")
    print("=" * 60)

    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
