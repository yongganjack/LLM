"""
Week07 中文资料问答 Agent 测试脚本

测试范围：
  - _route_input() 三类路由判断
  - should_call_tool() 兼容接口
  - _split_sections() 资料切分
  - _search_sections() 关键词检索
  - format_final_answer() 四段式输出
  - run_agent() 全部路径（直接回答 / 本地检索 / 范围外 / 模拟失败）
  - JSONL 运行记录写入
  - 边界条件（空输入、特殊字符、重复运行）

运行方式：
  python week07/test_simple_agent.py

特点：
  - 默认不依赖真实模型服务（Ollama / DeepSeek）
  - 不依赖 pytest，使用标准库 assert
  - 打印 [PASS] / [FAIL] 风格结果
  - 不需要网络连接
  - 不改写真实资料文件
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# ── 确保可以导入 week07.simple_agent ──
_week07_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(_week07_dir)
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)

# 直接导入模块（不通过包，因为 week07 可能没有 __init__.py）
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "simple_agent",
    os.path.join(_week07_dir, "simple_agent.py"),
)
_simple_agent_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_simple_agent_module)

# 从模块中获取所有需要的函数和常量
_route_input = _simple_agent_module._route_input
should_call_tool = _simple_agent_module.should_call_tool
_split_sections = _simple_agent_module._split_sections
_clean_query = _simple_agent_module._clean_query
_search_sections = _simple_agent_module._search_sections
_generate_run_id = _simple_agent_module._generate_run_id
_write_run_log = _simple_agent_module._write_run_log
_build_run_record = _simple_agent_module._build_run_record
format_final_answer = _simple_agent_module.format_final_answer
run_agent = _simple_agent_module.run_agent
read_local_knowledge = _simple_agent_module.read_local_knowledge
_resolve_knowledge_path = _simple_agent_module._resolve_knowledge_path
_resolve_runs_path = _simple_agent_module._resolve_runs_path
_SECTION_CONCLUSION = _simple_agent_module._SECTION_CONCLUSION
_SECTION_EVIDENCE = _simple_agent_module._SECTION_EVIDENCE
_SECTION_TOOL_STATUS = _simple_agent_module._SECTION_TOOL_STATUS
_SECTION_RUN_RECORD = _simple_agent_module._SECTION_RUN_RECORD


def _default_model_stub(**kwargs):
    """默认模型替身，确保自动测试不会访问真实模型服务。"""
    results = kwargs.get("search_results", [])
    if results:
        body = results[0].get("body", "")[:180]
        return {"ok": True, "answer": f"【模拟模型回答】\n{body}", "error": ""}
    return {"ok": True, "answer": "【模拟模型回答】当前资料未命中相关片段。", "error": ""}


# Week07 自动测试必须隔离真实 API、Ollama 与网络。
_production_generate_model_answer = _simple_agent_module._generate_model_answer
_simple_agent_module._generate_model_answer = _default_model_stub
_simple_agent_module._try_model_enhance = lambda **kwargs: None


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
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════
#  _route_input 测试
# ═══════════════════════════════════════════════════════════

def test_route_direct_answer_greetings():
    """问候语应路由为 direct_answer"""
    greetings = [
        "你好",
        "嗨",
        "hello",
        "你是谁",
        "你叫什么",
    ]
    for g in greetings:
        assert _route_input(g) == "direct_answer", f"'{g}' 应路由为 direct_answer"


def test_route_direct_answer_help():
    """帮助请求应路由为 direct_answer"""
    help_questions = [
        "你能帮助我做什么？",
        "你能做什么",
        "你可以做什么",
        "你会做什么",
        "有什么功能",
        "功能有哪些",
        "怎么用",
        "如何使用",
        "介绍一下你自己",
    ]
    for q in help_questions:
        assert _route_input(q) == "direct_answer", f"'{q}' 应路由为 direct_answer"


def test_route_local_retrieval_explicit():
    """显式资料请求应路由为 local_retrieval"""
    queries = [
        "根据课程资料，提示词模板由哪些部分组成？",
        "请根据知识库说明上下文窗口",
        "课程资料里关于RAG的内容",
        "根据本地资料说明Agent工具调用",
    ]
    for q in queries:
        assert _route_input(q) == "local_retrieval", f"'{q}' 应路由为 local_retrieval"


def test_route_local_retrieval_concepts():
    """课程概念问题应路由为 local_retrieval"""
    queries = [
        "提示词模板是什么？",
        "上下文窗口有什么作用？",
        "什么是RAG？",
        "什么是rag？",
        "Agent如何调用工具？",
        "使用边界是什么？",
        "token是什么？",
    ]
    for q in queries:
        assert _route_input(q) == "local_retrieval", f"'{q}' 应路由为 local_retrieval"


def test_route_out_of_scope():
    """范围外问题应路由为 out_of_scope"""
    queries = [
        "帮我查询今天上海的天气",
        "Python中如何实现单例模式？",
        "帮我写一个排序算法",
        "今天星期几？",
        "推荐一本小说",
    ]
    for q in queries:
        assert _route_input(q) == "out_of_scope", f"'{q}' 应路由为 out_of_scope"


def test_route_empty_input():
    """空输入应路由为 direct_answer"""
    assert _route_input("") == "direct_answer"
    assert _route_input("   ") == "direct_answer"


def test_route_case_insensitive():
    """路由对英文关键词大小写不敏感"""
    assert _route_input("什么是RAG？") == "local_retrieval"
    assert _route_input("什么是rag？") == "local_retrieval"
    assert _route_input("什么是Rag？") == "local_retrieval"
    assert _route_input("什么是agent？") == "local_retrieval"
    assert _route_input("什么是Agent？") == "local_retrieval"
    assert _route_input("什么是AGENT？") == "local_retrieval"


# ═══════════════════════════════════════════════════════════
#  should_call_tool 测试（兼容接口）
# ═══════════════════════════════════════════════════════════

def test_should_call_tool_true():
    """资料问题应返回 True"""
    assert should_call_tool("根据课程资料，提示词模板由哪些部分组成？") is True
    assert should_call_tool("什么是RAG？") is True
    assert should_call_tool("上下文窗口") is True


def test_should_call_tool_false():
    """非资料问题应返回 False"""
    assert should_call_tool("你好") is False
    assert should_call_tool("你能做什么") is False
    assert should_call_tool("今天天气怎么样") is False
    assert should_call_tool("") is False


# ═══════════════════════════════════════════════════════════
#  _split_sections 测试
# ═══════════════════════════════════════════════════════════

def test_split_sections():
    """按 ## 标题切分"""
    content = """# 主标题

一些前言内容。

## 第一节

第一节正文。

## 第二节

第二节正文。"""

    sections = _split_sections(content, "test.md")
    headings = [s["heading"] for s in sections]
    assert "第一节" in headings, f"应包含'第一节'，实际: {headings}"
    assert "第二节" in headings, f"应包含'第二节'，实际: {headings}"


def test_split_sections_has_body():
    """每个片段应包含正文"""
    content = """## 测试标题

这是测试正文内容。"""

    sections = _split_sections(content, "test.md")
    assert len(sections) >= 1
    assert sections[0]["heading"] == "测试标题"
    assert "测试正文" in sections[0]["body"]


def test_split_sections_empty():
    """空内容返回空列表"""
    sections = _split_sections("", "test.md")
    assert sections == []


# ═══════════════════════════════════════════════════════════
#  _clean_query 测试
# ═══════════════════════════════════════════════════════════

def test_clean_query_removes_punctuation():
    """清洗后去除标点"""
    result = _clean_query("根据课程资料，提示词模板由哪些部分组成？")
    assert "根据" not in result, "停用词'根据'应被移除"
    assert "提示词模板" in result, "核心词'提示词模板'应保留"
    assert "哪些" not in result, "停用词'哪些'应被移除"


def test_clean_query_removes_stop_words():
    """清洗后去除停用词"""
    result = _clean_query("什么是RAG？")
    assert "RAG" in result or "rag" in result


# ═══════════════════════════════════════════════════════════
#  _search_sections 测试
# ═══════════════════════════════════════════════════════════

def test_search_sections_finds_heading():
    """标题命中应优先"""
    sections = [
        {"index": 0, "heading": "提示词模板", "body": "提示词模板的基础知识...", "source": "test.md"},
        {"index": 1, "heading": "上下文窗口", "body": "上下文窗口的概念...", "source": "test.md"},
        {"index": 2, "heading": "RAG 与本地检索", "body": "RAG是检索增强生成...", "source": "test.md"},
    ]
    results = _search_sections("提示词模板由哪些部分组成？", sections)
    assert len(results) >= 1, "应至少命中一个片段"
    assert results[0]["heading"] == "提示词模板", f"第一个命中应为'提示词模板'，实际: {results[0]['heading']}"


def test_search_sections_finds_rag():
    """RAG 查询应命中对应章节"""
    sections = [
        {"index": 0, "heading": "提示词模板", "body": "...", "source": "test.md"},
        {"index": 1, "heading": "上下文窗口", "body": "...", "source": "test.md"},
        {"index": 2, "heading": "RAG 与本地检索", "body": "RAG是检索增强生成...", "source": "test.md"},
    ]
    results = _search_sections("什么是RAG？", sections)
    assert len(results) >= 1
    assert "RAG" in results[0]["heading"] or "rag" in results[0]["heading"].lower()


def test_search_sections_context_window():
    """上下文窗口查询应命中对应章节"""
    sections = [
        {"index": 0, "heading": "提示词模板", "body": "...", "source": "test.md"},
        {"index": 1, "heading": "上下文窗口", "body": "上下文窗口的作用...", "source": "test.md"},
        {"index": 2, "heading": "RAG 与本地检索", "body": "...", "source": "test.md"},
    ]
    results = _search_sections("上下文窗口的作用是什么？", sections)
    assert len(results) >= 1
    assert "上下文窗口" in results[0]["heading"]


def test_search_sections_no_match():
    """无命中返回空列表"""
    sections = [
        {"index": 0, "heading": "提示词模板", "body": "...", "source": "test.md"},
    ]
    results = _search_sections("完全无关的查询xyz123", sections)
    assert results == [], "无命中应返回空列表"


def test_search_sections_returns_matched_terms():
    """搜索结果应包含命中词"""
    sections = [
        {"index": 0, "heading": "提示词模板", "body": "提示词模板用于构建prompt...", "source": "test.md"},
    ]
    results = _search_sections("什么是提示词模板？", sections)
    assert len(results) >= 1
    assert len(results[0].get("matched_terms", [])) > 0, "应有命中词"


# ═══════════════════════════════════════════════════════════
#  read_local_knowledge 测试
# ═══════════════════════════════════════════════════════════

def test_read_knowledge_success():
    """读取 week07/local_knowledge.md 应成功"""
    result = read_local_knowledge()
    assert result["ok"] is True, f"读取失败: {result.get('error')}"
    assert len(result["content"]) > 0, "文件内容不应为空"
    assert "提示词模板" in result["content"], "本地知识文件应包含'提示词模板'"
    assert result["error"] == "", "成功时 error 应为空字符串"


def test_read_knowledge_file_not_found():
    """指定不存在的文件路径时应返回失败结构"""
    missing_path = str(Path(__file__).parent / "nonexistent_file_12345.md")
    result = read_local_knowledge(missing_path)
    assert result["ok"] is False, "不存在的文件应返回 ok=False"
    assert result["content"] == "", "失败时 content 应为空字符串"
    assert len(result["error"]) > 0, "失败时 error 不应为空"


def test_read_knowledge_returns_all_fields():
    """返回值始终包含 ok, content, source, error 四个字段"""
    result_ok = read_local_knowledge()
    for field in ("ok", "content", "source", "error"):
        assert field in result_ok, f"成功返回值缺少字段: {field}"

    result_fail = read_local_knowledge("missing.md")
    for field in ("ok", "content", "source", "error"):
        assert field in result_fail, f"失败返回值缺少字段: {field}"


# ═══════════════════════════════════════════════════════════
#  format_final_answer 测试
# ═══════════════════════════════════════════════════════════

def test_format_contains_four_sections():
    """format_final_answer 输出必须包含四段标题"""
    result = format_final_answer(
        conclusion="测试结论",
        evidence="测试依据",
        tool_status="是",
        run_record="测试运行记录",
    )
    assert _SECTION_CONCLUSION in result, "输出缺少 '结论：'"
    assert _SECTION_EVIDENCE in result, "输出缺少 '依据：'"
    assert _SECTION_TOOL_STATUS in result, "输出缺少 '是否调用工具：'"
    assert _SECTION_RUN_RECORD in result, "输出缺少 '运行记录：'"


def test_format_content_appears():
    """format_final_answer 输出应包含传入的内容"""
    result = format_final_answer(
        conclusion="自定义结论内容",
        evidence="自定义依据说明",
        tool_status="否",
        run_record="run_001",
    )
    assert "自定义结论内容" in result
    assert "自定义依据说明" in result
    assert "否" in result
    assert "run_001" in result


def test_format_all_valid_statuses():
    """format_final_answer 应接受所有合法的 tool_status"""
    for status in ("是", "否", "是，但失败"):
        result = format_final_answer("c", "e", status, "r")
        assert status in result, f"工具状态 '{status}' 应出现在输出中"


def test_format_invalid_status_fallback():
    """非法的 tool_status 应被修正为 '否'"""
    result = format_final_answer("c", "e", "invalid_status", "r")
    status_section = result.split(_SECTION_TOOL_STATUS, 1)[1].split(_SECTION_RUN_RECORD)[0]
    assert "否" in status_section, "非法 tool_status 应被修正为 '否'"


def test_format_empty_run_record():
    """空 run_record 应有默认值"""
    result = format_final_answer("c", "e", "否")
    assert _SECTION_RUN_RECORD in result
    assert "无运行记录" in result


# ═══════════════════════════════════════════════════════════
#  _generate_run_id 测试
# ═══════════════════════════════════════════════════════════

def test_generate_run_id_unique():
    """连续生成的 run_id 应互不相同"""
    ids = [_generate_run_id() for _ in range(10)]
    assert len(set(ids)) == len(ids), "run_id 应唯一"


def test_generate_run_id_format():
    """run_id 应符合预期格式"""
    rid = _generate_run_id()
    assert rid.startswith("run_"), f"run_id 应以 'run_' 开头: {rid}"
    parts = rid.split("_")
    assert len(parts) >= 4, f"run_id 应有足够的下划线分段: {rid}"


# ═══════════════════════════════════════════════════════════
#  _write_run_log 测试
# ═══════════════════════════════════════════════════════════

def test_write_run_log_success():
    """写入运行记录应成功"""
    import tempfile
    # 使用临时目录
    original_resolve = _simple_agent_module._resolve_runs_path
    original_resolve_file = _simple_agent_module._resolve_runs_file_path
    tmpdir = tempfile.mkdtemp()
    try:
        _simple_agent_module._resolve_runs_path = lambda: Path(tmpdir)
        _simple_agent_module._resolve_runs_file_path = lambda: Path(tmpdir) / "test_runs.jsonl"

        record = _build_run_record(
            run_id="test_001",
            user_input="测试问题",
            route="local_retrieval",
            tool_called=True,
            tool_status="是",
            source_info="test.md",
            sections_hit="测试章节",
            matched_terms="测试、问题",
            error_info="",
            answer_summary="测试摘要",
            log_ok=True,
        )
        result = _write_run_log(record)
        assert result["ok"] is True, f"日志写入应成功: {result.get('error')}"

        # 验证文件内容
        log_path = Path(tmpdir) / "test_runs.jsonl"
        assert log_path.exists(), "日志文件应存在"
        content = log_path.read_text(encoding="utf-8")
        assert len(content.strip()) > 0, "日志内容不应为空"

        # 验证 JSON 可解析
        parsed = json.loads(content.strip().split("\n")[0])
        assert parsed["run_id"] == "test_001"
        assert parsed["用户输入"] == "测试问题"
        assert parsed["路由"] == "local_retrieval"
        assert parsed["工具状态"] == "是"
        assert "时间" in parsed
        assert "是否调用工具" in parsed
        assert "来源" in parsed
        assert "章节" in parsed
        assert "命中词" in parsed
        assert "错误信息" in parsed
        assert "回答摘要" in parsed
        assert "日志写入状态" in parsed
    finally:
        _simple_agent_module._resolve_runs_path = original_resolve
        _simple_agent_module._resolve_runs_file_path = original_resolve_file
        # 清理
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_write_run_log_permission_error_handled():
    """日志写入失败不应抛异常"""
    original_resolve = _simple_agent_module._resolve_runs_file_path
    original_resolve_dir = _simple_agent_module._resolve_runs_path
    tmpdir = tempfile.mkdtemp()
    try:
        # 创建一个只读目录
        readonly_dir = Path(tmpdir) / "readonly"
        readonly_dir.mkdir()
        log_file = readonly_dir / "test.jsonl"
        log_file.write_text("", encoding="utf-8")
        # 在 Windows 上设置只读
        os.chmod(str(log_file), 0o444)

        _simple_agent_module._resolve_runs_path = lambda: readonly_dir
        _simple_agent_module._resolve_runs_file_path = lambda: log_file

        record = _build_run_record(
            run_id="test_002",
            user_input="测试",
            route="direct_answer",
            tool_called=False,
            tool_status="否",
            source_info="",
            sections_hit="",
            matched_terms="",
            error_info="",
            answer_summary="测试",
            log_ok=False,
        )
        result = _write_run_log(record)
        # 可能成功也可能失败，但不应抛异常
        assert isinstance(result, dict)
        assert "ok" in result
    finally:
        _simple_agent_module._resolve_runs_path = original_resolve_dir
        _simple_agent_module._resolve_runs_file_path = original_resolve
        import shutil
        # 恢复权限后清理
        try:
            os.chmod(str(Path(tmpdir) / "readonly" / "test.jsonl"), 0o666)
        except Exception:
            pass
        shutil.rmtree(tmpdir, ignore_errors=True)


# ═══════════════════════════════════════════════════════════
#  run_agent 测试（模型调用由替身隔离）
# ═══════════════════════════════════════════════════════════

def test_run_agent_empty_input():
    """空输入应返回输入提示，状态为'否'"""
    for empty in ("", "   "):
        result = run_agent(empty)
        assert _SECTION_CONCLUSION in result, "输出缺少 '结论：'"
        assert _SECTION_RUN_RECORD in result, "输出缺少 '运行记录：'"
        assert "请输入有效问题" in result, "应提示输入有效问题"


def test_run_agent_direct_answer():
    """直接回答不应调用工具，状态为'否'"""
    result = run_agent("你能帮助我做什么？")
    assert _SECTION_CONCLUSION in result
    assert _SECTION_EVIDENCE in result
    assert _SECTION_TOOL_STATUS in result
    assert _SECTION_RUN_RECORD in result

    # 检查工具状态为"否"
    status_section = result.split(_SECTION_TOOL_STATUS, 1)[1]
    assert "否" in status_section.split(_SECTION_RUN_RECORD)[0], (
        f"直接回答的 tool_status 应为 '否'"
    )

    # 不应包含资料引用
    assert "local_knowledge.md#提示词模板" not in result, (
        "直接回答不应包含具体资料章节引用"
    )


def test_run_agent_local_retrieval():
    """本地检索应调用工具，状态为'是'，依据包含正确章节"""
    result = run_agent("根据课程资料，提示词模板由哪些部分组成？")

    # 四段格式
    assert _SECTION_CONCLUSION in result
    assert _SECTION_EVIDENCE in result
    assert _SECTION_TOOL_STATUS in result
    assert _SECTION_RUN_RECORD in result

    # 工具状态为"是"
    status_section = result.split(_SECTION_TOOL_STATUS, 1)[1]
    status_text = status_section.split(_SECTION_RUN_RECORD)[0].strip()
    assert "是" in status_text and "失败" not in status_text, (
        f"检索成功的 tool_status 应为 '是'，实际: '{status_text}'"
    )

    # 依据包含正确章节
    assert "提示词模板" in result, "依据应包含命中章节'提示词模板'"
    assert "local_knowledge.md" in result, "依据应包含来源文件"


def test_run_agent_rag_retrieval():
    """RAG 问题应命中正确章节"""
    result = run_agent("什么是RAG？")
    assert "RAG" in result or "rag" in result.lower()
    assert "local_knowledge.md" in result


def test_run_agent_context_window():
    """上下文窗口问题应命中正确章节"""
    result = run_agent("上下文窗口的作用是什么？")
    assert "上下文窗口" in result
    assert "local_knowledge.md" in result


def test_run_agent_no_match():
    """资料无命中时不虚构来源"""
    result = run_agent("根据课程资料，什么是量子计算？")
    # 应说明资料未覆盖
    assert "未覆盖" in result or "无命中" in result or "未找到" in result, (
        "无命中时应说明资料未覆盖"
    )
    # 不应虚构章节
    assert "量子计算" not in result.split(_SECTION_EVIDENCE, 1)[1].split(_SECTION_TOOL_STATUS)[0], (
        "依据中不应包含虚构的资料章节"
    )


def test_run_agent_out_of_scope():
    """范围外问题应说明边界，状态为'否'"""
    result = run_agent("帮我查询今天上海的天气")

    assert _SECTION_CONCLUSION in result
    assert _SECTION_RUN_RECORD in result

    # 应说明不能联网
    assert "联网" in result or "范围" in result or "超出" in result, (
        "范围外回答应说明不能联网或超出范围"
    )

    # 状态为"否"
    status_section = result.split(_SECTION_TOOL_STATUS, 1)[1]
    status_text = status_section.split(_SECTION_RUN_RECORD)[0].strip()
    assert "否" in status_text and "失败" not in status_text, (
        f"范围外的 tool_status 应为 '否'，实际: '{status_text}'"
    )


def test_run_agent_simulate_tool_failure():
    """--simulate-tool-failure 应返回'未获取到资料'和'是，但失败'"""
    result = run_agent(
        "根据课程资料，什么是 RAG？",
        simulate_tool_failure=True,
    )

    assert _SECTION_CONCLUSION in result
    assert _SECTION_EVIDENCE in result
    assert _SECTION_TOOL_STATUS in result
    assert _SECTION_RUN_RECORD in result

    # 结论含"未获取到资料"
    assert "未获取到资料" in result, "模拟失败时结论应包含'未获取到资料'"

    # 状态为"是，但失败"
    status_section = result.split(_SECTION_TOOL_STATUS, 1)[1]
    status_text = status_section.split(_SECTION_RUN_RECORD)[0].strip()
    assert "是，但失败" in status_text, (
        f"模拟失败时 tool_status 应为 '是，但失败'，实际: '{status_text}'"
    )

    # 依据包含模拟错误说明
    assert "模拟" in result, "依据应包含模拟失败说明"


def test_run_agent_simulate_failure_does_not_modify_file():
    """模拟失败后真实资料仍可正常读取"""
    # 先模拟一次失败
    run_agent("测试问题", simulate_tool_failure=True)
    # 再正常读取资料
    result = read_local_knowledge()
    assert result["ok"] is True, "模拟失败后真实资料应仍可正常读取"
    assert "提示词模板" in result["content"], "模拟失败不应修改真实资料内容"


def test_run_agent_all_outputs_have_four_sections():
    """所有路径的输出都应包含四段标题"""
    test_cases = [
        ("", "空输入"),
        ("你能帮助我做什么？", "直接回答"),
        ("根据课程资料，提示词模板由哪些部分组成？", "本地检索"),
        ("帮我查询今天上海的天气", "范围外"),
    ]
    for user_input, desc in test_cases:
        result = run_agent(user_input)
        for marker in [_SECTION_CONCLUSION, _SECTION_EVIDENCE,
                       _SECTION_TOOL_STATUS, _SECTION_RUN_RECORD]:
            assert re_search(marker, result), (
                f"{desc} ('{user_input[:30]}') 输出缺少 '{marker}'"
            )


def test_run_agent_does_not_crash():
    """run_agent 在任何输入下都不应崩溃"""
    edge_cases = [
        "",
        "   ",
        "你能帮助我做什么？",
        "根据课程资料，提示词模板由哪些部分组成？",
        "什么是RAG？",
        "帮我查询今天上海的天气",
        "hello world",
        "a" * 1000,
        "!@#$%^&*()",
        "你好\n换行\n测试",
        "🐍🎉 emoji 测试",
        "   \t\n   ",
    ]
    for inp in edge_cases:
        try:
            result = run_agent(inp)
            assert isinstance(result, str), f"返回值应为 str: {type(result)}"
            assert len(result) > 0, "返回值不应为空字符串"
        except Exception as e:
            raise AssertionError(
                f"run_agent('{inp[:50]}...') 崩溃: {type(e).__name__}: {e}"
            )


def test_run_agent_repeatable():
    """同一输入重复运行应产生一致的路由和工具状态"""
    question = "根据课程资料，提示词模板由哪些部分组成？"
    results = [run_agent(question) for _ in range(3)]

    # 所有结果都应包含"是"且不含"失败"
    for i, r in enumerate(results):
        status_section = r.split(_SECTION_TOOL_STATUS, 1)[1]
        status_text = status_section.split(_SECTION_RUN_RECORD)[0].strip()
        assert "是" in status_text and "失败" not in status_text, (
            f"第 {i+1} 次运行的 tool_status 应为 '是': '{status_text}'"
        )
        assert "提示词模板" in r, f"第 {i+1} 次运行应命中'提示词模板'"


def test_model_answer_is_used_and_local_evidence_is_appended():
    """课程问答应使用模型结论，并由本地代码补充可验证依据。"""
    original_generate = _simple_agent_module._generate_model_answer
    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "answer": "这是模型生成的提示词模板答案。", "error": ""}

    try:
        _simple_agent_module._generate_model_answer = fake_generate
        result = run_agent("根据课程资料，提示词模板由哪些部分组成？")
    finally:
        _simple_agent_module._generate_model_answer = original_generate

    assert "这是模型生成的提示词模板答案。" in result
    assert "本地来源：week07/local_knowledge.md#提示词模板" in result
    assert len(captured["search_results"]) in (1, 2)
    assert captured["search_results"][0]["heading"] == "提示词模板"


def test_model_failure_returns_four_sections_without_local_excerpt():
    """模型失败应保留检索依据，但不能把本地摘录伪装成模型答案。"""
    original_generate = _simple_agent_module._generate_model_answer
    try:
        _simple_agent_module._generate_model_answer = lambda **kwargs: {
            "ok": False,
            "answer": "",
            "error": "模型请求超时: C:/private/api.log",
        }
        result = run_agent("什么是 RAG？")
    finally:
        _simple_agent_module._generate_model_answer = original_generate

    assert "模型服务不可用，未生成最终答案" in result
    assert "C:/private/api.log" not in result
    assert "本地来源：week07/local_knowledge.md#RAG 与本地检索" in result
    assert "是" in result.split(_SECTION_TOOL_STATUS, 1)[1].split(_SECTION_RUN_RECORD, 1)[0]


def test_model_generator_extracts_answer_from_week05_structured_response():
    """Week05 模型链返回 JSON 时，只能使用其中的 answer 字段。"""
    import week05.qa_assistant_structured as structured

    class FakeResponse:
        content = ('```json\n{"answer": "提取后的模型结论", "summary": "摘要", '
                   '"intent": "qa", "follow_up": "", "confidence": 1.0}\n```')

    class FakeChain:
        def invoke(self, payload):
            return FakeResponse()

    original_build = structured.build_chat_chain
    try:
        structured.build_chat_chain = lambda **kwargs: FakeChain()
        result = _production_generate_model_answer(
            user_input="什么是 RAG？",
            search_results=[{"heading": "RAG 与本地检索", "body": "RAG 是检索增强生成。"}],
            provider="cloud", device="gpu", temperature=0.7, max_tokens=128,
        )
    finally:
        structured.build_chat_chain = original_build

    assert result["ok"] is True
    assert result["answer"] == "提取后的模型结论", result


def test_each_agent_call_writes_exactly_one_truthful_log_record():
    """一次课程问答只能写入一条成功状态正确的 JSONL 记录。"""
    original_resolve = _simple_agent_module._resolve_runs_path
    original_resolve_file = _simple_agent_module._resolve_runs_file_path
    tmpdir = tempfile.mkdtemp()
    try:
        log_file = Path(tmpdir) / "agent_runs.jsonl"
        _simple_agent_module._resolve_runs_path = lambda: Path(tmpdir)
        _simple_agent_module._resolve_runs_file_path = lambda: log_file

        run_agent("什么是 RAG？")
        records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line]
    finally:
        _simple_agent_module._resolve_runs_path = original_resolve
        _simple_agent_module._resolve_runs_file_path = original_resolve_file
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert len(records) == 1, f"一次调用应仅写一条日志，实际 {len(records)} 条"
    assert records[0]["日志写入状态"] == "成功"


def test_read_failure_sanitizes_absolute_paths_in_answer_and_log():
    """真实资料读取失败不得在回答或日志中泄露绝对路径。"""
    original_read = _simple_agent_module.read_local_knowledge
    original_resolve = _simple_agent_module._resolve_runs_path
    original_resolve_file = _simple_agent_module._resolve_runs_file_path
    tmpdir = tempfile.mkdtemp()
    absolute_path = "C:/Users/demo/private/local_knowledge.md"
    try:
        log_file = Path(tmpdir) / "agent_runs.jsonl"
        _simple_agent_module.read_local_knowledge = lambda: {
            "ok": False,
            "content": "",
            "source": absolute_path,
            "error": f"文件不存在: {absolute_path}",
        }
        _simple_agent_module._resolve_runs_path = lambda: Path(tmpdir)
        _simple_agent_module._resolve_runs_file_path = lambda: log_file

        result = run_agent("什么是 RAG？")
        records = [json.loads(line) for line in log_file.read_text(encoding="utf-8").splitlines() if line]
        record = records[-1]
    finally:
        _simple_agent_module.read_local_knowledge = original_read
        _simple_agent_module._resolve_runs_path = original_resolve
        _simple_agent_module._resolve_runs_file_path = original_resolve_file
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    assert absolute_path not in result
    assert absolute_path not in json.dumps(record, ensure_ascii=False)
    assert record["来源"] == "week07/local_knowledge.md"


def test_log_write_failure_keeps_answer_complete_and_writes_once():
    """日志失败不影响主回答，且不会重复尝试写入占位日志。"""
    original_write = _simple_agent_module._write_run_log
    calls = []
    try:
        def fake_write(record):
            calls.append(record)
            return {"ok": False, "path": "week07/runs/agent_runs.jsonl", "error": "写入被测试替身拒绝"}

        _simple_agent_module._write_run_log = fake_write
        result = run_agent("你能帮助我做什么？")
    finally:
        _simple_agent_module._write_run_log = original_write

    assert len(calls) == 1
    assert all(marker in result for marker in [_SECTION_CONCLUSION, _SECTION_EVIDENCE,
                                                _SECTION_TOOL_STATUS, _SECTION_RUN_RECORD])
    assert "日志写入失败：写入被测试替身拒绝" in result


def test_terminal_formatter_wraps_long_content_and_preserves_sections():
    """终端展示应自动折行，同时保留四段标题和已有空行。"""
    text = (
        "结论：\n"
        "RAG是一种在生成回答之前先检索相关资料并将其作为上下文提供给模型的架构模式。\n\n"
        "依据：\n"
        "本地来源：week07/local_knowledge.md#RAG 与本地检索。\n\n"
        "是否调用工具：\n是\n\n"
        "运行记录：\nrun_id=test_001"
    )

    formatted = _simple_agent_module._format_terminal_text(text, width=20)

    assert formatted.splitlines()[0] == "结论："
    assert "\n\n依据：\n" in formatted
    assert "\n是否调用工具：\n是\n" in formatted
    assert "\n运行记录：\nrun_id=test_001" in formatted
    for line in formatted.splitlines():
        if line and line not in {"结论：", "依据：", "是否调用工具：", "运行记录："}:
            assert len(line) <= 20, f"显示行超过设定宽度：{line}"


# ═══════════════════════════════════════════════════════════
#  辅助
# ═══════════════════════════════════════════════════════════

def re_search(pattern, text):
    """简单的正则搜索，兼容 Python 标准库。"""
    import re
    return re.search(pattern, text) is not None


# ═══════════════════════════════════════════════════════════
#  入口
# ═══════════════════════════════════════════════════════════

def main():
    global _passed, _failed
    _passed = 0
    _failed = 0

    print("=" * 60)
    print("  Week07 中文资料问答 Agent — 测试")
    print("=" * 60)
    print()
    print("注意：以下测试默认不依赖真实模型服务、网络或 API Key。")
    print()

    # ── _route_input ──
    print("─ _route_input（输入路由）─")
    _run_test("问候语路由为 direct_answer", test_route_direct_answer_greetings)
    _run_test("帮助请求路由为 direct_answer", test_route_direct_answer_help)
    _run_test("显式资料请求路由为 local_retrieval", test_route_local_retrieval_explicit)
    _run_test("课程概念路由为 local_retrieval", test_route_local_retrieval_concepts)
    _run_test("范围外路由为 out_of_scope", test_route_out_of_scope)
    _run_test("空输入路由为 direct_answer", test_route_empty_input)
    _run_test("英文关键词大小写不敏感", test_route_case_insensitive)

    # ── should_call_tool ──
    print("\n─ should_call_tool（兼容接口）─")
    _run_test("资料问题返回 True", test_should_call_tool_true)
    _run_test("非资料问题返回 False", test_should_call_tool_false)

    # ── _split_sections ──
    print("\n─ _split_sections（资料切分）─")
    _run_test("按 ## 标题切分", test_split_sections)
    _run_test("每个片段包含正文", test_split_sections_has_body)
    _run_test("空内容返回空列表", test_split_sections_empty)

    # ── _clean_query ──
    print("\n─ _clean_query（查询清洗）─")
    _run_test("去除标点和停用词", test_clean_query_removes_punctuation)
    _run_test("保留核心关键词", test_clean_query_removes_stop_words)

    # ── _search_sections ──
    print("\n─ _search_sections（关键词检索）─")
    _run_test("标题命中优先", test_search_sections_finds_heading)
    _run_test("RAG 查询命中对应章节", test_search_sections_finds_rag)
    _run_test("上下文窗口查询命中对应章节", test_search_sections_context_window)
    _run_test("无命中返回空列表", test_search_sections_no_match)
    _run_test("搜索结果包含命中词", test_search_sections_returns_matched_terms)

    # ── read_local_knowledge ──
    print("\n─ read_local_knowledge（文件读取）─")
    _run_test("成功读取本地知识文件", test_read_knowledge_success)
    _run_test("不存在文件返回失败但不抛异常", test_read_knowledge_file_not_found)
    _run_test("返回值始终包含四个字段", test_read_knowledge_returns_all_fields)

    # ── format_final_answer ──
    print("\n─ format_final_answer（四段式格式化）─")
    _run_test("输出包含四段标题", test_format_contains_four_sections)
    _run_test("输出包含传入内容", test_format_content_appears)
    _run_test("接受所有合法 tool_status", test_format_all_valid_statuses)
    _run_test("非法 tool_status 修正为 '否'", test_format_invalid_status_fallback)
    _run_test("空 run_record 有默认值", test_format_empty_run_record)

    # ── run_id ──
    print("\n─ _generate_run_id（运行 ID）─")
    _run_test("连续生成 ID 唯一", test_generate_run_id_unique)
    _run_test("ID 格式正确", test_generate_run_id_format)

    # ── 日志 ──
    print("\n─ 日志写入 ─")
    _run_test("写入 JSONL 成功且字段完整", test_write_run_log_success)
    _run_test("写入失败不抛异常", test_write_run_log_permission_error_handled)

    # ── run_agent ──
    print("\n─ run_agent（模型调用使用测试替身）─")
    _run_test("空输入返回输入提示", test_run_agent_empty_input)
    _run_test("直接回答不调用工具", test_run_agent_direct_answer)
    _run_test("提示词模板检索成功", test_run_agent_local_retrieval)
    _run_test("RAG 检索命中正确章节", test_run_agent_rag_retrieval)
    _run_test("上下文窗口检索命中", test_run_agent_context_window)
    _run_test("无命中时不虚构来源", test_run_agent_no_match)
    _run_test("范围外说明边界", test_run_agent_out_of_scope)
    _run_test("模拟工具失败正确降级", test_run_agent_simulate_tool_failure)
    _run_test("模拟失败后真实文件仍可读", test_run_agent_simulate_failure_does_not_modify_file)
    _run_test("所有路径输出四段格式", test_run_agent_all_outputs_have_four_sections)
    _run_test("各类边界输入不崩溃", test_run_agent_does_not_crash)
    _run_test("重复运行结果一致", test_run_agent_repeatable)
    _run_test("模型结论与本地依据职责分离", test_model_answer_is_used_and_local_evidence_is_appended)
    _run_test("模型失败返回完整降级结果", test_model_failure_returns_four_sections_without_local_excerpt)
    _run_test("模型 JSON 仅提取 answer 字段", test_model_generator_extracts_answer_from_week05_structured_response)
    _run_test("每次调用仅写一条如实日志", test_each_agent_call_writes_exactly_one_truthful_log_record)
    _run_test("读取失败不泄露绝对路径", test_read_failure_sanitizes_absolute_paths_in_answer_and_log)
    _run_test("日志失败不影响回答且不重复写入", test_log_write_failure_keeps_answer_complete_and_writes_once)
    _run_test("终端输出自动折行并保留段落", test_terminal_formatter_wraps_long_content_and_preserves_sections)

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
