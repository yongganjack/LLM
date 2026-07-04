"""
LangChain 冒烟测试 — 验证依赖安装 + 模型连通性

用途: 在运行 qa_assistant_lc.py 之前, 快速验证:
  1. LangChain 核心包是否正确安装
  2. ChatPromptTemplate 能否正常构建
  3. ChatOllama 能否连接本地 Ollama 服务
  4. ChatDeepSeek 能否连接云端 API
  5. StrOutputParser 是否正常工作
  6. 单次问答 Chain 能否跑通

使用:
  python langchain_smoke_test.py              # 全部测试
  python langchain_smoke_test.py --local      # 仅本地 Ollama 测试
  python langchain_smoke_test.py --cloud      # 仅云端 DeepSeek 测试
"""

import sys
import os
import argparse
import time

# ── Windows 控制台 UTF-8 编码修复 ──────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

# ── 确保可以 import 项目 config ──
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label} — {detail}")


# ═══════════════════════════════════════════════════════════
#  Test 1: 依赖检查
# ═══════════════════════════════════════════════════════════

def test_imports():
    print("\n" + "=" * 60)
    print("  Test 1: LangChain 依赖检查")
    print("=" * 60)

    try:
        import langchain_core
        check("langchain_core", True, f"v{langchain_core.__version__}")
    except Exception as e:
        check("langchain_core", False, str(e))

    try:
        import langchain_ollama
        check("langchain_ollama", True, f"v{langchain_ollama.__version__}")
    except Exception as e:
        check("langchain_ollama", False, str(e))

    try:
        import langchain_deepseek
        check("langchain_deepseek", True, f"v{langchain_deepseek.__version__}")
    except Exception as e:
        check("langchain_deepseek", False, str(e))

    try:
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        check("Messages (System/Human/AI)", True)
    except Exception as e:
        check("Messages (System/Human/AI)", False, str(e))

    try:
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        check("ChatPromptTemplate + MessagesPlaceholder", True)
    except Exception as e:
        check("ChatPromptTemplate + MessagesPlaceholder", False, str(e))

    try:
        from langchain_core.output_parsers import StrOutputParser
        check("StrOutputParser", True)
    except Exception as e:
        check("StrOutputParser", False, str(e))

    try:
        from config import API_KEY
        has_key = bool(API_KEY and API_KEY != "sk-your-key-here")
        check("config.API_KEY", has_key,
              "API_KEY 已配置" if has_key else "API_KEY 未设置或为占位值")
    except Exception as e:
        check("config.API_KEY", False, str(e))


# ═══════════════════════════════════════════════════════════
#  Test 2: Prompt Template 构建
# ═══════════════════════════════════════════════════════════

def test_prompt_template():
    print("\n" + "=" * 60)
    print("  Test 2: Prompt Template 构建")
    print("=" * 60)

    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    # 单轮模板
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个助手"),
            ("human", "{question}"),
        ])
        check("单轮 ChatPromptTemplate", True)
    except Exception as e:
        check("单轮 ChatPromptTemplate", False, str(e))
        return

    # 验证占位符
    try:
        msgs = prompt.invoke({"question": "测试问题"})
        check("单轮 prompt.invoke()", len(msgs.messages) == 2,
              f"消息数: {len(msgs.messages)}")
    except Exception as e:
        check("单轮 prompt.invoke()", False, str(e))

    # 多轮模板 (带 history)
    try:
        chat_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个助手"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        check("多轮 ChatPromptTemplate (带 MessagesPlaceholder)", True)
    except Exception as e:
        check("多轮 ChatPromptTemplate (带 MessagesPlaceholder)", False, str(e))

    # 角色模板 (5 个角色)
    try:
        from week04.qa_assistant_lc import ROLE_PROMPTS, build_prompt_template
        check("ROLE_PROMPTS 导入", len(ROLE_PROMPTS) == 5,
              f"共 {len(ROLE_PROMPTS)} 个角色")
        for r in ["default", "teacher", "coder"]:
            tpl = build_prompt_template(r)
            msgs = tpl.invoke({"question": "测试"})
            check(f"角色 '{r}' 模板构建", len(msgs.messages) >= 2,
                  f"消息数: {len(msgs.messages)}")
    except Exception as e:
        check("角色模板构建", False, str(e))


# ═══════════════════════════════════════════════════════════
#  Test 3: 本地 Ollama 连通性
# ═══════════════════════════════════════════════════════════

def test_local_ollama():
    print("\n" + "=" * 60)
    print("  Test 3: 本地 Ollama 连通性")
    print("=" * 60)

    from langchain_ollama import ChatOllama

    # 初始化模型
    try:
        model = ChatOllama(
            model="llama3.2:1b",
            temperature=0.7,
            num_predict=32,
            num_gpu=-1,
        )
        check("ChatOllama 初始化", True)
    except Exception as e:
        check("ChatOllama 初始化", False, str(e))
        return

    # 非流式调用
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        t0 = time.perf_counter()
        response = model.invoke([
            SystemMessage(content="请用中文回答，极其简洁。"),
            HumanMessage(content="说'你好'"),
        ])
        elapsed = time.perf_counter() - t0
        content = response.content if hasattr(response, 'content') else str(response)
        check("ChatOllama.invoke() (非流式)",
              len(content) > 0,
              f"耗时: {elapsed:.1f}s | 回复: {content[:50]}...")
    except Exception as e:
        check("ChatOllama.invoke() (非流式)", False, str(e))
        return

    # 流式调用
    try:
        t0 = time.perf_counter()
        chunks = []
        for chunk in model.stream([
            SystemMessage(content="请用中文回答，极其简洁。"),
            HumanMessage(content="说'你好'"),
        ]):
            c = chunk.content if hasattr(chunk, 'content') else str(chunk)
            chunks.append(c)
        elapsed = time.perf_counter() - t0
        full = "".join(chunks)
        check("ChatOllama.stream() (流式)",
              len(full) > 0,
              f"耗时: {elapsed:.1f}s | chunk数: {len(chunks)} | 回复: {full[:50]}...")
    except Exception as e:
        check("ChatOllama.stream() (流式)", False, str(e))

    # CPU 模式
    try:
        model_cpu = ChatOllama(
            model="llama3.2:1b",
            temperature=0.7,
            num_predict=16,
            num_gpu=0,
        )
        check("ChatOllama(num_gpu=0) CPU模式初始化", True)
    except Exception as e:
        check("ChatOllama(num_gpu=0) CPU模式初始化", False, str(e))


# ═══════════════════════════════════════════════════════════
#  Test 4: 云端 DeepSeek 连通性
# ═══════════════════════════════════════════════════════════

def test_cloud_deepseek():
    print("\n" + "=" * 60)
    print("  Test 4: 云端 DeepSeek 连通性")
    print("=" * 60)

    from config import API_KEY
    if not API_KEY or API_KEY == "sk-your-key-here":
        print("  [SKIP] API_KEY 未配置, 跳过云端测试")
        return

    from langchain_deepseek import ChatDeepSeek

    # 初始化模型
    try:
        model = ChatDeepSeek(
            model_name="deepseek-chat",
            temperature=0.7,
            max_tokens=32,
            api_key=API_KEY,
        )
        check("ChatDeepSeek 初始化", True)
    except Exception as e:
        check("ChatDeepSeek 初始化", False, str(e))
        return

    # 非流式调用
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        t0 = time.perf_counter()
        response = model.invoke([
            SystemMessage(content="请用中文回答，极其简洁。"),
            HumanMessage(content="说'你好'"),
        ])
        elapsed = time.perf_counter() - t0
        content = response.content if hasattr(response, 'content') else str(response)
        check("ChatDeepSeek.invoke() (非流式)",
              len(content) > 0,
              f"耗时: {elapsed:.1f}s | 回复: {content[:50]}...")
    except Exception as e:
        check("ChatDeepSeek.invoke() (非流式)", False, str(e))

    # 流式调用
    try:
        t0 = time.perf_counter()
        chunks = []
        for chunk in model.stream([
            SystemMessage(content="请用中文回答，极其简洁。"),
            HumanMessage(content="说'你好'"),
        ]):
            c = chunk.content if hasattr(chunk, 'content') else str(chunk)
            chunks.append(c)
        elapsed = time.perf_counter() - t0
        full = "".join(chunks)
        check("ChatDeepSeek.stream() (流式)",
              len(full) > 0,
              f"耗时: {elapsed:.1f}s | chunk数: {len(chunks)} | 回复: {full[:50]}...")
    except Exception as e:
        check("ChatDeepSeek.stream() (流式)", False, str(e))


# ═══════════════════════════════════════════════════════════
#  Test 5: LCEL Chain 集成测试
# ═══════════════════════════════════════════════════════════

def test_lcel_chain(test_cloud: bool = False):
    print("\n" + "=" * 60)
    print("  Test 5: LCEL Chain 集成测试")
    print("=" * 60)

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_ollama import ChatOllama

    # 构建 chain: prompt | model | parser
    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "请用中文回答，极其简洁，不超过10个字。"),
            ("human", "{question}"),
        ])
        model = ChatOllama(model="llama3.2:1b", temperature=0.7, num_predict=32)
        chain = prompt | model | StrOutputParser()
        check("LCEL Chain 构建 (prompt | model | StrOutputParser)", True)
    except Exception as e:
        check("LCEL Chain 构建", False, str(e))
        return

    # invoke
    try:
        t0 = time.perf_counter()
        result = chain.invoke({"question": "1+1等于几？"})
        elapsed = time.perf_counter() - t0
        check("chain.invoke()", len(result) > 0,
              f"耗时: {elapsed:.1f}s | 回复: {result[:50]}...")
    except Exception as e:
        check("chain.invoke()", False, str(e))

    # stream
    try:
        t0 = time.perf_counter()
        chunks = []
        for chunk in chain.stream({"question": "天空是什么颜色？"}):
            chunks.append(chunk)
        elapsed = time.perf_counter() - t0
        full = "".join(chunks)
        check("chain.stream()", len(full) > 0,
              f"耗时: {elapsed:.1f}s | chunk数: {len(chunks)} | 回复: {full[:50]}...")
    except Exception as e:
        check("chain.stream()", False, str(e))

    # 模型工厂测试
    try:
        from week04.qa_assistant_lc import get_model
        local_model = get_model("local", 0.7, 512, "gpu")
        check("get_model('local', ...) GPU", True)
        local_cpu = get_model("local", 0.7, 512, "cpu")
        check("get_model('local', ...) CPU", True)

        if test_cloud:
            from config import API_KEY
            if API_KEY and API_KEY != "sk-your-key-here":
                cloud_model = get_model("cloud", 0.7, 512)
                check("get_model('cloud', ...)", True)
    except Exception as e:
        check("get_model() 工厂", False, str(e))


# ═══════════════════════════════════════════════════════════
#  Test 6: 多轮对话历史传递
# ═══════════════════════════════════════════════════════════

def test_multi_turn_history():
    print("\n" + "=" * 60)
    print("  Test 6: 多轮对话历史传递")
    print("=" * 60)

    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.output_parsers import StrOutputParser
    from langchain_ollama import ChatOllama

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个助手，请记住上下文。用中文回答，极其简洁。"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        model = ChatOllama(model="llama3.2:1b", temperature=0.7, num_predict=64)
        chain = prompt | model | StrOutputParser()
        check("多轮 Chat Chain 构建", True)
    except Exception as e:
        check("多轮 Chat Chain 构建", False, str(e))
        return

    # 模拟两轮对话
    try:
        history = [SystemMessage(content="你是一个助手，请记住上下文。用中文回答，极其简洁。")]

        # 第一轮
        q1 = "我叫小明"
        result1 = chain.invoke({"history": history, "question": q1})
        history.append(HumanMessage(content=q1))
        history.append(AIMessage(content=result1))
        check("第一轮对话", len(result1) > 0, f"回复: {result1[:50]}...")

        # 第二轮 — 验证模型是否记住名字
        q2 = "我叫什么名字？"
        result2 = chain.invoke({"history": history, "question": q2})
        history.append(HumanMessage(content=q2))
        history.append(AIMessage(content=result2))
        check("第二轮对话 (上下文记忆)", len(result2) > 0,
              f"回复: {result2[:80]}...")
        # 简单检查: 回复中是否包含 "小明"
        has_context = "小明" in result2
        check("上下文保持 ('小明' 出现在回复中)", has_context,
              "(可能因模型输出偏差而失败, 不影响脚本功能)")

    except Exception as e:
        check("多轮对话测试", False, str(e))


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LangChain 冒烟测试 — 验证依赖 + 模型连通性"
    )
    parser.add_argument("--local", action="store_true", default=False,
                        help="仅测试本地 Ollama")
    parser.add_argument("--cloud", action="store_true", default=False,
                        help="仅测试云端 DeepSeek")
    parser.add_argument("--quick", action="store_true", default=False,
                        help="快速模式: 仅检查依赖和 Prompt Template")
    args = parser.parse_args()

    print("=" * 60)
    print("  LangChain Smoke Test — Week04")
    print("=" * 60)
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    test_imports()
    test_prompt_template()

    if args.quick:
        print(f"\n{'=' * 60}")
        print(f"  结果: {PASS} PASS, {FAIL} FAIL")
        print(f"{'=' * 60}")
        return 0 if FAIL == 0 else 1

    if not args.cloud:
        test_local_ollama()
        test_lcel_chain(test_cloud=False)
        test_multi_turn_history()

    if not args.local:
        test_cloud_deepseek()

    print(f"\n{'=' * 60}")
    print(f"  结果: {PASS} PASS, {FAIL} FAIL")
    print(f"{'=' * 60}")

    if FAIL > 0:
        print("\n  提示:")
        print("    - 本地测试失败: 确认 'ollama serve' 已启动且 'llama3.2:1b' 已拉取")
        print("    - 云端测试失败: 确认 config.py 中 API_KEY 有效")
        print("    - 依赖测试失败: pip install langchain langchain-ollama langchain-deepseek")
        return 1

    print("  所有测试通过! 可以运行 qa_assistant_lc.py 了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
