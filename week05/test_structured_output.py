"""
Week05 structured output smoke test.
"""

import argparse
import os
import sys
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore

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
        print(f"  [FAIL] {label} - {detail}")


def warn(label: str, detail: str = ""):
    suffix = f" - {detail}" if detail else ""
    print(f"  [WARN] {label}{suffix}")


def test_imports(strict_optional: bool = True) -> bool:
    print("\n" + "=" * 60)
    print("  Test 1: dependency check")
    print("=" * 60)

    required_ok = True
    optional_ok = True

    try:
        import langchain_core  # noqa: F401
        check("langchain_core", True)
    except Exception as e:
        required_ok = False
        check("langchain_core", False, str(e))

    try:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage  # noqa: F401
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # noqa: F401
        check("core message/prompt classes", True)
    except Exception as e:
        required_ok = False
        check("core message/prompt classes", False, str(e))

    try:
        from config import API_KEY  # noqa: F401
        check("config.API_KEY", True)
    except Exception as e:
        required_ok = False
        check("config.API_KEY", False, str(e))

    try:
        from week05.qa_assistant_structured import (  # noqa: F401
            OUTPUT_CONTRACT,
            build_chat_prompt_template,
            build_prompt_template,
            build_system_text,
            parse_structured_response,
        )
        check("week05.qa_assistant_structured", True)
    except Exception as e:
        required_ok = False
        check("week05.qa_assistant_structured", False, str(e))

    try:
        import langchain_ollama  # noqa: F401
        check("langchain_ollama", True)
    except Exception as e:
        optional_ok = False
        if strict_optional:
            check("langchain_ollama", False, str(e))
        else:
            warn("langchain_ollama", str(e))

    try:
        import langchain_deepseek  # noqa: F401
        check("langchain_deepseek", True)
    except Exception as e:
        optional_ok = False
        if strict_optional:
            check("langchain_deepseek", False, str(e))
        else:
            warn("langchain_deepseek", str(e))

    return required_ok and (optional_ok or not strict_optional)


def test_output_contract():
    print("\n" + "=" * 60)
    print("  Test 2: OUTPUT_CONTRACT")
    print("=" * 60)

    from week05.qa_assistant_structured import OUTPUT_CONTRACT

    check("not empty", len(OUTPUT_CONTRACT) > 50)
    check("answer", "answer" in OUTPUT_CONTRACT)
    check("summary", "summary" in OUTPUT_CONTRACT)
    check("intent", "intent" in OUTPUT_CONTRACT)
    check("follow_up", "follow_up" in OUTPUT_CONTRACT)
    check("confidence", "confidence" in OUTPUT_CONTRACT)
    check("json fence", "```json" in OUTPUT_CONTRACT)
    check("mentions Chinese output", "中文" in OUTPUT_CONTRACT)


def test_parser_unit():
    print("\n" + "=" * 60)
    print("  Test 3: parser unit tests")
    print("=" * 60)

    from week05.qa_assistant_structured import parse_structured_response

    valid = (
        '{"answer": "Python uses open()", '
        '"summary": "file reading", '
        '"intent": "qa", '
        '"follow_up": "Need an example?", '
        '"confidence": 0.9}'
    )
    r = parse_structured_response(valid)
    check("3a parse_ok", r["parse_ok"] is True)
    check("3a answer", r["answer"] == "Python uses open()")
    check("3a summary", r["summary"] == "file reading")
    check("3a intent", r["intent"] == "qa")
    check("3a follow_up", r["follow_up"] == "Need an example?")
    check("3a confidence", r["confidence"] == 0.9)
    check("3a raw_text kept", r["raw_text"] == valid)

    fenced = "before\n```json\n" + valid + "\n```\nafter"
    r2 = parse_structured_response(fenced)
    check("3b fenced parse", r2["parse_ok"] is True)
    check("3b fenced answer", r2["answer"] == "Python uses open()")

    embedded = "prefix " + valid + " suffix"
    r3 = parse_structured_response(embedded)
    check("3c embedded parse", r3["parse_ok"] is True)
    check("3c embedded answer", r3["answer"] == "Python uses open()")

    plain = "plain text answer only"
    r4 = parse_structured_response(plain)
    check("3d plain fallback", r4["parse_ok"] is False)
    check("3d plain answer", r4["answer"] == plain)
    check("3d plain confidence default", r4["confidence"] == 0.5)

    empty = parse_structured_response("")
    check("3e empty fallback", empty["parse_ok"] is False)
    check("3e empty confidence", empty["confidence"] == 0.0)

    bounded = parse_structured_response(
        '{"answer":"x","summary":"","intent":"hacking","follow_up":"","confidence":999}'
    )
    check("3f invalid intent -> unknown", bounded["intent"] == "unknown")
    check("3f confidence clamped", bounded["confidence"] == 1.0)

    typed = parse_structured_response(
        '{"answer": 123, "summary": null, "intent": "qa", "follow_up": true, "confidence": "0.8"}'
    )
    check("3g answer stringified", typed["answer"] == "123")
    check("3g summary empty", typed["summary"] == "")
    check("3g follow_up stringified", typed["follow_up"] == "True")
    check("3g confidence cast", typed["confidence"] == 0.8)


def test_prompt_contract():
    print("\n" + "=" * 60)
    print("  Test 4: prompt contract")
    print("=" * 60)

    from week05.qa_assistant_structured import (
        OUTPUT_CONTRACT,
        build_chat_prompt_template,
        build_prompt_template,
        build_system_text,
    )

    sys_default = build_system_text("default")
    check("default includes contract", OUTPUT_CONTRACT.strip() in sys_default)

    sys_custom = build_system_text("default", custom_system="custom system")
    check("custom includes contract", OUTPUT_CONTRACT.strip() in sys_custom)
    check("custom keeps text", "custom system" in sys_custom)

    tpl = build_prompt_template("default")
    msgs = tpl.invoke({"question": "test"})
    check("single shot system contract", OUTPUT_CONTRACT.strip() in msgs.messages[0].content)

    chat_tpl = build_chat_prompt_template("default")
    chat_msgs = chat_tpl.invoke({"history": [], "question": "test"})
    check("chat system contract", OUTPUT_CONTRACT.strip() in chat_msgs.messages[0].content)


def test_two_turn_structured():
    print("\n" + "=" * 60)
    print("  Test 5: two-turn history")
    print("=" * 60)

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_ollama import ChatOllama
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from week05.qa_assistant_structured import build_system_text, parse_structured_response

    system_text = build_system_text("default")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_text),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ]
    )
    chain = prompt | ChatOllama(model="llama3.2:1b", temperature=0.7, num_predict=128) | StrOutputParser()

    history = [SystemMessage(content=system_text)]

    q1 = "Python file reading?"
    r1 = chain.invoke({"history": history, "question": q1})
    p1 = parse_structured_response(r1)
    a1 = p1["answer"] if p1["parse_ok"] else r1
    history.append(HumanMessage(content=q1))
    history.append(AIMessage(content=a1))
    check("turn1 got response", bool(r1))

    q2 = "How about line-by-line?"
    r2 = chain.invoke({"history": history, "question": q2})
    p2 = parse_structured_response(r2)
    a2 = p2["answer"] if p2["parse_ok"] else r2
    history.append(HumanMessage(content=q2))
    history.append(AIMessage(content=a2))
    check("turn2 got response", bool(r2))
    check("history length", len(history) == 5)
    check(
        "ai messages clean",
        all(not m.content.strip().startswith("{") for m in history if isinstance(m, AIMessage)),
    )


def main():
    parser = argparse.ArgumentParser(description="Week05 structured output smoke test")
    parser.add_argument("--quick", action="store_true", default=False, help="skip provider-required tests")
    parser.add_argument("--local", action="store_true", default=False, help="include local Ollama test")
    args = parser.parse_args()

    print("=" * 60)
    print("  Week05 Structured Output Smoke Test")
    print("=" * 60)
    print(f"  time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    imports_ok = test_imports(strict_optional=not args.quick)
    if not imports_ok:
        print(f"\n{'=' * 60}")
        print(f"  result: {PASS} PASS, {FAIL} FAIL")
        print(f"{'=' * 60}")
        print("\n  tip:")
        print("    - missing base deps: install requirements")
        print("    - quick mode should not require provider packages")
        return 1

    test_output_contract()
    test_parser_unit()
    test_prompt_contract()

    if not args.quick and args.local:
        test_two_turn_structured()

    print(f"\n{'=' * 60}")
    print(f"  result: {PASS} PASS, {FAIL} FAIL")
    print(f"{'=' * 60}")

    if FAIL > 0:
        return 1
    print("  all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
