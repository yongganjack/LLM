# Week05 Structured Chatbot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Week04 LangChain 聊天机器人的基础上，补齐系统提示词、历史消息管理和稳定输出格式，为后续 Agent 搭建准备 Prompt Template、上下文结构和输出解析约定。

**Architecture:** 保留 Week04 的单文件主实现和现有 CLI 入口，不做大范围重构。围绕 `ChatPromptTemplate + MessagesPlaceholder + BaseMessage history` 增加结构化提示词、输出格式说明、解析与降级策略，并用文档和冒烟测试证明至少 2 轮连续对话可用。

**Tech Stack:** Python, LangChain Core, langchain-ollama, langchain-deepseek, Ollama, DeepSeek API, Markdown.

---

## 1. Project Context

当前项目已经完成 Week03 到 Week04 的迁移：

- `week03/qa_assistant.py`: 原始手写 HTTP 版本，作为历史对照保留。
- `week04/qa_assistant_lc.py`: LangChain 版本主脚本，已经具备 `ChatPromptTemplate`、`MessagesPlaceholder`、`SystemMessage`、`HumanMessage`、`AIMessage`、本地 Ollama 与云端 DeepSeek 双后端。
- `week04/langchain_smoke_test.py`: 冒烟测试脚本，已经覆盖依赖导入、Prompt Template、模型调用和多轮 history。
- `week04/week04_demo.md`: Week04 学习与重构记录。
- `week05/`: 当前用于承接本次结构化聊天机器人增强任务。

本次任务不是重写聊天机器人，也不是实现完整 Agent。重点是让聊天机器人输出更稳定，并把后续 Agent 需要的输入、上下文和输出约定先整理出来。

## 2. External References

本计划参考以下外部知识库：

- https://zhuanlan.zhihu.com/p/2000239845751137469
- https://github.com/datawhalechina/llm-cookbook
- https://github.com/datawhalechina/llm-universe

和本任务最相关的要点：

- Prompt 应明确角色、任务、上下文和输出格式。
- 多轮对话需要显式维护历史消息，避免丢失上下文。
- 输出格式要尽量稳定，便于程序解析和后续 Agent 使用。
- 解析失败时应提供降级路径，不能让整个 CLI 崩溃。

## 3. Files To Touch

### Modify

- `week04/qa_assistant_lc.py`
  - 增强 system prompt。
  - 增加结构化输出格式说明。
  - 增加输出解析与 fallback 逻辑。
  - 保留现有 CLI 使用方式。

- `week04/langchain_smoke_test.py`
  - 增加 Prompt Template 变量测试。
  - 增加结构化输出解析测试。
  - 增加至少 2 轮连续对话演示或测试。

### Create

- `week05/prompt_template_output_parser_reference.md`
  - 记录 Prompt Template 与输出解析对照。
  - 记录两轮连续对话演示。
  - 记录测试命令与验收标准。

### Do Not Modify

- `week03/qa_assistant.py`
- `config.py`
- `.idea/`
- `.claude/`
- `__pycache__/`
- `comparison_results.json`
- `comparison_results.md`
- `prompt_comparison.py`
- `cloud_api_demo.py`
- `test.py`

## 4. Recommended Design

采用“轻量结构化增强”方案：

1. 保留 Week04 的 `prompt | model` LCEL 结构。
2. 保留现有手动维护 `history: List[BaseMessage]` 的方式。
3. 在 system prompt 中加入稳定输出要求。
4. 在 Prompt Template 中显式加入输出格式说明。
5. 用解析函数把模型回答转成统一结构。
6. 如果模型没有返回合法结构，保留原始文本并包装成 fallback 结构。
7. 用文档和 smoke test 证明结构化输出与多轮对话都可用。

暂不采用：

- `RunnableWithMessageHistory`
- 复杂 Agent 框架
- 外部数据库记忆
- 大规模拆分模块
- 复杂 Tool 调用

这些内容适合后续 Week06 或 Agent 阶段再做。

## 5. Target Output Contract

结构化输出建议统一为以下语义字段：

| Field | Type | Purpose |
|---|---|---|
| `answer` | string | 面向用户展示的主要回答 |
| `summary` | string | 对回答的一句话摘要 |
| `intent` | string | 用户意图分类，例如 `qa`、`coding`、`translation`、`medical_advice`、`unknown` |
| `follow_up` | string | 如果需要追问，给出下一步问题；否则为空字符串 |
| `confidence` | number | 0 到 1 的置信度 |
| `raw_text` | string | 解析失败时保留原始模型输出 |
| `parse_ok` | boolean | 标记结构化解析是否成功 |

Agent 后续可以优先消费 `intent`、`summary`、`follow_up`、`confidence`，CLI 用户则主要看到 `answer`。

## 6. Prompt Template And Parser Mapping

| Scenario | Template Structure | Input Variables | Parser Strategy |
|---|---|---|---|
| Single-shot QA | system + human question + output format instruction | `question` | Parse structured response; fallback to plain answer |
| Multi-turn chat | system + history + human question + output format instruction | `history`, `question` | Parse structured response; append user and assistant messages to history |
| Custom role | custom system overrides role prompt | `system`, `question` | Same parser; custom system cannot remove output contract |
| Agent preparation | system requires intent, summary, follow_up, confidence | `history`, `question` | Parsed dict becomes future Agent state input |

## 7. Task List

### Task 1: Confirm Baseline Behavior

**Files:**

- Read: `week04/qa_assistant_lc.py`
- Read: `week04/langchain_smoke_test.py`
- No modifications in this task.

- [ ] Step 1: Check current git status.

  Run:

  ```bash
  git status --short
  ```

  Expected:

  - Existing untracked or user files are noted.
  - No unrelated files are modified by this task.

- [ ] Step 2: Run quick smoke test.

  Run:

  ```bash
  python week04/langchain_smoke_test.py --quick
  ```

  Expected:

  - LangChain imports pass.
  - Prompt Template construction passes.
  - If dependencies are missing, record the exact missing package instead of changing code immediately.

- [ ] Step 3: Inspect current prompt and history functions.

  Confirm these functions still exist:

  - `build_system_text`
  - `build_prompt_template`
  - `build_chat_prompt_template`
  - `build_single_shot_chain`
  - `build_chat_chain`
  - `interactive_session_lc`

### Task 2: Strengthen System Prompt Contract

**Files:**

- Modify: `week04/qa_assistant_lc.py`
- Test: `week04/langchain_smoke_test.py`

- [ ] Step 1: Add a shared output contract text near the existing prompt configuration.

  Acceptance:

  - The contract tells the model to answer in Chinese by default.
  - The contract tells the model to keep output parseable.
  - The contract does not remove existing role behavior.
  - Existing `ROLE_PROMPTS` remains compatible.

- [ ] Step 2: Update `build_system_text` or adjacent prompt construction logic.

  Acceptance:

  - Role prompts still work.
  - `--system` custom prompt still works.
  - The output contract is still applied even when `--system` is used.

- [ ] Step 3: Add a quick test that confirms the generated prompt contains both role instruction and output contract.

  Run:

  ```bash
  python week04/langchain_smoke_test.py --quick
  ```

  Expected:

  - Prompt Template tests pass.
  - No model network call is required for this quick check.

### Task 3: Add Structured Output Parsing

**Files:**

- Modify: `week04/qa_assistant_lc.py`
- Test: `week04/langchain_smoke_test.py`

- [ ] Step 1: Add a parser helper for structured model text.

  Acceptance:

  - Valid structured text returns `parse_ok=True`.
  - Invalid structured text returns `parse_ok=False`.
  - Fallback result always includes `answer`, `summary`, `intent`, `follow_up`, `confidence`, `raw_text`, and `parse_ok`.

- [ ] Step 2: Use the parser in non-streaming single-shot mode.

  Acceptance:

  - CLI still prints a human-readable answer.
  - Internal result has stable fields.
  - Existing `--save` behavior is not broken.

- [ ] Step 3: Use the parser in non-streaming interactive mode.

  Acceptance:

  - The displayed answer remains readable.
  - The history stores the assistant answer text, not an unreadable Python object.
  - Multi-turn context still works.

- [ ] Step 4: Decide how to handle streaming mode.

  Recommended:

  - Keep streaming behavior mostly unchanged.
  - Accumulate full text.
  - Parse only after the stream completes.
  - If parsing fails, display the streamed text as normal and store fallback answer in history.

### Task 4: Preserve History Message Semantics

**Files:**

- Modify: `week04/qa_assistant_lc.py`
- Test: `week04/langchain_smoke_test.py`

- [ ] Step 1: Confirm `history` starts with one `SystemMessage`.

  Acceptance:

  - `interactive_session_lc` initializes history with the current system text.
  - `/clear` preserves the system message.

- [ ] Step 2: Confirm each successful turn appends exactly two messages.

  Expected order:

  - `HumanMessage(content=user_input)`
  - `AIMessage(content=answer_text)`

- [ ] Step 3: Add or update a smoke test for two-turn history.

  Test dialogue:

  ```text
  User 1: Python 如何读取文件？
  Assistant 1: ...
  User 2: 那如何逐行读取？
  Assistant 2: ...
  ```

  Acceptance:

  - The second prompt receives the first turn in `history`.
  - The second answer can refer to file reading context.

### Task 5: Document Prompt Template And Output Parser Mapping

**Files:**

- Create: `week05/prompt_template_output_parser_reference.md`

- [ ] Step 1: Add a table mapping prompt templates to parser behavior.

  Required rows:

  - Single-shot QA
  - Multi-turn chat
  - Role prompt
  - Custom system prompt
  - Future Agent state

- [ ] Step 2: Add the target output contract.

  Required fields:

  - `answer`
  - `summary`
  - `intent`
  - `follow_up`
  - `confidence`
  - `raw_text`
  - `parse_ok`

- [ ] Step 3: Add at least 2 rounds of continuous dialogue demonstration.

  Required:

  - First user message establishes context.
  - Second user message contains a pronoun or ellipsis such as “那”, “它”, or “刚才”.
  - The demonstration explains why this proves history is being used.

### Task 6: Final Verification

**Files:**

- Verify: `week04/qa_assistant_lc.py`
- Verify: `week04/langchain_smoke_test.py`
- Verify: `week05/prompt_template_output_parser_reference.md`

- [ ] Step 1: Run quick smoke test.

  Run:

  ```bash
  python week04/langchain_smoke_test.py --quick
  ```

  Expected:

  - Dependency and prompt tests pass.

- [ ] Step 2: Run parser-specific tests.

  Run the narrowest available parser test once it exists.

  Expected:

  - Valid structured output parses successfully.
  - Plain text fallback does not crash.

- [ ] Step 3: Run a single-shot CLI check.

  Run:

  ```bash
  python week04/qa_assistant_lc.py "什么是 Prompt Template？" --no-echo
  ```

  Expected:

  - The command prints a useful Chinese answer.
  - The output parser does not crash.

- [ ] Step 4: Run a two-turn interactive demonstration.

  Run:

  ```bash
  python week04/qa_assistant_lc.py
  ```

  Then input:

  ```text
  Python 如何读取文件？
  那如何逐行读取？
  q
  ```

  Expected:

  - The second answer uses the file-reading context from the first turn.
  - The session exits normally.

- [ ] Step 5: Check git diff.

  Run:

  ```bash
  git diff -- week04/qa_assistant_lc.py week04/langchain_smoke_test.py week05/prompt_template_output_parser_reference.md
  ```

  Expected:

  - Only planned files changed.
  - No changes to `week03/qa_assistant.py` or `config.py`.

## 8. Acceptance Criteria

The implementation is complete when all of the following are true:

- System prompt includes role behavior and structured output requirements.
- Single-shot mode produces a stable parsed structure or a safe fallback.
- Interactive mode keeps multi-turn history through `MessagesPlaceholder`.
- At least 2 continuous dialogue turns are demonstrated.
- Prompt Template and output parser mapping is documented in Week05.
- Existing CLI flags continue to work.
- Week03 original script remains untouched.
- `config.py` remains untouched.
- Tests or smoke checks document any environment-dependent failures, such as missing Ollama service or missing DeepSeek API key.

## 9. Claude Handoff Context

Use this when asking Claude to generate the code:

```text
请基于当前项目 Week04 的 LangChain 聊天机器人做最小增强，不要大范围重构。

核心文件：
- week04/qa_assistant_lc.py
- week04/langchain_smoke_test.py

需要新增文档：
- week05/prompt_template_output_parser_reference.md

现状：
- 已有 ChatPromptTemplate、MessagesPlaceholder、SystemMessage/HumanMessage/AIMessage。
- 已支持 local Ollama 与 cloud DeepSeek。
- 已支持单轮问答和交互式多轮对话。
- history 当前手动维护。

目标：
1. 补齐更稳定的系统提示词。
2. 给单轮和多轮对话增加结构化输出格式。
3. 增加输出解析逻辑：优先解析结构化输出，失败时 fallback。
4. 整理 Prompt Template 与输出解析对照文档。
5. 完成至少 2 轮连续对话演示。

限制：
- 不修改 week03/qa_assistant.py。
- 不改 config.py。
- 不引入复杂 Agent。
- 不大范围拆文件。
- 保留原 CLI 使用方式。
```

## 10. Risks And Mitigations

| Risk | Mitigation |
|---|---|
| 模型没有严格输出结构化内容 | 增加 fallback parser，保留原始文本 |
| 流式输出难以实时解析 | 流式期间继续显示文本，结束后再解析 |
| 小模型不稳定遵守格式 | 将验收重点放在解析与 fallback 稳定性 |
| 自定义 system prompt 覆盖输出规则 | 将输出规则作为不可省略的附加 contract |
| 多轮 history 被重复注入当前问题 | 保持当前“模型调用成功后再 append 本轮消息”的策略 |
| 云端 API Key 或 Ollama 环境不可用 | quick test 不依赖网络，模型调用失败时记录环境原因 |

## 11. Suggested Commit Plan

Commit 1:

```bash
git add week05/week05_structured_chatbot_development_plan.md
git commit -m "docs: add week05 structured chatbot development plan"
```

Commit 2, after implementation:

```bash
git add week04/qa_assistant_lc.py week04/langchain_smoke_test.py week05/prompt_template_output_parser_reference.md
git commit -m "feat: add structured chatbot prompt and output parsing"
```
