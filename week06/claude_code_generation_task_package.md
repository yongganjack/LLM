# Week06 Simple Agent: Claude Code Generation Task Package

> 用途: 将 Week06 简单 Agent 的实现需求整理为一份可直接交给 Claude 生成代码的任务包。
> 目标: 让 Claude 获得足够清晰的项目背景、接口边界、实现范围和验收标准，而不是只收到一句“帮我写代码”。

---

## 1. 背景说明

当前项目是一个按周递进的 LLM 学习项目：

```text
LlmResearchProject/
├─ week03/  原始问答助手，偏手写 API 调用
├─ week04/  LangChain 版聊天机器人
├─ week05/  结构化输出聊天机器人
└─ week06/  本次简单 Agent 原型生成位置
```

项目演进关系：

```text
week03 手写问答助手
  -> week04 LangChain 聊天机器人
  -> week05 结构化输出聊天机器人
  -> week06 简单 Agent 原型
```

本次任务是在 `week06/` 中新增一个简单 Agent 原型。Agent 应基于 Week05 的聊天机器人能力扩展，但不能修改 Week05 或其他已有目录。

本次实现的学习目标不是使用现成 Agent 框架，而是自己实现一个最小可运行 Agent 流程：

```text
接收用户输入
  -> 判断是否需要调用工具
  -> 调用一个本地工具读取资料
  -> 按固定格式组织回答
```

Agent 功能必须保持简单，重点体现：

- Agent 自己接收输入。
- Agent 自己判断是否需要工具。
- Agent 只调用一个本地工具。
- Agent 组织最终回答。
- Agent 明确自身功能边界和失败处理方式。

---

## 2. 当前相关代码摘要

### 2.1 已有相关文件

```text
week05/qa_assistant_structured.py
week05/test_structured_output.py
week05/prompt_template_output_parser_reference.md
week06/local_knowledge.md
week06/week06_agent_development_plan.md
```

### 2.2 Week05 聊天机器人能力

`week05/qa_assistant_structured.py` 已具备：

- 单轮问答。
- 交互式多轮对话。
- local Ollama / cloud DeepSeek 双后端。
- 结构化输出约束。
- 输出解析与 fallback。
- 从 `week04.qa_assistant_lc` 复用模型配置和通用工具函数。

Week06 可以从 `week05.qa_assistant_structured` 导入能力，但不能修改 Week05 源码。

### 2.3 Week06 已有资料文件

`week06/local_knowledge.md` 已作为本地知识工具的数据源，内容包括：

- 简单 Agent 定义。
- Agent 四步流程。
- 工具调用条件。
- 输出格式。
- 工具失败处理。
- 功能边界。

### 2.4 Week06 已有计划文件

`week06/week06_agent_development_plan.md` 已说明：

- 项目结构理解。
- 相关核心文件。
- 推荐实现方案。
- 每个文件的职责。
- 输入格式、工具调用条件、输出格式和失败处理方式。
- 测试与验收标准。

---

## 3. 需要 Claude 生成的具体代码

请只在 `week06/` 中新增代码文件：

```text
week06/simple_agent.py
week06/test_simple_agent.py
```

不要修改任何已有文件。

### 3.1 `week06/simple_agent.py`

实现简单 Agent 主程序。

建议实现以下函数。

#### `should_call_tool(user_input: str) -> bool`

作用：

- 根据用户输入判断是否需要调用本地知识工具。
- 命中工具关键词时返回 `True`。
- 普通问题返回 `False`。

工具关键词：

```text
资料
知识库
文档
根据本地
本地资料
week06
Agent
工具
流程
依据
```

#### `read_local_knowledge(path: str | None = None) -> dict`

作用：

- 读取 `week06/local_knowledge.md`。
- 成功时返回包含资料内容的 dict。
- 失败时返回包含错误信息的 dict。
- 不要让文件读取异常直接导致 CLI 崩溃。

建议返回结构：

```python
{
    "ok": True,
    "content": "...",
    "source": "week06/local_knowledge.md",
    "error": ""
}
```

失败时：

```python
{
    "ok": False,
    "content": "",
    "source": "week06/local_knowledge.md",
    "error": "..."
}
```

#### `build_agent_prompt(user_input: str, tool_result: dict, need_tool: bool) -> str`

作用：

- 将用户问题、工具调用状态和工具结果组织成给聊天机器人的 prompt。
- 明确要求最终回答使用固定三段格式：
  - `结论`
  - `依据`
  - `是否调用工具`

如果调用工具成功，prompt 中应包含本地资料内容。

如果未调用工具，prompt 中应说明：

```text
未调用本地工具，请基于当前能力回答，并在依据中说明未调用工具。
```

如果工具失败，prompt 中应说明：

```text
本地工具读取失败，请输出“未获取到资料 + 当前可回答部分”。
```

#### `format_final_answer(conclusion: str, evidence: str, tool_status: str) -> str`

作用：

- 对最终结果做兜底格式化。
- 确保输出始终包含固定三段标题。

固定格式：

```text
结论：
<conclusion>

依据：
<evidence>

是否调用工具：
<tool_status>
```

`tool_status` 只允许以下语义：

- `是`
- `否`
- `是，但失败`

#### `run_agent(user_input: str, provider="local", device="gpu", ...) -> str`

作用：

- 执行完整 Agent 流程。
- 流程必须包括：
  1. 接收用户输入。
  2. 调用 `should_call_tool()` 判断是否需要工具。
  3. 必要时调用 `read_local_knowledge()`。
  4. 组织 prompt。
  5. 尝试复用 Week05 聊天机器人能力生成回答。
  6. 如果模型调用失败，使用本地 fallback 生成固定格式回答。

注意：

- 模型调用失败时不能崩溃。
- 工具失败时不能崩溃。
- 空输入时也应返回可读错误提示。

#### `main()`

作用：

- 提供 CLI 入口。
- 支持单次命令行运行。

示例：

```bash
python week06/simple_agent.py "请根据本地资料说明 Agent 的工作流程"
```

可选：

- 无参数时进入简单交互模式。
- 输入 `q`、`quit`、`exit` 退出。

### 3.2 `week06/test_simple_agent.py`

实现基础测试脚本，默认不依赖真实模型服务。

至少测试：

- `should_call_tool()` 对工具关键词返回 `True`。
- `should_call_tool()` 对普通问题返回 `False`。
- `read_local_knowledge()` 能读取 `week06/local_knowledge.md`。
- 指定不存在的路径时，`read_local_knowledge()` 返回失败结构但不抛出异常。
- `format_final_answer()` 输出包含：
  - `结论：`
  - `依据：`
  - `是否调用工具：`
- 完整工具路径至少能走通一次。

测试脚本运行方式：

```bash
python week06/test_simple_agent.py
```

测试脚本可以使用简单断言，不要求引入 pytest。

---

## 4. 输入输出要求

### 4.1 输入格式

Agent 接收用户自然语言问题或任务说明。

CLI 示例：

```bash
python week06/simple_agent.py "请根据本地资料说明 Agent 的工作流程"
```

### 4.2 工具调用判断条件

如果用户输入包含以下关键词之一，则调用本地知识工具：

```text
资料
知识库
文档
根据本地
本地资料
week06
Agent
工具
流程
依据
```

示例：

```text
请根据本地资料说明 Agent 的工作流程
```

应调用工具。

```text
什么是 Prompt Template？
```

不调用工具。

### 4.3 固定输出格式

最终输出必须固定为三部分：

```text
结论：
<最终回答>

依据：
<回答依据>

是否调用工具：
<是 / 否 / 是，但失败>
```

### 4.4 工具调用成功输出示例

```text
结论：
这个简单 Agent 的工作流程是：接收用户输入，判断是否需要额外资料，需要时读取本地知识文件，然后按固定格式组织回答。

依据：
依据来自 week06/local_knowledge.md，其中说明了 Agent 的定义、判断逻辑、工具边界和输出格式。

是否调用工具：
是
```

### 4.5 不调用工具输出示例

```text
结论：
Prompt Template 是一种提示词模板，用于把固定的系统要求和动态输入组合成完整提示。

依据：
未调用本地工具，基于聊天机器人当前能力回答。

是否调用工具：
否
```

---

## 5. 边界条件

必须处理以下情况：

- 用户输入为空。
- 用户输入只包含空格。
- 用户输入没有命中工具关键词。
- 用户输入命中工具关键词。
- `local_knowledge.md` 不存在。
- 文件路径错误。
- 文件读取失败。
- Week05 模型调用失败。
- Ollama 未启动。
- DeepSeek API Key 不可用。
- 模型没有按预期返回三段式内容。

### 5.1 空输入处理

空输入时不要调用模型或工具，直接返回：

```text
结论：
请输入有效问题或任务说明。

依据：
输入为空，未执行工具调用。

是否调用工具：
否
```

### 5.2 工具失败处理

如果本地知识文件不存在或读取失败，最终输出：

```text
结论：
未获取到资料。以下是当前可回答部分：<基于用户问题给出的有限回答>

依据：
本地工具读取失败：<错误信息>

是否调用工具：
是，但失败
```

### 5.3 模型失败处理

如果模型调用失败：

- 不要崩溃。
- 使用本地 fallback 生成三段式回答。
- 依据中说明模型调用失败。

示例：

```text
结论：
未能调用模型生成完整回答。当前可确认的是：该问题已完成工具判断，并按规则处理。

依据：
模型调用失败：<错误信息>

是否调用工具：
是
```

如果同时工具失败和模型失败，则优先说明工具失败：

```text
是否调用工具：
是，但失败
```

---

## 6. 不能改动的接口

不能修改以下路径或文件：

```text
week03/
week04/
week05/
config.py
requirements.txt
API_REFERENCE.md
cloud_api_demo.py
prompt_comparison.py
test.py
comparison_results.*
```

尤其不能修改：

```text
week05/qa_assistant_structured.py
week04/qa_assistant_lc.py
config.py
requirements.txt
```

可以从 `week05.qa_assistant_structured` 导入能力，但不要修改它。

不要使用：

- LangChain Agent
- AutoGPT
- CrewAI
- 外部 Agent 平台
- 向量数据库
- 网络搜索工具
- 新的第三方依赖

不要改变 Week05 既有 CLI、函数名、输出契约或测试文件。

---

## 7. 代码风格要求

整体风格：

- 简单、清晰、适合作业展示。
- 不做大范围抽象。
- 不引入新依赖。
- 函数职责明确。
- 错误处理显式。
- CLI 输出使用中文。
- 注释简短，只解释关键流程。

路径处理：

- 使用 `pathlib.Path(__file__).parent` 或 `os.path.dirname(__file__)` 定位 `week06/local_knowledge.md`。
- 不使用硬编码绝对路径。

测试风格：

- 默认不依赖真实模型服务。
- 可以使用标准库 `assert`。
- 可以打印 `[PASS]` / `[FAIL]` 风格结果。
- 不要求 pytest。

Agent 实现风格：

- 工具调用判断应由本地规则完成，不交给模型判断。
- 只允许一个工具：读取本地知识文件。
- 工具函数只读文件，不产生副作用。
- 最终输出必须稳定为三段。

Fallback 要求：

- 工具失败要 fallback。
- 模型失败要 fallback。
- 模型输出格式不稳定时要 fallback 到 `format_final_answer()`。

---

## 8. 预期测试方式

### 8.1 基础测试

运行：

```bash
python week06/test_simple_agent.py
```

预期：

- 全部基础测试通过。
- 不需要 Ollama。
- 不需要 DeepSeek API。
- 不需要网络。

应覆盖：

- 工具关键词判断。
- 非工具问题判断。
- 本地知识文件读取成功。
- 文件读取失败。
- 固定格式输出。
- 完整工具路径流程。

### 8.2 工具调用路径

运行：

```bash
python week06/simple_agent.py "请根据本地资料说明 Agent 的工作流程"
```

预期：

- 读取 `week06/local_knowledge.md`。
- 输出包含：
  - `结论：`
  - `依据：`
  - `是否调用工具：`
- `是否调用工具` 为 `是`。

### 8.3 非工具路径

运行：

```bash
python week06/simple_agent.py "什么是 Prompt Template？"
```

预期：

- 不读取本地知识文件。
- 输出固定三段。
- `是否调用工具` 为 `否`。

### 8.4 工具失败路径

可以在测试中传入不存在的文件路径，例如：

```python
read_local_knowledge("missing_file.md")
```

预期：

- 返回 `ok=False`。
- 返回错误信息。
- 程序不崩溃。
- 最终输出 `是否调用工具：是，但失败`。

### 8.5 文件边界检查

运行：

```bash
git diff -- week03 week04 week05 requirements.txt config.py
```

预期：

```text
无输出
```

只允许 `week06/` 中新增：

```text
simple_agent.py
test_simple_agent.py
```

已有的 Week06 文档文件可以保留：

```text
week06/local_knowledge.md
week06/week06_agent_development_plan.md
week06/claude_code_generation_task_package.md
```

---

## 9. 推荐实现顺序

1. 在 `week06/simple_agent.py` 中创建基础 CLI。
2. 实现 `should_call_tool()`。
3. 实现 `read_local_knowledge()`。
4. 实现 `format_final_answer()`。
5. 实现 `build_agent_prompt()`。
6. 实现 `run_agent()`。
7. 加入 Week05 聊天机器人调用。
8. 加入模型失败 fallback。
9. 创建 `week06/test_simple_agent.py`。
10. 先跑无模型依赖测试。
11. 再跑真实 CLI 示例。
12. 检查没有改动 `week03/`、`week04/`、`week05/`、`requirements.txt`、`config.py`。

---

## 10. 最终交付要求

Claude 完成后，项目中应新增：

```text
week06/simple_agent.py
week06/test_simple_agent.py
```

并满足：

- `python week06/test_simple_agent.py` 通过。
- 工具路径可以运行。
- 非工具路径可以运行。
- 工具失败路径有降级输出。
- 输出格式始终为三段。
- 没有修改 Week06 之外的任何文件。

