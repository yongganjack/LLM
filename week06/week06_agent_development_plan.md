# Week06 Simple Agent Development Plan

> 目标: 基于 Week05 的结构化聊天机器人，在 `week06/` 中新增一个可以本地运行的简单 Agent 原型。
> 范围: 只在 `week06/` 中新增文件，不修改 `week03/`、`week04/`、`week05/` 或项目根目录已有文件。

---

## 1. 当前项目结构理解

当前项目按周递进：

```text
LlmResearchProject/
├─ week03/  原始问答助手，偏手写 API 调用
├─ week04/  LangChain 版聊天机器人
├─ week05/  结构化输出聊天机器人
├─ week06/  本次简单 Agent 原型生成位置
├─ requirements.txt
└─ config.py
```

项目演进关系：

```text
week03 手写问答助手
  -> week04 LangChain 聊天机器人
  -> week05 结构化输出聊天机器人
  -> week06 简单 Agent 原型
```

Week05 已经具备：

- 单轮问答和交互式多轮对话。
- 本地 Ollama 与云端 DeepSeek 双后端。
- 结构化输出约束。
- 输出解析与 fallback。
- 干净的多轮历史消息管理。

Week06 不需要重写聊天机器人，应在 Week05 的基础上增加一个极简 Agent 控制流程。

---

## 2. 和需求相关的核心文件

### Week05 可复用文件

- `week05/qa_assistant_structured.py`
  - 已有结构化聊天机器人主逻辑。
  - 可复用模型调用、prompt 构建、输出解析思路。

- `week05/test_structured_output.py`
  - 可参考其测试风格。
  - 可参考 parser、prompt、history 的验收方式。

- `week05/prompt_template_output_parser_reference.md`
  - 说明了 Week05 的 Prompt Template、输出解析和多轮历史设计。

### Week04 底层来源

- `week04/qa_assistant_lc.py`
  - Week05 从这里导入模型工厂、配置常量和通用工具函数。
  - Week06 不应直接修改它。

---

## 3. 需要新增或修改的文件

本次只允许在 `week06/` 中新增文件。

推荐新增：

```text
week06/
├─ simple_agent.py                  # Agent 主程序，后续实现
├─ local_knowledge.md               # 本地知识工具读取的数据源
├─ test_simple_agent.py             # Agent 测试脚本，后续实现
└─ week06_agent_development_plan.md # 本开发计划
```

本次已准备：

- `week06/week06_agent_development_plan.md`
- `week06/local_knowledge.md`

后续代码生成时再新增：

- `week06/simple_agent.py`
- `week06/test_simple_agent.py`

---

## 4. 不应该修改的文件

不要修改：

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
.claude/
.idea/
__pycache__/
```

尤其不要修改：

- `week05/qa_assistant_structured.py`
- `week04/qa_assistant_lc.py`
- `config.py`
- `requirements.txt`

---

## 5. 推荐实现方案

采用“规则判断 + 一个本地工具 + Week05 聊天机器人组织回答”的轻量 Agent 方案。

Agent 固定流程：

1. 接收用户输入。
2. 判断是否需要调用工具。
3. 如需要，调用唯一工具读取 `week06/local_knowledge.md`。
4. 按固定格式输出最终结果。

固定输出格式：

```text
结论：
...

依据：
...

是否调用工具：
是/否
```

该方案符合需求：

- 不依赖现成外部 Agent 平台。
- 工具是本地文件读取工具。
- Agent 自己负责判断、调用、组织回答。
- 功能边界清楚，便于测试和验收。

不推荐方案：

- 不使用 LangChain Agent、AutoGPT、CrewAI 等现成 Agent 框架。
- 不引入向量数据库。
- 不做复杂 RAG。
- 不拆分大量模块。
- 不修改 Week05 原始实现。

---

## 6. 每个文件具体要做什么

### 6.1 `week06/simple_agent.py`

后续实现 Agent 主程序。

建议包含以下函数：

- `should_call_tool(user_input: str) -> bool`
  - 判断用户问题是否需要额外资料。

- `read_local_knowledge(path: str) -> dict`
  - 读取 `week06/local_knowledge.md`。
  - 成功返回资料内容。
  - 失败返回错误信息，不抛出到 CLI 顶层。

- `build_agent_prompt(user_input, tool_result, need_tool, tool_error) -> str`
  - 组织给 Week05 聊天机器人的提示词。
  - 明确要求最终输出三段格式。

- `run_agent(user_input: str, ...) -> str`
  - 执行完整 Agent 流程。

- `main()`
  - 支持 CLI 单次运行。
  - 可选支持无参数时进入交互式输入。

推荐 CLI：

```bash
python week06/simple_agent.py "请根据本地资料说明 Agent 的工作流程"
```

### 6.2 `week06/local_knowledge.md`

本地工具读取的数据源。

内容应包含：

- 简单 Agent 的定义。
- Agent 的四步流程。
- 工具调用条件。
- 输出格式。
- 失败处理方式。
- 功能边界。

### 6.3 `week06/test_simple_agent.py`

后续实现测试脚本。

优先覆盖不依赖模型的逻辑：

- 工具调用判断。
- 本地文件读取成功。
- 本地文件读取失败。
- 固定输出格式。
- 完整流程至少执行一次。

可选测试：

- 如果本地 Ollama 或 DeepSeek 可用，再测试真实模型回答。

---

## 7. Agent 输入格式、工具调用条件、输出格式和失败处理

### 输入格式

Agent 接收一段用户自然语言输入：

```text
请根据本地资料说明 Agent 的工作流程
```

CLI 中作为单个字符串参数传入：

```bash
python week06/simple_agent.py "请根据本地资料说明 Agent 的工作流程"
```

### 工具调用条件

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

- `请根据本地资料说明 Agent 的工作流程` -> 调用工具。
- `这个 Agent 的工具边界是什么` -> 调用工具。
- `什么是 Prompt Template` -> 不调用工具。

### 工具定义

只允许一个工具：

```text
read_local_knowledge
```

作用：

- 读取 `week06/local_knowledge.md`。
- 返回文本内容。
- 不访问网络。
- 不读取其他目录。
- 不修改任何文件。

### 输出格式

Agent 最终输出固定为：

```text
结论：
<面向用户的最终回答>

依据：
<说明是否使用了本地资料，以及使用了哪些信息>

是否调用工具：
<是 / 否 / 是，但失败>
```

### 失败处理

如果工具失败，例如 `local_knowledge.md` 不存在或读取失败：

```text
结论：
未获取到资料。以下是当前可回答部分：...

依据：
本地工具读取失败：<错误信息>

是否调用工具：
是，但失败
```

---

## 8. 给 Claude 生成代码时应该提供的上下文

```text
请只在 week06 文件夹中新增代码文件，不要修改 week03、week04、week05、requirements.txt、config.py 或其他已有文件。

当前项目已有：
- week05/qa_assistant_structured.py：结构化聊天机器人，基于 week04，支持 local Ollama 和 cloud DeepSeek。
- week05/test_structured_output.py：测试结构化输出、prompt、history。
- week06/local_knowledge.md：本地知识工具的数据源。
- week06/week06_agent_development_plan.md：本次 Agent 开发计划。

任务：
基于 week05 的聊天机器人，在 week06 中实现一个简单可运行 Agent 原型。

Agent 固定流程：
1. 接收用户输入。
2. 判断是否需要调用工具。
3. 如需要，只调用一个本地工具读取 week06/local_knowledge.md。
4. 固定输出三部分：结论、依据、是否调用工具。

工具调用条件：
用户输入包含“资料、知识库、文档、根据本地、本地资料、week06、Agent、工具、流程、依据”等关键词时调用工具，否则不调用。

失败处理：
如果本地知识文件读取失败，不崩溃，输出“未获取到资料 + 当前可回答部分”，并说明工具调用失败原因。

需要新增：
- week06/simple_agent.py
- week06/test_simple_agent.py

限制：
- 不使用 LangChain Agent、AutoGPT、CrewAI 等现成 Agent 平台。
- 不新增复杂依赖。
- 不大范围重构。
- 不修改 week05 原文件。
- 测试优先覆盖规则判断、工具读取、失败处理、固定输出格式。
```

---

## 9. 测试与验收标准

### 9.1 基础测试

后续实现测试脚本后运行：

```bash
python week06/test_simple_agent.py
```

应通过：

- `should_call_tool()` 对工具关键词返回 True。
- 普通问题返回 False。
- `read_local_knowledge()` 能读取 `week06/local_knowledge.md`。
- 文件缺失时不崩溃。
- 最终输出包含三段标题：
  - `结论：`
  - `依据：`
  - `是否调用工具：`

### 9.2 工具路径验收

```bash
python week06/simple_agent.py "请根据本地资料说明 Agent 的工作流程"
```

预期：

```text
结论：
说明简单 Agent 的工作流程。

依据：
使用了 week06/local_knowledge.md 中的本地资料。

是否调用工具：
是
```

### 9.3 非工具路径验收

```bash
python week06/simple_agent.py "什么是 Prompt Template？"
```

预期：

```text
结论：
回答 Prompt Template 的基本概念。

依据：
未调用本地工具，基于模型当前能力回答。

是否调用工具：
否
```

### 9.4 失败路径验收

临时让工具读取一个不存在的文件，预期：

```text
结论：
未获取到资料。以下是当前可回答部分：...

依据：
本地工具读取失败：...

是否调用工具：
是，但失败
```

### 9.5 文件边界验收

运行：

```bash
git diff -- week03 week04 week05 requirements.txt config.py
```

预期：

- 无输出。
- 只允许 `week06/` 中出现新增文件。

---

## 10. 推荐开发顺序

1. 创建 `week06/simple_agent.py`。
2. 实现 `should_call_tool()`。
3. 实现 `read_local_knowledge()`。
4. 实现固定格式输出组装。
5. 接入 Week05 聊天机器人能力。
6. 创建 `week06/test_simple_agent.py`。
7. 先跑无模型依赖测试。
8. 再跑一次真实 CLI 示例。
9. 检查只修改了 `week06/`。

