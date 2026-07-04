# Week04 执行 Plan：LangChain 基础学习与现有脚本重构

## 一、知识库链接

本周学习与实现参考以下知识库：

1. [LangChain Demo实践｜基础知识｜简单易学](https://zhuanlan.zhihu.com/p/1993697472309121564)
2. [【Python版2026 从零学Langchain 1.x】（一）快速开始和LCEL](https://zhuanlan.zhihu.com/p/2000239845751137469)
3. [Datawhale LLM Cookbook](https://github.com/datawhalechina/llm-cookbook)
4. [Datawhale LLM Universe](https://github.com/datawhalechina/llm-universe)

---

## 二、任务目标

本周任务是基于 Week03 已完成的模型调用脚本，学习 LangChain 基础组件，并完成第一次 LangChain 重构。

核心目标：

1. 学习 `Model`、`Messages`、`Prompt Template` 等 LangChain 核心组件。
2. 使用 LangChain 重构现有模型调用脚本。
3. 完成一个支持连续对话的最小聊天机器人。
4. 要求 Agent 在重构前先只读现有代码与配置，再给出改造方案。
5. 保留 Week03 原始脚本，不直接覆盖原文件。
6. 新增 LangChain 版本脚本，优先保证最小可运行。

---

## 三、执行原则

### 1. 先只读，后改造

Agent 第一阶段只能只读项目，不允许修改、删除、移动、新建任何文件。

允许操作：

```text
查看项目结构
阅读 README.md
阅读 qa_assistant.py
阅读 langchain_migration_guide.md
阅读 学习任务.md
查看 requirements.txt 或相关配置文件
使用 grep / rg / cat / sed 等只读命令分析代码
使用 git status 查看当前工作区状态
```

禁止操作：

```text
直接编辑文件
自动格式化代码
安装依赖
运行会产生缓存或修改文件的命令
删除或覆盖原有 Week03 脚本
```

只读阶段结束后，必须先输出改造方案，经确认后再开始修改代码。

---

### 2. 不推倒重写，做等价迁移

现有 Week03 脚本已经具备较清晰的三阶段结构：

```text
输入处理
→ 模型调用
→ 输出处理
```

Week04 的重点不是重新设计项目，而是把已有的手写模型调用逻辑迁移到 LangChain：

```text
手写 messages
→ ChatPromptTemplate

手写 HTTP 请求
→ ChatOllama / ChatDeepSeek

手写 call_api
→ model.invoke()

手写 call_api_stream
→ model.stream()

手写多轮 messages 累积
→ history / MessagesPlaceholder / RunnableWithMessageHistory
```

---

### 3. 保留原始脚本，新增 LangChain 版本

不直接覆盖原始 `qa_assistant.py`。

建议新增：

```text
qa_assistant_lc.py
```

用于保存 Week04 的 LangChain 重构版本。

可选新增：

```text
langchain_smoke_test.py
week04_demo.md
requirements_langchain.txt
```

---

## 四、只读调查阶段执行内容

Agent 先阅读现有项目，并输出以下内容。

### 1. 当前项目结构

需要列出当前目录中的关键文件，例如：

```text
qa_assistant.py
README.md
langchain_migration_guide.md
学习任务.md
requirements.txt
config.py 或其他配置文件
```

### 2. 现有模型调用链路

需要梳理当前脚本的调用流程，例如：

```text
parse_args()
→ validate_input()
→ build_system_prompt()
→ build_messages()
→ build_payload()
→ call_api() / call_api_stream()
→ safe_invoke_nonstream_only() / safe_invoke_stream_only()
→ single_shot() / interactive_session()
```

### 3. 可复用组件清单

需要找出可直接迁移到 LangChain 版本的组件：

```text
ROLE_PROMPTS
DEFAULT_TEMPERATURE
DEFAULT_MAX_TOKENS
DEFAULT_PROVIDER
DEFAULT_DEVICE
PROVIDER_INFO
CLOUD_MODEL
LOCAL_MODEL
API_KEY 读取方式
CLI 参数解析逻辑
交互式输入循环
错误处理思路
保存结果逻辑
```

### 4. LangChain 迁移映射表

需要输出类似以下映射：

| Week03 当前实现 | Week04 LangChain 实现 |
|---|---|
| `build_messages()` | `ChatPromptTemplate` |
| `ROLE_PROMPTS` | `ChatPromptTemplate.from_messages()` |
| `build_payload()` | `ChatOllama` / `ChatDeepSeek` 初始化参数 |
| `call_api()` | `model.invoke()` |
| `call_api_stream()` | `model.stream()` |
| `interactive_session()` | 手动 history 或 `RunnableWithMessageHistory` |
| `messages` 列表 | `SystemMessage` / `HumanMessage` / `AIMessage` |
| `provider` 分发 | `get_model()` 模型工厂 |
| `--device cpu/gpu` | `ChatOllama(num_gpu=0/-1)` |

### 5. 拟新增或修改文件

只读调查后，需要明确说明计划新增或修改哪些文件：

```text
新增：qa_assistant_lc.py
新增：langchain_smoke_test.py
新增：week04_demo.md
可能修改：requirements.txt
不修改：qa_assistant.py
```

### 6. 风险点与回滚方案

需要提前说明可能遇到的问题：

```text
Ollama 服务未启动
本地模型 llama3.2:1b 未拉取
DeepSeek API Key 缺失或失效
LangChain 包版本不兼容
ChatOllama / ChatDeepSeek 参数名变化
流式输出格式与原脚本不完全一致
```

回滚原则：

```text
不覆盖 qa_assistant.py
新增文件独立实现
出错时删除新增文件即可恢复 Week03 状态
所有改动前先 git status
所有阶段成果单独 commit
```

---

## 五、核心组件学习执行内容

### 1. 学习 Model

需要掌握：

```text
ChatOllama
ChatDeepSeek
model.invoke()
model.stream()
temperature
max_tokens / num_predict
provider 切换
device 切换
```

实现目标：

```python
def get_model(provider: str, temperature: float, max_tokens: int, device: str = "gpu"):
    if provider == "local":
        return ChatOllama(
            model=LOCAL_MODEL,
            temperature=temperature,
            num_predict=max_tokens,
            num_gpu=-1 if device == "gpu" else 0,
        )

    if provider == "cloud":
        return ChatDeepSeek(
            model=CLOUD_MODEL,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=API_KEY,
        )

    raise ValueError(f"未知 provider: {provider}")
```

验收目标：

```bash
python qa_assistant_lc.py "什么是机器学习？"
python qa_assistant_lc.py -d cpu "什么是机器学习？"
python qa_assistant_lc.py -p cloud "什么是机器学习？"
```

---

### 2. 学习 Messages

需要掌握：

```text
SystemMessage
HumanMessage
AIMessage
多轮历史消息
history 传递
上下文清空
```

最小消息结构：

```python
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

messages = [
    SystemMessage(content="你是一个智能问答助手"),
    HumanMessage(content="什么是机器学习？"),
    AIMessage(content="机器学习是..."),
    HumanMessage(content="那深度学习呢？"),
]
```

连续对话中，每一轮需要追加：

```python
history.append(HumanMessage(content=user_input))
history.append(AIMessage(content=answer))
```

---

### 3. 学习 Prompt Template

需要掌握：

```text
ChatPromptTemplate
MessagesPlaceholder
system prompt
human prompt
question 占位符
history 占位符
```

单轮 Prompt Template：

```python
from langchain_core.prompts import ChatPromptTemplate

def build_prompt_template(role: str, custom_system: str | None = None):
    system_text = custom_system or ROLE_PROMPTS.get(role, ROLE_PROMPTS["default"])

    return ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "{question}"),
    ])
```

多轮 Prompt Template：

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def build_chat_prompt_template(role: str, custom_system: str | None = None):
    system_text = custom_system or ROLE_PROMPTS.get(role, ROLE_PROMPTS["default"])

    return ChatPromptTemplate.from_messages([
        ("system", system_text),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
```

---

## 六、LangChain 重构执行内容

### 1. 建立 LangChain 版本脚本

新增：

```text
qa_assistant_lc.py
```

基本结构：

```text
配置区
角色提示词区
CLI 参数解析区
模型工厂 get_model()
Prompt Template 构建函数
Chain 构建函数
单次问答 single_shot_lc()
连续对话 interactive_session_lc()
main() 入口
```

---

### 2. 迁移配置参数

从 Week03 脚本中迁移：

```python
CLOUD_MODEL = "deepseek-chat"
LOCAL_MODEL = "llama3.2:1b"
DEFAULT_PROVIDER = "local"
DEFAULT_DEVICE = "gpu"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
```

保留 provider 逻辑：

```text
local → Ollama
cloud → DeepSeek
```

保留 device 逻辑：

```text
gpu → num_gpu=-1
cpu → num_gpu=0
```

---

### 3. 迁移角色提示词

保留原有角色：

```text
default
teacher
coder
doctor
translator
```

迁移方式：

```python
ROLE_PROMPTS = {
    "default": "...",
    "teacher": "...",
    "coder": "...",
    "doctor": "...",
    "translator": "...",
}
```

再通过 `ChatPromptTemplate` 使用。

---

### 4. 构建单次问答 Chain

实现：

```python
from langchain_core.output_parsers import StrOutputParser

def build_chain(provider, role, temperature, max_tokens, device="gpu", system=None):
    prompt = build_prompt_template(role, system)
    model = get_model(provider, temperature, max_tokens, device)
    return prompt | model | StrOutputParser()
```

调用：

```python
chain.invoke({"question": question})
```

支持命令：

```bash
python qa_assistant_lc.py "什么是 Prompt Template？"
python qa_assistant_lc.py --role teacher "什么是 Messages？"
python qa_assistant_lc.py --system "你是一个极简助手" "解释 RAG"
```

---

### 5. 构建连续对话机器人

优先采用手动维护 history 的方式，保证最小可运行。

核心逻辑：

```python
history = []

while True:
    user_input = input("[Q] ").strip()

    if user_input.lower() in ("q", "quit", "exit"):
        break

    if user_input == "/clear":
        history.clear()
        print("[OK] 对话上下文已清空")
        continue

    result = chain.invoke({
        "history": history,
        "question": user_input,
    })

    print("[A]", result)

    history.append(HumanMessage(content=user_input))
    history.append(AIMessage(content=result))
```

连续对话必须验证：

```text
用户：Python 如何读取文件？
助手：可以使用 open() ...

用户：那如何逐行读取？
助手：可以使用 for line in file ...
```

第二轮需要能够理解“那”指代上一轮的“读取文件”。

---

### 6. 支持流式输出

保留 `--stream` 参数。

非流式：

```python
answer = chain.invoke({"question": question})
print(answer)
```

流式：

```python
for chunk in chain.stream({"question": question}):
    print(chunk, end="", flush=True)
```

连续对话中的流式输出需要累积完整 answer，方便写入 history：

```python
answer = ""

for chunk in chain.stream({
    "history": history,
    "question": user_input,
}):
    print(chunk, end="", flush=True)
    answer += chunk

history.append(HumanMessage(content=user_input))
history.append(AIMessage(content=answer))
```

---

### 7. 保留必要 CLI 参数

必须保留：

```text
question
--provider / -p
--device / -d
--role / -r
--system / -s
--temperature / -t
--max-tokens / -m
--stream
```

可以暂缓：

```text
/save
/stats
复杂 token 统计
完整 fallback
复杂会话命令系统
```

交互模式建议保留：

```text
/help
/clear
/role
/provider
/stream
q / quit / exit
```

---

## 七、验收测试

### 1. 单次问答测试

```bash
python qa_assistant_lc.py "什么是 LangChain Model？"
```

通过标准：

```text
可以正常输出中文回答
```

---

### 2. 角色测试

```bash
python qa_assistant_lc.py --role teacher "什么是 Prompt Template？"
python qa_assistant_lc.py --role coder "用 Python 写一个读取文件的例子"
```

通过标准：

```text
teacher 角色回答更适合教学
coder 角色能给出代码解释
```

---

### 3. 本地 CPU/GPU 测试

```bash
python qa_assistant_lc.py -d cpu "什么是 RAG？"
python qa_assistant_lc.py -d gpu "什么是 RAG？"
```

通过标准：

```text
CPU/GPU 参数能正常传递给 ChatOllama
```

---

### 4. 云端测试

```bash
python qa_assistant_lc.py -p cloud "什么是 Messages？"
```

通过标准：

```text
能成功调用 DeepSeek
如果失败，需要输出 API Key 或网络相关错误提示
```

---

### 5. 连续对话测试

```bash
python qa_assistant_lc.py
```

测试对话：

```text
[Q] Python 如何读取文件？
[A] ...

[Q] 那如何逐行读取？
[A] ...
```

通过标准：

```text
第二轮回答能结合上一轮上下文
```

---

### 6. 流式输出测试

```bash
python qa_assistant_lc.py --stream "写一段介绍 LangChain 的话"
```

通过标准：

```text
回答可以逐段或逐 token 输出
```

---

### 7. 清空上下文测试

交互模式中输入：

```text
/clear
```

通过标准：

```text
上下文被清空
后续问题不再依赖清空前的对话
```

---

## 八、最终交付物

最终应提交以下内容：

```text
qa_assistant_lc.py
langchain_smoke_test.py
week04_demo.md
必要时更新 requirements.txt
```

其中：

```text
qa_assistant_lc.py
```

用于保存 LangChain 重构后的主脚本。

```text
langchain_smoke_test.py
```

用于验证 LangChain 依赖、模型调用和 Prompt Template 是否可用。

```text
week04_demo.md
```

用于记录本周学习内容、重构说明、运行命令、测试结果和问题记录。

---

## 九、week04_demo.md 内容模板

```markdown
# Week04 LangChain 重构记录

## 1. 本周知识库

- https://zhuanlan.zhihu.com/p/1993697472309121564
- https://zhuanlan.zhihu.com/p/2000239845751137469
- https://github.com/datawhalechina/llm-cookbook
- https://github.com/datawhalechina/llm-universe

## 2. 学习内容

本周学习了 LangChain 的三个核心组件：

1. Model：负责封装不同模型后端。
2. Messages：负责表达 system、human、AI 等多轮消息。
3. Prompt Template：负责将提示词模板化，替代手动拼接 messages。

## 3. 重构前后对比

| Week03 | Week04 LangChain |
|---|---|
| build_messages | ChatPromptTemplate |
| build_payload | ChatOllama / ChatDeepSeek |
| call_api | model.invoke |
| call_api_stream | model.stream |
| messages 手动拼接 | history / MessagesPlaceholder |

## 4. 新增文件

- qa_assistant_lc.py
- langchain_smoke_test.py
- week04_demo.md

## 5. 运行命令

```bash
python qa_assistant_lc.py "什么是 LangChain？"
python qa_assistant_lc.py --role teacher "什么是 Prompt Template？"
python qa_assistant_lc.py -p cloud "什么是 Messages？"
python qa_assistant_lc.py
```

## 6. 连续对话测试

```text
用户：Python 如何读取文件？
助手：可以使用 open() ...

用户：那如何逐行读取？
助手：可以使用 for line in file ...
```

## 7. 遇到的问题

- Ollama 服务未启动
- DeepSeek API Key 缺失
- LangChain 版本不兼容
- 流式输出格式差异

## 8. 后续改进

- 补齐更完整的会话命令
- 恢复保存对话功能
- 增加统一错误处理
- 为 Week05 的 Prompt Template 和输出格式整理做准备
```

---

## 十、Git 提交要求

执行前：

```bash
git status
```

只读调查后建议提交：

```bash
git add Week04_只读调查报告.md
git commit -m "week04: add readonly investigation report"
```

LangChain 重构完成后提交：

```bash
git add qa_assistant_lc.py langchain_smoke_test.py week04_demo.md requirements.txt
git commit -m "week04: migrate qa assistant to LangChain"
```

---

## 十一、给 Agent 的执行提示词

```text
你是我的 AI 编程协作 Agent。现在执行 Week04 LangChain 重构任务。

知识库参考：
1. https://zhuanlan.zhihu.com/p/1993697472309121564
2. https://zhuanlan.zhihu.com/p/2000239845751137469
3. https://github.com/datawhalechina/llm-cookbook
4. https://github.com/datawhalechina/llm-universe

任务目标：
1. 学习 Model、Messages、Prompt Template 等 LangChain 核心组件。
2. 使用 LangChain 重构现有模型调用脚本。
3. 完成一个支持连续对话的最小聊天机器人。
4. 重构前必须先只读现有代码与配置，再给出改造方案。

第一阶段只读，不允许修改任何文件。
请先阅读：
- 学习任务.md
- README.md
- qa_assistant.py
- langchain_migration_guide.md
- requirements.txt 或相关配置文件

只读阶段结束后，请输出：
1. 当前项目结构。
2. 现有模型调用链路。
3. 可复用组件清单。
4. LangChain 迁移映射表。
5. 拟新增或修改的文件列表。
6. 风险点与回滚方案。

确认后再进入代码修改阶段。

代码改造原则：
1. 不覆盖原始 qa_assistant.py。
2. 新建 qa_assistant_lc.py 作为 LangChain 版本。
3. 优先实现最小可运行版本。
4. 保留本地 Ollama 和云端 DeepSeek 双后端。
5. 保留 --provider、--device、--role、--system、--temperature、--max-tokens、--stream 等核心参数。
6. 完成单次问答和连续对话两种模式。
7. 支持 /clear、/help、q/quit/exit 等最小交互命令。
8. 最终输出运行说明和测试记录。
```

---

## 十二、完成标准

Week04 完成标准：

1. Agent 已先完成只读调查，没有直接改代码。
2. 已输出清晰的迁移方案。
3. 已学习并说明 Model、Messages、Prompt Template 的作用。
4. 已用 `ChatPromptTemplate` 替代手动拼接 prompt。
5. 已用 `ChatOllama` / `ChatDeepSeek` 替代手写 HTTP 请求。
6. 已完成单次问答 LangChain 版本。
7. 已完成支持上下文的最小连续对话机器人。
8. 已保留 Week03 原始脚本，避免破坏已有成果。
9. 已整理 `week04_demo.md`。
10. 已完成 Git 提交。

---

## 十三、核心原则

```text
先只读调查，再输出方案。
先等价迁移，再局部优化。
先跑通主链路，再补齐高级功能。
不覆盖旧代码，不破坏 Week03 成果。
```
