# Week03 → Week04 LangChain 迁移参考

> 本文档整理 `qa_assistant.py` 中所有可复用的提示词模板、参数配置和架构设计，
> 供 Week04 LangChain 重构时直接对照搬运。

---

## 一、提示词模板 (Prompt Templates)

### 1.1 角色预设 → ChatPromptTemplate

当前 `ROLE_PROMPTS` 字典 (5 个角色)，迁移后每个角色对应一个 `ChatPromptTemplate`:

| 角色 key | 功能 | System Prompt |
|----------|------|--------------|
| `default` | 通用问答助手 | *你是一个智能问答助手，擅长用中文清晰、准确地回答问题。如果问题涉及复杂概念，请用通俗易懂的语言解释。* |
| `teacher` | 中文教师 | *你是一位经验丰富的中文教师。请用通俗易懂的语言讲解概念，配合生活化的例子和类比，帮助学生真正理解。回答控制在 300 字以内。* |
| `coder` | 软件工程师 | *你是一位资深软件工程师。请用中文回答技术问题，提供清晰的代码示例，并解释关键逻辑。代码块使用 Markdown 格式。* |
| `doctor` | 医学顾问 | *你是一位医学顾问。请用中文提供专业的健康建议，引用权威医学知识，并始终提醒用户必要时咨询医生。回答应清晰、准确、有依据。* |
| `translator` | 中英翻译 | *你是一位专业的中英翻译。请准确翻译用户提供的内容，保持原文风格和专业术语的一致性。如果是中文翻译成英文，输出英文；反之输出中文。* |

**LangChain 对照写法:**

```python
from langchain_core.prompts import ChatPromptTemplate

# 为每个角色生成 prompt template
def build_prompt_template(role: str) -> ChatPromptTemplate:
    system_text = ROLE_PROMPTS.get(role, ROLE_PROMPTS["default"])
    return ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("placeholder", "{history}"),   # ← 多轮上下文占位
        ("human", "{question}"),
    ])
```

### 1.2 System Prompt 覆盖机制

当前逻辑: `--system` 参数覆盖 `--role`，迁移后等价于:

```python
# 自定义 system 覆盖角色预设
system_text = custom_system if custom_system else ROLE_PROMPTS[role]
prompt = ChatPromptTemplate.from_messages([
    ("system", system_text),
    ("placeholder", "{history}"),
    ("human", "{question}"),
])
```

---

## 二、模型配置参数

### 2.1 全局参数速查表

| 参数 | Python 变量 / CLI 参数 | 当前默认值 | 作用 | LangChain 等价 |
|------|----------------------|-----------|------|---------------|
| **temperature** | `DEFAULT_TEMPERATURE` / `-t` | `0.7` | 生成温度 0~2 | `model.bind(temperature=0.7)` 或 `model = ChatOllama(temperature=0.7)` |
| **max_tokens** | `DEFAULT_MAX_TOKENS` / `-m` | `1024` | 最大输出 token 数 | Cloud: `model.bind(max_tokens=1024)`, Local: `model.bind(num_predict=1024)` |
| **stream** | `--stream` | `False` | 流式逐字输出 | `model.stream(messages)` |
| **provider** | `DEFAULT_PROVIDER` / `-p` | `local` | 模型后端选择 | `ChatOllama` vs `ChatDeepSeek` 实例切换 |
| **role** | `--role` / `-r` | `default` | 角色预设 | `ChatPromptTemplate` 切换 |
| **system** | `--system` / `-s` | `None` | 自定义 system prompt | 直接传给 `ChatPromptTemplate` |
| **save** | `--save` | `None` | 保存到文件 | 无关 (输出层逻辑) |
| **no_echo** | `--no-echo` | `False` | 隐藏统计行 | 无关 |

### 2.2 后端模型信息

| 后端 | provider 值 | 模型名 | API 端点 | 鉴权 |
|------|-----------|--------|---------|------|
| **本地 CPU** | `local` | `llama3.2:1b` | `http://localhost:11434/api/chat` | 无 |
| **云端** | `cloud` | `deepseek-chat` | `https://api.deepseek.com/v1/chat/completions` | `Bearer <API_KEY>` |

### 2.3 本地模型专用参数

| 参数 | 当前值 | 说明 |
|------|--------|------|
| `LOCAL_NUM_GPU` | `0` | 强制 CPU 推理 (不加载 GPU 层) |
| `LOCAL_NUM_THREAD` | `None` | CPU 线程数 (None = 自动) |
| `REQUEST_TIMEOUT` | `60` | 请求超时秒数 |

**LangChain 等价:**

```python
from langchain_ollama import ChatOllama

model = ChatOllama(
    model="llama3.2:1b",
    temperature=0.7,
    num_predict=1024,
    num_gpu=0,          # CPU 推理
)
```

---

## 三、三阶段流水线架构

### 3.1 当前结构 → LangChain LCEL

```
当前 (qa_assistant.py)           LangChain LCEL
───────────────────────────────────────────────────
Stage 1: 输入处理                  ChatPromptTemplate
  parse_args()                      ↓
  validate_input()                RunnablePassthrough
  build_system_prompt()             ↓
  build_messages()                ChatDeepSeek / ChatOllama
    ↓                               ↓
Stage 2: 模型调用                  StrOutputParser
  build_payload()                   ↓
  call_api() / call_api_stream()  (自定义 output parser)
    ↓
Stage 3: 输出处理
  print_result()
  token 统计 + 存档
```

### 3.2 关键函数 → LangChain 映射

| 当前函数 | 行号 | 职责 | LangChain 等价 |
|---------|------|------|---------------|
| `parse_args()` | ~101 | CLI 参数解析 | 后续可替换为 Pydantic `BaseModel` input schema |
| `build_system_prompt()` | ~319 | 构建 system prompt | `ChatPromptTemplate.from_messages([("system", ...)])` |
| `build_messages()` | ~326 | 单次模式消息构建 | `prompt.invoke({"question": q})` |
| `build_payload()` | ~343 | 按 provider 构建请求体 | 无需 (model 内部处理) |
| `call_api()` | ~360 | 非流式调用分发 | `model.invoke(messages)` |
| `call_api_stream()` | ~368 | 流式调用分发 | `model.stream(messages)` |
| `interactive_session()` | ~600+ | 多轮对话循环 | `RunnableWithMessageHistory` |
| `single_shot()` | ~550+ | 单次问答 | `chain.invoke({"question": q})` |
| `handle_session_command()` | ~202 | 会话指令解析 | 自定义 Tool / 或直接切模型实例 |
| `safe_invoke_*()` | ~533 | 错误处理包装 | `chain.with_fallbacks([...])` |

---

## 四、返回格式 (输出 Schema)

### 4.1 统一 Result Dict

两后端归一化为相同格式，上层代码无需区分:

```python
# 非流式返回
{
    "content": str,           # 模型回复文本
    "model": str,             # 模型名 (deepseek-chat / llama3.2:1b)
    "finish_reason": str,     # "stop" | "length"
    "usage": {
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int,
    },
    "elapsed": float,          # 调用耗时 (秒)
}

# 流式返回
{
    "content": str,            # 累积的完整回复
    "elapsed": float,
}
```

**LangChain 等价:**
- 非流式: `AIMessage(content=..., response_metadata={...})`
- 流式: `AIMessageChunk` 的叠加

### 4.2 输入消息格式 (两后端共用)

```python
# 标准 OpenAI 格式
messages = [
    {"role": "system", "content": "你是..."},
    {"role": "user",   "content": "什么是 ML？"},
    {"role": "assistant", "content": "ML 是..."},   # 多轮历史
    {"role": "user",   "content": "那 DL 呢？"},
]
```

---

## 五、LCEL 重构伪代码

### 5.1 单次问答 Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek

# ── 模型工厂 ──
def get_model(provider: str, temperature: float, max_tokens: int):
    if provider == "local":
        return ChatOllama(
            model="llama3.2:1b",
            temperature=temperature,
            num_predict=max_tokens,
            num_gpu=0,  # CPU 推理
        )
    else:
        return ChatDeepSeek(
            model="deepseek-chat",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=API_KEY,
        )

# ── Prompt ──
def get_prompt(role: str, custom_system: str | None = None):
    system_text = custom_system or ROLE_PROMPTS[role]
    return ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", "{question}"),
    ])

# ── Chain ──
def build_chain(provider, role, temperature, max_tokens):
    prompt = get_prompt(role)
    model = get_model(provider, temperature, max_tokens)
    return prompt | model | StrOutputParser()

# 使用
chain = build_chain("local", "teacher", 0.7, 1024)
result = chain.invoke({"question": "什么是机器学习？"})
```

### 5.2 多轮对话 (带历史记忆)

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

store = {}  # session_id → ChatMessageHistory

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    build_chain(provider, role, temp, max_tokens),
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

# 多轮调用
chain_with_history.invoke(
    {"question": "Python 如何读取文件？"},
    config={"configurable": {"session_id": "user_1"}},
)
chain_with_history.invoke(
    {"question": "那如何逐行读取呢？"},   # ← 记住上文
    config={"configurable": {"session_id": "user_1"}},
)
```

### 5.3 流式输出

```python
# LCEL 流式 (无需手动处理 SSE / JSON lines)
for chunk in chain.stream({"question": "写一首诗"}):
    print(chunk, end="", flush=True)
```

---

## 六、迁移检查清单

| # | 迁移项 | 当前文件 | 状态 |
|---|--------|---------|------|
| 1 | `ROLE_PROMPTS` → `ChatPromptTemplate` × 5 | qa_assistant.py | 5 个模板，含 system + human 占位 |
| 2 | `DEFAULT_TEMPERATURE` / `DEFAULT_MAX_TOKENS` → model 构造参数 | qa_assistant.py | 直接传给 `ChatOllama` / `ChatDeepSeek` |
| 3 | `DEFAULT_PROVIDER = "local"` → `get_model()` 工厂函数 | qa_assistant.py | 返回对应实例 |
| 4 | `LOCAL_NUM_GPU = 0` → `ChatOllama(num_gpu=0)` | qa_assistant.py | 保留 CPU 推理 |
| 5 | `build_messages()` → `prompt.invoke()` | qa_assistant.py | 移除手动拼接 |
| 6 | `build_payload()` / `call_api()` → `model.invoke()` | qa_assistant.py | 移除手动 HTTP 调用 |
| 7 | `call_api_stream()` → `model.stream()` | qa_assistant.py | 移除 SSE/JSON-lines 解析 |
| 8 | `interactive_session()` → `RunnableWithMessageHistory` | qa_assistant.py | 用 session_id 管理上下文 |
| 9 | `handle_session_command()` → 自定义 Tool 或保留 | qa_assistant.py | `/provider` `/role` `/clear` 等 |
| 10 | `safe_invoke_*()` → `chain.with_fallbacks()` | qa_assistant.py | 连接/超时/HTTP 错误处理 |
| 11 | `API_KEY` 从 `config.py` → `ChatDeepSeek(api_key=...)` | qa_assistant.py | 环境变量或 Secrets |
| 12 | 统一 Result Dict → `StrOutputParser` + `AIMessage` | qa_assistant.py | 自动解析 |

---

## 七、依赖对照

```txt
# 当前 (requirements.txt)
requests>=2.28.0

# Week04 追加 (不需要 requests 了)
langchain>=0.3.0
langchain-core>=0.3.0
langchain-ollama>=0.1.0
langchain-deepseek>=0.1.0
```

---

> **文件索引:**
> - [qa_assistant.py](qa_assistant.py) — 当前实现 (863 行)
> - [README.md](README.md) — 使用文档
> - [requirements.txt](requirements.txt) — 依赖清单
> - 项目根 [config.py](../config.py) — API Key
> - 项目根 [cloud_api_demo.py](../cloud_api_demo.py) — DeepSeek 参考实现
> - 项目根 [test.py](../test.py) — Ollama 参考实现
