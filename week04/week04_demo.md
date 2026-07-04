# Week04 LangChain 重构记录

## 1. 本周知识库

- [LangChain Demo实践｜基础知识｜简单易学](https://zhuanlan.zhihu.com/p/1993697472309121564)
- [Python版2026 从零学Langchain 1.x（一）快速开始和LCEL](https://zhuanlan.zhihu.com/p/2000239845751137469)
- [Datawhale LLM Cookbook](https://github.com/datawhalechina/llm-cookbook)
- [Datawhale LLM Universe](https://github.com/datawhalechina/llm-universe)

## 2. 学习内容

本周学习了 LangChain 的三个核心组件：

### 2.1 Model（模型）

LangChain 将不同的模型后端封装为统一的 `ChatModel` 接口：

- **ChatOllama** — 本地 Ollama 模型封装。支持 `num_gpu` 参数控制 GPU 加速（`-1`=自动 GPU，`0`=纯 CPU）。
- **ChatDeepSeek** — 云端 DeepSeek API 封装。兼容 OpenAI SDK 格式，通过 `api_key` 鉴权。
- **model.invoke()** — 非流式调用，一次性返回完整结果。
- **model.stream()** — 流式调用，逐 token 返回 `AIMessageChunk`。

工厂模式实现双后端一键切换：

```python
def get_model(provider: str, temperature: float, max_tokens: int, device: str = "gpu"):
    if provider == "local":
        return ChatOllama(model=LOCAL_MODEL, temperature=temperature,
                          num_predict=max_tokens, num_gpu=-1 if device == "gpu" else 0)
    if provider == "cloud":
        return ChatDeepSeek(model_name=CLOUD_MODEL, temperature=temperature,
                            max_tokens=max_tokens, api_key=API_KEY)
```

### 2.2 Messages（消息）

LangChain 使用类型化消息替代手写 `{role, content}` 字典：

| 消息类型 | 用途 | 原 Week03 等价 |
|---|---|---|
| `SystemMessage(content=...)` | 系统提示词 | `{"role": "system", "content": ...}` |
| `HumanMessage(content=...)` | 用户输入 | `{"role": "user", "content": ...}` |
| `AIMessage(content=...)` | 模型回复 | `{"role": "assistant", "content": ...}` |

多轮对话中，history 就是 `List[BaseMessage]`，每次追加 `HumanMessage` 和 `AIMessage`：

```python
history = [SystemMessage(content="你是一个助手")]
history.append(HumanMessage(content="Python 如何读取文件？"))
# ... 调用模型 ...
history.append(AIMessage(content="可以使用 open() ..."))
```

### 2.3 Prompt Template（提示词模板）

`ChatPromptTemplate` 替代手写 `build_messages()`：

- **单轮模板** — `("system", "...")` + `("human", "{question}")`
- **多轮模板** — 加入 `MessagesPlaceholder(variable_name="history")` 承载对话历史
- **角色切换** — `ROLE_PROMPTS` 字典 + 模板动态构建

```python
# 单轮
prompt = ChatPromptTemplate.from_messages([
    ("system", system_text),
    ("human", "{question}"),
])

# 多轮
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_text),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])
```

### 2.4 LCEL (LangChain Expression Language)

使用 `|` 管道符串联组件。主脚本为了读取 `AIMessage` / `AIMessageChunk` 中的 token 元数据，实际保留 `prompt | model`，再由调用方提取 `content`；如果只需要纯文本，也可以追加 `StrOutputParser()`：

```python
chain = prompt | model
response = chain.invoke({"question": "什么是机器学习？"})
result = response.content

# 流式
for chunk in chain.stream({"question": "什么是机器学习？"}):
    print(chunk.content, end="", flush=True)
```

## 3. 重构前后对比

| Week03 原始实现 | Week04 LangChain 实现 | 说明 |
|---|---|---|
| `build_messages()` / `build_system_prompt()` | `ChatPromptTemplate.from_messages([...])` | 模板替代手动拼接 |
| `ROLE_PROMPTS` dict | `ChatPromptTemplate` + `build_system_text()` | 语义不变, 通过模板使用 |
| `build_payload()` + 手写 dict | `ChatOllama(...)` / `ChatDeepSeek(...)` | 模型封装内置请求构建 |
| `call_api()` | `model.invoke()` | 一行替代 HTTP 调用 |
| `call_api_stream()` + SSE/JSON-lines 解析 | `model.stream()` | 无需手动解析流式协议 |
| `messages: List[Dict]` | `List[BaseMessage]` (SystemMessage/HumanMessage/AIMessage) | 类型化消息 |
| `interactive_session()` 手动维护 dict 列表 | `interactive_session_lc()` + `MessagesPlaceholder("history")` | history 作为模板变量传入 |
| `provider` if/else 分发 | `get_model()` 工厂函数 | 语义等价, 更清晰 |
| `--device cpu/gpu` → `num_gpu` 手动映射 | `ChatOllama(num_gpu=0/-1)` | 参数直接传递给模型 |
| `safe_invoke_*()` 错误包装 | `_handle_langchain_error()` | 捕获底层异常并输出中文提示 |
| `requests` HTTP 调用 | LangChain 内部处理 | 消除手动 HTTP 依赖 |

## 4. 新增文件

```
week04/
├── qa_assistant_lc.py          # LangChain 重构版主脚本 (866行)
├── langchain_smoke_test.py     # 冒烟测试 (依赖 + 连通性 + LCEL)
└── week04_demo.md              # 本文档
```

项目根目录还提供 `requirements.txt`，作为 Week03/Week04 共用依赖入口。

## 5. 运行命令

### 冒烟测试

```bash
# 安装依赖
pip install -r requirements.txt

# 全部测试
python week04/langchain_smoke_test.py

# 快速模式 (仅依赖 + Prompt Template)
python week04/langchain_smoke_test.py --quick

# 仅本地 Ollama
python week04/langchain_smoke_test.py --local

# 仅云端 DeepSeek
python week04/langchain_smoke_test.py --cloud
```

### 单次问答

```bash
# 本地 Ollama (默认, GPU加速)
python week04/qa_assistant_lc.py "什么是 LangChain？"
python week04/qa_assistant_lc.py --stream "介绍一下 Prompt Template"

# 本地 Ollama (CPU 推理)
python week04/qa_assistant_lc.py -d cpu "什么是 RAG？"

# 角色切换
python week04/qa_assistant_lc.py --role teacher "什么是 Messages？"
python week04/qa_assistant_lc.py --role coder "用 Python 写一个读取文件的例子"

# 自定义 system prompt
python week04/qa_assistant_lc.py --system "你是一个极简助手，回答不超过20字" "解释 RAG"

# 云端 DeepSeek
python week04/qa_assistant_lc.py -p cloud "什么是机器学习？"
python week04/qa_assistant_lc.py -p cloud --stream --role teacher "解释量子计算"

# 保存结果
python week04/qa_assistant_lc.py "什么是深度学习？" --save answer.txt
```

### 连续对话

```bash
# 本地交互模式 (默认 GPU)
python week04/qa_assistant_lc.py

# 本地 CPU
python week04/qa_assistant_lc.py -d cpu

# 云端交互
python week04/qa_assistant_lc.py -p cloud

# 云端 + 流式 + coder 角色
python week04/qa_assistant_lc.py -p cloud --stream --role coder
```

## 6. 连续对话测试

交互模式中验证上下文记忆：

```
[1] [Q] Python 如何读取文件？
[1] [A]
----------------------------------------
可以使用 open() 函数打开文件...

[2] [Q] 那如何逐行读取？
[2] [A]
----------------------------------------
可以使用 for line in file: 来逐行读取...
```

第二轮 "那" 指代上一轮的"读取文件" — 模型通过 `MessagesPlaceholder("history")` 获取了完整上下文。

### 会话指令

| 指令 | 说明 |
|------|------|
| `/help` | 显示帮助 |
| `/role <name>` | 切换角色 (default/teacher/coder/doctor/translator) |
| `/stream` | 切换流式输出 |
| `/clear` | 清空对话上下文 |
| `/provider <name>` | 切换后端 (local/cloud) |
| `/device <cpu\|gpu>` | 切换本地推理设备 |
| `/save [path]` | 保存对话记录 |
| `/stats` | 查看会话统计 |
| `q` / `quit` / `exit` | 退出 |

## 7. 遇到的问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| ModuleNotFoundError (langchain_core 等) | LangChain 包未安装 | `pip install -r requirements.txt` |
| Ollama 服务未启动 | 本地 Ollama 未运行 | `ollama serve` 启动服务 |
| 模型未找到 (llama3.2:1b) | 模型未拉取 | `ollama pull llama3.2:1b` |
| DeepSeek API Key 缺失/无效 | config.py 未配置 | 在 [platform.deepseek.com](https://platform.deepseek.com) 获取 Key |
| ChatDeepSeek 参数名差异 | LangChain 1.x 使用 `model_name` 而非 `model` | 查阅 API 签名后修正 |
| 流式输出格式差异 | LangChain 内部处理, 无需手动解析 | 比 Week03 更简洁 |

## 8. 代码行数对比

| 文件 | 行数 | 说明 |
|------|------|------|
| `week03/qa_assistant.py` | 933 行 | 手写 HTTP 请求 + SSE/JSON-lines 解析 |
| `week04/qa_assistant_lc.py` | 866 行 | LangChain 封装, 并保留交互指令、保存、统计与 token 元数据展示 |

减少的代码主要集中在:
- 手写 `_build_cloud_payload()` / `_build_local_payload()` (~30行)
- 手写 `_call_cloud_api()` / `_call_local_api()` (~60行)
- 手写 `_call_cloud_api_stream()` / `_call_local_api_stream()` (~70行)
- 流式协议解析 (SSE / JSON-lines) (~40行)
- `safe_invoke_nonstream_only()` / `safe_invoke_stream_only()` 重复包装 (~50行)

## 9. 后续改进

- 补齐 `RunnableWithMessageHistory` 自动管理会话 (当前手动维护 history)
- 恢复 `/save` 和 `/stats` 的完整 token 统计
- 增加 `chain.with_fallbacks()` 实现后端自动切换
- 为 Week05 的 Prompt Template 和输出格式整理做准备
- 考虑用 Pydantic 模型替代 argparse 处理 CLI 参数
