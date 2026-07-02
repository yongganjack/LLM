# Week03 — CLI 中文问答助手

基于 **本地 Ollama (CPU/GPU) + 云端 DeepSeek 双后端** 的命令行中文问答工具，支持 **单次问答** 和 **连续对话** 两种模式。按 **输入处理 → 模型调用 → 输出** 三阶段流水线设计，后端可一键切换，接口清晰、职责单一，可直接复用到后续 LangChain 重构。

> **默认使用本地 Ollama 模型 + GPU 加速** (`--device gpu`, `num_gpu=-1`)。
> 本地切 CPU: `-d cpu` (`num_gpu=0`)。
> 切换云端: `-p cloud`。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备后端:
#    本地 Ollama (默认) — 先安装 Ollama 并拉取模型: ollama pull llama3.2:1b
#    云端 DeepSeek        — 确保 ../config.py 中 API_KEY 已配置

# 3. 运行
python qa_assistant.py "什么是机器学习？"                      # 本地 GPU (默认)
python qa_assistant.py -d cpu "什么是机器学习？"                # 本地 CPU
python qa_assistant.py -p cloud "什么是机器学习？"              # 云端
```

## 使用方式

### 单次问答 (命令行传参)

```bash
# ── 本地 Ollama (默认, GPU加速) ──
python qa_assistant.py "解释一下量子计算"
python qa_assistant.py --stream --role teacher "什么是 REST API？"
python qa_assistant.py --temperature 0.3 --max-tokens 512 --save result.txt "总结法国大革命"

# ── 本地 Ollama (CPU 推理) ──
python qa_assistant.py -d cpu "解释一下量子计算"
python qa_assistant.py -d cpu --stream --role teacher "什么是 REST API？"

# ── 云端 DeepSeek ──
python qa_assistant.py -p cloud "用 Python 写一个 Hello World"
python qa_assistant.py -p cloud --stream --role teacher "解释量子计算"
python qa_assistant.py -p cloud --save answer.txt "总结人工智能的发展历史"
```

### 连续对话 (交互模式)

不传 question 参数即可进入交互模式，模型记住对话上下文：

```bash
python qa_assistant.py                    # 本地交互 (默认, GPU加速)
python qa_assistant.py -d cpu             # 本地交互 (CPU 推理)
python qa_assistant.py -p cloud           # 云端交互
python qa_assistant.py -p cloud --stream  # 云端 + 流式交互
```

对话示例 (模型会记住上文):

```
[INFO] 后端: Ollama (本地) (llama3.2:1b)
[INFO] 角色: coder  |  流式: OFF  |  Temperature: 0.7  |  Max Tokens: 1024

[1] [Q] Python 中如何读取文件？
[1] [A]
----------------------------------------
在 Python 中可以使用 open() 函数...
--- llama3.2:1b | 1.2s | Token: ... | 结束: stop ---

[2] [Q] 那如何逐行读取呢？                              ← 模型知道"那"指的是文件读取
[2] [A]
----------------------------------------
可以使用 for line in file: ...
```

输入 `q` / `quit` / `exit` 退出。

### 交互模式 — 会话指令

| 指令 | 说明 | 示例 |
|------|------|------|
| `/help` | 显示帮助信息 | `/help` |
| `/provider <name>` | 切换后端: cloud / local | `/provider local` |
| `/provider` | 查看当前后端和可用选项 | `/provider` |
| `/device <cpu|gpu>` | 切换本地推理设备 (仅 local 后端) | `/device gpu` |
| `/device` | 查看当前设备和可用选项 | `/device` |
| `/role <name>` | 切换角色预设 | `/role teacher` |
| `/system <prompt>` | 设置自定义 system prompt | `/system 你是一位诗人` |
| `/stream` | 切换流式/非流式输出 | `/stream` |
| `/clear` | 清空对话上下文 (保留角色) | `/clear` |
| `/save [path]` | 保存对话记录到文件 | `/save` |
| `/stats` | 查看会话统计 | `/stats` |

## 参数说明

| 参数 | 简写 | 说明 | 默认值 |
|------|------|------|--------|
| `question` | — | 要提问的问题 (不提供则进入交互模式) | — |
| `--provider` | `-p` | 模型后端: `local` (Ollama) / `cloud` (DeepSeek) | `local` |
| `--device` | `-d` | 本地推理设备: `gpu` (GPU加速) / `cpu` (纯CPU, 仅 local 后端) | `gpu` |
| `--role` | `-r` | 角色: default / teacher / coder / doctor / translator | `default` |
| `--system` | `-s` | 自定义 system prompt (覆盖 `--role`) | — |
| `--temperature` | `-t` | 生成温度 0~2 | `0.7` |
| `--max-tokens` | `-m` | 最大输出 token 数 | `1024` |
| `--stream` | — | 启用流式逐字输出 | `False` |
| `--save` | — | 保存回答到文件 (仅单次模式) | — |
| `--no-echo` | — | 不显示统计信息行 | `False` |

## 架构设计

### 后端分发架构

```
                    build_payload(messages, provider, ...)
                    call_api(payload, provider)
                    call_api_stream(payload, provider)
                            │
                   provider=?
                     ┌──────┴──────┐
                     ▼              ▼
              ┌──────────────┐ ┌──────────────┐
              │  CLOUD PATH   │ │  LOCAL PATH   │
              │  DeepSeek API │ │  Ollama API   │
              │               │ │               │
              │ /v1/chat/     │ │ /api/chat     │
              │  completions  │ │               │
              │ (SSE stream)  │ │ (JSON lines)  │
              └──────────────┘ └──────────────┘
                     │              │
                     └──────┬──────┘
                            ▼
              Unified result dict:
              {"content", "model", "finish_reason",
               "usage": {prompt_tokens, completion_tokens, total_tokens},
               "elapsed"}
```

### 运行模式分流

```
main() → parse_args() → run(provider, ...)
                              │
                question 非空?  │
                   ┌───────────┴───────────┐
                   ▼                       ▼
             single_shot()        interactive_session()
             (一问一答)             (循环对话 + 上下文记忆)
                   │                       │
                   └───────────┬───────────┘
                               ▼
                safe_invoke_nonstream_only(payload, provider)
                safe_invoke_stream_only(payload, provider)
```

### 双后端对比

| 维度 | cloud (DeepSeek) | local (Ollama) |
|------|-----------------|----------------|
| **模型** | `deepseek-chat` | `llama3.2:1b` |
| **端点** | `https://api.deepseek.com/v1/chat/completions` | `http://localhost:11434/api/chat` |
| **鉴权** | `Bearer <API_KEY>` | 无需 |
| **推理设备** | 云端 GPU (自动) | CPU (`--device cpu`) / GPU (`--device gpu`) |
| **流式协议** | SSE (`data: <json>`) | 逐行 JSON |
| **Token 字段** | `usage.prompt_tokens` / `completion_tokens` | `prompt_eval_count` / `eval_count` |
| **费用** | 按量付费 | 免费 (本地算力) |
| **中文能力** | 优秀 | 中等 (1B 小模型) |

### 关键设计决策

| 决策 | 说明 | LangChain 可重用的部分 |
|------|------|----------------------|
| **统一返回格式** | 两后端归一化为同一 dict | 直接映射到 `BaseMessage` / `Generation` |
| **消息格式** | 标准 `[{role, content}]` (两后端均兼容) | 直接映射到 `ChatPromptTemplate` |
| **分发层** | `build_payload` / `call_api` 按 provider 路由 | 对应 `ChatDeepSeek` vs `ChatOllama` 切换 |
| **设备切换** | `--device cpu/gpu` + `/device` 指令 | `ChatOllama(num_gpu=0)` vs `ChatOllama(num_gpu=-1)` |
| **多轮上下文** | `messages` 跨轮次累积 | 对应 `RunnableWithMessageHistory` |
| **双模式分流** | `run()` 按 CLI 参数分流 | 单次=`invoke()`, 交互=`chain + history` |

### 未来 LangChain 迁移对照

| 当前实现 | LangChain 等价 |
|----------|---------------|
| `build_payload(provider=...)` | `ChatDeepSeek(...)` / `ChatOllama(...)` |
| `call_api(provider=...)` | `model.invoke(messages)` |
| `call_api_stream(provider=...)` | `model.stream(messages)` |
| `safe_invoke_*(provider=...)` | `chain.with_fallbacks([...])` |
| `handle_session_command(/provider)` | 自定义 Tool 切换模型实例 |
| `interactive_session()` | `RunnableWithMessageHistory` |

LCEL 伪代码 (双后端 + 双设备):
```python
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama

def get_model(provider: str, device: str = "gpu"):
    if provider == "local":
        num_gpu = -1 if device == "gpu" else 0
        return ChatOllama(model="llama3.2:1b", temperature=0.7, num_gpu=num_gpu)
    else:
        return ChatDeepSeek(model="deepseek-chat", temperature=0.7)

chain = prompt | get_model(provider, device) | StrOutputParser()
result = chain.invoke({"question": "...", "system_prompt": "..."})
```

## 项目结构

```
week03/
├── qa_assistant.py     # 主程序 — 双后端 + 双模式 + 三阶段流水线
├── requirements.txt    # Python 依赖 + Ollama 安装说明
└── README.md           # 本文档
```

依赖项目根目录的 `config.py` (仅 cloud 后端需要 `API_KEY`)。
