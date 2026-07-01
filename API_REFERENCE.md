# 本地模型 & 云端 API 接入文档

> 记录 Llama 3.2 (Ollama 本地) 与 DeepSeek (云端) 的服务地址、接入方式、请求/返回格式及调用差异。

---

## 1. 服务地址与基本信息

| 项目 | Ollama (本地) | DeepSeek (云端) |
|---|---|---|
| **模型名称** | `llama3.2:1b` (1.2B 参数, Q8_0 量化) | `deepseek-chat` / `deepseek-reasoner` |
| **服务地址** | `http://localhost:11434` | `https://api.deepseek.com` |
| **API 端点** | `/api/generate` | `/v1/chat/completions` |
| **协议兼容** | Ollama 自有格式 | OpenAI SDK 兼容 |
| **鉴权方式** | 无需 | `Authorization: Bearer <api_key>` |
| **GPU 支持** | 本地 GPU（`num_gpu` 参数控制） | 云端自动管理，无需关注 |
| **模型存储** | `E:\LLM-Learning\ollama\models` | 云端托管 |

### Ollama 服务管理

```bash
# 启动服务
ollama serve

# 查看已安装模型
ollama list

# 拉取模型
ollama pull llama3.2:1b

# 查看模型详情
curl http://localhost:11434/api/show -d '{"name":"llama3.2:1b"}'
```

### DeepSeek Key 获取

1. 访问 [platform.deepseek.com](https://platform.deepseek.com)
2. 注册 / 登录 → API Keys → 创建新 Key
3. 复制 Key 到代码或环境变量

---

## 2. 请求格式对比

### 2.1 Ollama — `/api/generate`

```json
{
    "model": "llama3.2:1b",
    "prompt": "用一句话介绍深度学习。",
    "system": "请用中文回答，简洁明了。",
    "stream": false,
    "options": {
        "temperature": 0.7,
        "num_predict": 256,
        "num_gpu": 0
    }
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | string | 模型名称，需与 `ollama list` 一致 |
| `prompt` | string | 用户输入文本 |
| `system` | string (可选) | 系统提示词，设定模型行为 |
| `stream` | bool | `true` = 流式输出 |
| `options.temperature` | float | 随机性控制 (0~2)，越高越随机 |
| `options.num_predict` | int | 最大生成 token 数 |
| `options.num_gpu` | int | GPU 层数：`999`=自动加载全部, `0`=纯 CPU |

### 2.2 DeepSeek — `/v1/chat/completions`

```json
{
    "model": "deepseek-chat",
    "messages": [
        { "role": "system", "content": "请用中文回答，简洁明了。" },
        { "role": "user",   "content": "用一句话介绍深度学习。" }
    ],
    "temperature": 0.7,
    "max_tokens": 256,
    "stream": false
}
```

| 字段 | 类型 | 说明 |
|---|---|---|
| `model` | string | `deepseek-chat` 或 `deepseek-reasoner` |
| `messages` | array | 对话消息列表，OpenAI 标准格式 |
| `messages[].role` | string | `system` / `user` / `assistant` |
| `messages[].content` | string | 消息文本内容 |
| `temperature` | float | 随机性控制 (0~2) |
| `max_tokens` | int | 最大生成 token 数 |
| `stream` | bool | `true` = SSE 流式输出 |
| **Headers** | | |
| `Content-Type` | string | `application/json` |
| `Authorization` | string | `Bearer <api_key>` |

---

## 3. 返回格式对比

### 3.1 Ollama — 非流式响应

```json
{
    "model": "llama3.2:1b",
    "created_at": "2026-07-01T08:40:02.9258427Z",
    "response": "深度学习是一种基于多层神经网络的机器学习方法...",
    "done": true,
    "done_reason": "stop",
    "context": [128006, 9125, 128007, 271, ...],
    "total_duration": 2266925200,
    "load_duration": 1778368800,
    "prompt_eval_count": 31,
    "prompt_eval_duration": 210198000,
    "eval_count": 10,
    "eval_duration": 267293999
}
```

关键返回字段:

| 字段 | 类型 | 说明 |
|---|---|---|
| `response` | string | 模型生成的文本（**核心字段**） |
| `done` | bool | 是否生成完毕 |
| `done_reason` | string | 结束原因: `stop` / `length` / `load` |
| `eval_count` | int | 生成的 token 数量 |
| `prompt_eval_count` | int | 输入 prompt 的 token 数量 |
| `eval_duration` | int | 生成耗时（纳秒） |
| `total_duration` | int | 总耗时（纳秒），含模型加载 |
| `load_duration` | int | 模型加载耗时（纳秒） |

### 3.2 DeepSeek — 非流式响应

```json
{
    "id": "chatcmpl-xxxxxxxx",
    "object": "chat.completion",
    "created": 1710000000,
    "model": "deepseek-chat",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "深度学习是一种基于多层神经网络的机器学习方法..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 18,
        "completion_tokens": 12,
        "total_tokens": 30
    }
}
```

关键返回字段:

| 字段 | 类型 | 说明 |
|---|---|---|
| `choices[0].message.content` | string | 模型生成的文本（**核心字段**） |
| `choices[0].message.role` | string | 固定为 `assistant` |
| `choices[0].finish_reason` | string | 结束原因: `stop` / `length` / `content_filter` |
| `usage.prompt_tokens` | int | 输入消耗的 token 数 |
| `usage.completion_tokens` | int | 输出消耗的 token 数 |
| `usage.total_tokens` | int | 总 token 消耗 |
| `model` | string | 实际使用的模型名称 |

---

## 4. 流式输出对比

### 4.1 Ollama 流式

**协议**: 逐行 JSON（每行一个独立 JSON 对象）

```
{"model":"llama3.2:1b","response":"深度","done":false}
{"model":"llama3.2:1b","response":"学习","done":false}
...
{"model":"llama3.2:1b","response":"法","done":true,"done_reason":"stop",...}
```

解析方式:
```python
for line in resp.iter_lines(decode_unicode=True):
    if not line:
        continue
    chunk = json.loads(line)
    token = chunk.get("response", "")
    if chunk.get("done"):
        break
```

### 4.2 DeepSeek 流式 (SSE)

**协议**: Server-Sent Events (`data: <json>\n\n`)

```
data: {"id":"...","choices":[{"delta":{"content":"深度"},"index":0}]}

data: {"id":"...","choices":[{"delta":{"content":"学习"},"index":0}]}

...

data: [DONE]
```

解析方式:
```python
for line in resp.iter_lines(decode_unicode=True):
    if not line.startswith("data: "):
        continue
    data_str = line[6:]           # 去掉 "data: " 前缀
    if data_str == "[DONE]":
        break
    chunk = json.loads(data_str)
    token = chunk["choices"][0]["delta"].get("content", "")
```

---

## 5. 核心调用差异对照

| 维度 | Ollama (本地) | DeepSeek (云端) |
|---|---|---|
| **消息结构** | `prompt` + `system` (两个独立字符串) | `messages` 数组 (`[{role, content}]`) |
| **多轮对话** | 需传 `context` 字段或手动拼接 prompt | 直接将历史放入 `messages` 数组 |
| **Token 计数字段** | `eval_count` (生成), `prompt_eval_count` (输入) | `usage.prompt_tokens`, `usage.completion_tokens` |
| **耗时字段** | 纳秒 (`total_duration`, nanosecond) | 需自行用 `time.perf_counter()` 计量 |
| **模型名返回** | `model` (string) | `model` (string, 可能路由到不同版本) |
| **流式协议** | 逐行 JSON，内容在 `response` | SSE, 内容在 `choices[0].delta.content` |
| **流式结束标志** | `done: true` | `data: [DONE]` |
| **GPU 控制** | `options.num_gpu`，可精细控制层数 | 无需控制，云端自动 |
| **温度参数** | `options.temperature` (嵌套) | `temperature` (顶层) |
| **最大 Token** | `options.num_predict` | `max_tokens` |

---

## 6. 错误处理对照

| 错误场景 | Ollama 返回 | DeepSeek 返回 |
|---|---|---|
| 服务未启动 | `requests.ConnectionError` | `requests.ConnectionError` |
| 模型不存在 | HTTP 404, body: `{"error":"model not found"}` | HTTP 404, body: `{"error":{"message":"..."}}` |
| 鉴权失败 | 无需鉴权 | HTTP 401, body: `{"error":{"message":"..."}}` |
| 余额不足 | 无需付费 | HTTP 402 |
| 频率限制 | 无限流 | HTTP 429 |
| 服务内部错误 | HTTP 500 | HTTP 500 |

### 错误响应体提取方式

```python
# Ollama
detail = resp.json().get("error", "")

# DeepSeek
detail = resp.json().get("error", {}).get("message", "")
```

---

## 7. 参考代码文件

| 文件 | 说明 |
|---|---|
| [test.py](test.py) | Ollama 本地调用：请求构建、流式/非流式、错误处理 |
| [cloud_api_demo.py](cloud_api_demo.py) | DeepSeek 云端调用：请求构建、流式/非流式、错误处理 |
| [prompt_comparison.py](prompt_comparison.py) | 提示词对比实验：同一 prompt 下两个模型 side-by-side 对比 |
| [comparison_results.json](comparison_results.json) | 最近一次对比实验的完整结果（含耗时、token 用量） |

---

*文档更新于 2026-07-01*
