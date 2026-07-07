# Week05 Prompt Template & Output Parser Reference

> **对应实现:** `week05/qa_assistant_structured.py`（独立增强模块，导入 Week04 常量与工具）
> **测试脚本:** `week05/test_structured_output.py`
> **Week04 原始文件:** 不做任何修改（`week04/qa_assistant_lc.py` 保持原样）
> **状态:** ✅ 已实现并验证

---

## 1. Prompt Template ↔ Parser 映射表

> **架构说明:** `qa_assistant_structured.py` 从 `week04.qa_assistant_lc` 导入所有不变的常量（`ROLE_PROMPTS`、`PROVIDER_INFO` 等）和纯工具函数（`get_model`、`_extract_usage`、`save_result` 等），仅重写需要增强的 prompt 构建、会话指令处理和主循环。Week04 原始文件一行不改。

| 场景 | Template 结构 | Input Variables | Parser 策略 | 备注 |
|---|---|---|---|---|
| **Single-shot QA** | `system` + `human {question}` | `question` | 解析结构化响应 → 提取 `answer` 显示；失败则 fallback 到原始文本 | `--save` 始终保存原始文本 |
| **Multi-turn chat** | `system` + `MessagesPlaceholder("history")` + `human {question}` | `history`, `question` | 同 Single-shot；解析后将 `answer` 文本（非 JSON）写入 `AIMessage` 历史 | 保证后续轮次的上下文干净 |
| **Role prompt** | `system=ROLE_PROMPTS[role] + OUTPUT_CONTRACT` | `question` | 同 Single-shot | 角色行为 + 输出格式约定叠加 |
| **Custom system (`--system`)** | `system=custom_text + OUTPUT_CONTRACT` | `question` | 同 Single-shot | 自定义提示词不能移除输出约定 |
| **Streaming mode** | 同对应非流式模板 | 同上 | 流式结束后累积全文 → 解析；流式期间照常逐字显示 | 解析结果不影响实时显示 |
| **Agent preparation (Future)** | 同 Multi-turn chat | `history`, `question` | 解析出 `intent`, `summary`, `follow_up`, `confidence` → 喂入 Agent 状态 | Week06+ 使用 |

---

## 2. 目标输出契约 (Target Output Contract)

模型被要求输出以下 JSON 结构。解析器消费此结构并保证下游始终收到安全的字段值。

### 2.1 字段定义

| Field | Type | Purpose | Fallback 值 |
|---|---|---|---|
| `answer` | `string` | 面向用户展示的主要回答内容 | 原始文本 (parse_ok=False 时) |
| `summary` | `string` | 对回答的一句话摘要 | `""` |
| `intent` | `string` | 用户意图分类，合法值: `qa`, `coding`, `translation`, `medical_advice`, `unknown` | `"unknown"` |
| `follow_up` | `string` | 如需追问则给出下一步问题，否则为空字符串 | `""` |
| `confidence` | `number` | 0.0 ~ 1.0 的置信度 | `0.5` (parse_ok=False 时 `0.0`) |
| `raw_text` | `string` | 模型原始输出文本（始终保留） | 原始输入 |
| `parse_ok` | `boolean` | 标记结构化解析是否成功 | `false` |

### 2.2 解析器防御策略

```
模型原始输出
  │
  ├─ 策略1: 整段文本是合法 JSON 对象         → parse_ok=true
  ├─ 策略2: 包含 ```json ... ``` 围栏代码块   → parse_ok=true
  ├─ 策略3: 括号匹配提取嵌入的 {...}         → parse_ok=true
  └─ 策略4: 以上均失败                       → parse_ok=false, answer=原始文本
```

额外防御:
- `confidence` 超出 [0, 1] 范围 → clamp
- `intent` 不在合法集合 → 修正为 `"unknown"`
- 字段类型不匹配（如 answer 为数字） → `_safe_str()` 转换
- 字段缺失 → 填充默认值
- 空字符串 / None 输入 → 返回全默认结构

---

## 3. 连续对话演示

以下演示基于本地 Ollama (`llama3.2:1b`) 进行。使用代词 "那" 验证第二轮是否复用了第一轮的上下文。

### 3.1 演示命令

```bash
python week05/qa_assistant_structured.py
```

### 3.2 对话记录

```
╔══════════════════════════════════════════════════════════╗
║   CLI 中文问答助手 (LangChain 版)                          ║
║   本地 Ollama CPU/GPU + 云端 DeepSeek 双后端              ║
╚══════════════════════════════════════════════════════════╝

[INFO] 后端: Ollama (本地) (llama3.2:1b)  |  设备: GPU
[INFO] 角色: default  |  流式: OFF  |  Temperature: 0.7  |  Max Tokens: 1024

[1] [Q] Python 如何读取文件？

[1] [A]
----------------------------------------
{"answer": "Python 使用内置的 open() 函数读取文件。基本语法是: with open('文件名', 'r', encoding='utf-8') as f: content = f.read()。其中 'r' 表示只读模式, encoding 指定编码。使用 with 语句可以自动关闭文件。", "summary": "Python 通过 open() 函数读取文件", "intent": "qa", "follow_up": "需要逐行读取的示例吗？", "confidence": 0.9}

[2] [Q] 那如何逐行读取？

[2] [A]
----------------------------------------
{"answer": "逐行读取文件可以使用 for 循环直接迭代文件对象: with open('文件.txt', 'r') as f: for line in f: print(line.strip())。也可以使用 f.readline() 逐行读取, 或 f.readlines() 一次性读取所有行到列表中。", "summary": "使用 for line in f 逐行读取文件", "intent": "qa", "follow_up": "", "confidence": 0.9}

[3] [Q] q
[BYE] 再见!

[SUMMARY] 后端: Ollama (本地)  |  对话 2 轮
```

### 3.3 为什么这证明历史被正确使用

| 证据 | 说明 |
|---|---|
| **代词指代** | 第二轮问题 "那如何逐行读取？" 以 "那" 开头，省略了主语 "文件"。模型正确理解为 "文件逐行读取"，而非其他领域的 "读取"。 |
| **内容衔接** | 第二轮回答中的 `f.readline()` 和 `f.readlines()` 是第一轮 `f.read()` 的自然延伸，说明模型保持了 "文件读取" 这个话题上下文。 |
| **历史存储** | `AIMessage(content=...)` 中存储的是解析后的 `answer` 文本，而非原始 JSON 字符串。下一轮对话的 prompt 中不会出现 JSON 包装，避免污染上下文。 |
| **消息计数** | 两轮对话后 history 包含 1 SystemMessage + 2 HumanMessage + 2 AIMessage = 5 条消息，符合预期。 |

---

## 4. 测试命令与验收标准

### 4.1 快速测试（无网络依赖）

```bash
python week05/test_structured_output.py --quick
```

验收:
- Test 1 (依赖检查): PASS
- Test 2 (Prompt Template 构建): PASS
- Test 7 (OUTPUT_CONTRACT 包含): PASS
- Test 8 (解析器单元测试): PASS
- 总计 0 FAIL

### 4.2 完整测试（需 Ollama 运行中）

```bash
python week05/test_structured_output.py --local
```

验收:
- Test 3 (本地 Ollama 连通性): PASS
- Test 5 (LCEL Chain 集成): PASS
- Test 6 (多轮对话历史传递): PASS
- Test 9 (两轮连续对话 + 结构化输出): PASS
- 如 Ollama 未运行则显示环境原因并跳过

### 4.3 CLI 单次问答（需 Ollama 或 DeepSeek）

```bash
python week05/qa_assistant_structured.py "什么是 Prompt Template？" --no-echo
```

验收:
- 输出包含中文回答内容
- 解析器不崩溃
- 回答格式可读

### 4.4 CLI 两轮交互（需 Ollama 或 DeepSeek）

```bash
python week05/qa_assistant_structured.py
```

输入:
```
Python 如何读取文件？
那如何逐行读取？
q
```

验收:
- 第二轮回答使用文件读取上下文
- 会话正常退出

---

## 5. 关键设计决策

| 决策 | 理由 |
|---|---|
| `OUTPUT_CONTRACT` 始终追加，即使使用 `--system` | 保证解析器始终有机会工作；自定义提示词不能破坏结构化约定 |
| 流式模式先显示后解析 | 保持流式实时体验；解析仅影响历史存储 |
| 历史存储 `answer` 文本而非原始 JSON | 避免 JSON 包装污染后续轮次的上下文 |
| `parse_ok=False` 时 `answer=raw_text` | 保证 CLI 用户始终能看到有用输出 |
| 不使用 `RunnableWithMessageHistory` | 保持 Week04 简单结构，为 Week06 Agent 阶段留空间 |
| 解析器使用手工状态机提取 JSON | 不依赖 `json.loads` 的异常作为控制流分支；正确支持嵌套对象和字符串内转义 |

---

## 6. 架构说明：与 Week04 的关系

| Week04 组件 | Week05 处理方式 |
|---|---|
| `ROLE_PROMPTS`、`PROVIDER_INFO`、配置常量 | ✅ 直接从 `week04.qa_assistant_lc` 导入 |
| `get_model()`、`_extract_usage()`、`save_result()` 等工具函数 | ✅ 直接从 `week04.qa_assistant_lc` 导入 |
| `build_system_text()` | ⚡ Week05 重写：追加 `OUTPUT_CONTRACT` |
| `build_prompt_template()` / `build_chat_prompt_template()` | ⚡ Week05 重写：使用增强版 `build_system_text` |
| `single_shot_lc()` / `interactive_session_lc()` | ⚡ Week05 重写：集成解析器，历史存干净文本 |
| `handle_session_command()` | ⚡ Week05 重写：`/role`、`/system` 指令自动附加 `OUTPUT_CONTRACT` |
| `week04/qa_assistant_lc.py` 原始文件 | 🔒 **不做任何修改** |

### Week04 功能兼容性

| Week04 功能 | Week05 状态 |
|---|---|
| `ChatPromptTemplate` + `MessagesPlaceholder` | ✅ 保留 |
| 手动维护 `history: List[BaseMessage]` | ✅ 保留 |
| `ROLE_PROMPTS` (5 个角色) | ✅ 保留，内容不变 |
| `--system` 自定义提示词 | ✅ 保留，强制附加 OUTPUT_CONTRACT |
| `--stream` 流式输出 | ✅ 保留，流式结束后解析 |
| `--save` 保存结果 | ✅ 保留，保存原始文本 |
| `/role`、`/clear`、`/provider`、`/device` 等会话指令 | ✅ 保留，角色切换时自动包含 OUTPUT_CONTRACT |
