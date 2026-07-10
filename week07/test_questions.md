# Week07 中文资料问答 Agent — 手工演示手册

> 目的：快速验证 Week07 Agent 的直接回答、资料检索和工具失败降级三个核心演示场景。
> 推荐在项目根目录 `D:\Desktop\python_DL_project\LlmResearchProject` 下运行。
> 课程资料问答默认调用模型服务；执行第 2.2 节前，请配置所选 provider 的 API Key 或启动本地模型服务。

---

## 0. 环境说明

推荐使用 Anaconda 虚拟环境 `research`：

```powershell
cd D:\Desktop\python_DL_project\LlmResearchProject
$py = "E:\Anaconda\envs\research\python.exe"
```

如果你已经激活了环境，也可以简写：

```powershell
conda activate research
python week07\simple_agent.py --no-stream "你能帮助我做什么？"
```

基础自动测试（应先通过再手工演示）：

```powershell
& "E:\Anaconda\envs\research\python.exe" week07\test_simple_agent.py
```

预期：全部通过，进程退出码为 `0`。

---

## 1. 每次检查的核心项目

每次运行后检查以下四项：

1. **四段格式**：输出包含 `结论：`、`依据：`、`是否调用工具：`、`运行记录：` 四个段落标题。
2. **工具状态**：与下表预期一致。
3. **依据可验证**：成功时包含真实章节引用，失败时不包含假引用。
4. **日志有记录**：`week07/runs/agent_runs.jsonl` 中有对应记录。

| 场景 | 预期 `是否调用工具` |
|---|---|
| 问候、帮助、能力说明、范围外、空输入 | `否` |
| 已成功读取本地资料（包括资料无命中） | `是` |
| 尝试读取资料但失败或模拟失败 | `是，但失败` |

---

## 2. 必演示的三条命令

### 2.1 直接回答

```powershell
& "E:\Anaconda\envs\research\python.exe" week07\simple_agent.py --no-stream "你能帮助我做什么？"
```

**预期检查：**

- [ ] 输出包含四段：`结论：`、`依据：`、`是否调用工具：`、`运行记录：`
- [ ] `是否调用工具：` 下面为 `否`
- [ ] `依据：` 说明"未调用本地资料工具"
- [ ] `结论：` 描述了 Agent 的能力范围（提示词模板、上下文窗口、RAG、Agent 工具调用）
- [ ] 不包含 `local_knowledge.md#提示词模板` 等具体章节引用

**示例输出片段：**

```text
结论：
你好！我是 Week07 中文资料问答 Agent，一个专注于大模型入门课程资料的本地助手。
...

依据：
未调用本地资料工具。这是一条帮助/能力说明类问题，由确定性路由规则直接回答。

是否调用工具：
否

运行记录：
run_id=run_20260710_123456_abc12345_0001，日志已写入 week07/runs/agent_runs.jsonl
```

---

### 2.2 课程资料检索与模型生成

```powershell
& "E:\Anaconda\envs\research\python.exe" week07\simple_agent.py --no-stream "根据课程资料，提示词模板由哪些部分组成？"
```

**预期检查：**

- [ ] 输出包含四段
- [ ] `是否调用工具：` 下面为 `是`
- [ ] `依据：` 包含 `local_knowledge.md#提示词模板` 引用
- [ ] `依据：` 包含"本地关键词检索"和命中词
- [ ] `结论：` 由模型根据检索片段生成，不包含引用、工具状态或运行记录
- [ ] `运行记录：` 包含 `run_id` 和日志写入位置

**示例输出片段：**

```text
结论：
【提示词模板】
提示词模板（Prompt Template）是大模型应用中最基础的组件之一。它包含三个部分：
1. 系统指令...
...

依据：
本地来源：week07/local_knowledge.md#提示词模板；
检索方式：本地关键词检索（标题命中优先 ×3，正文命中次数 ×1）；
命中词：提示词、模板。

是否调用工具：
是

运行记录：
run_id=run_20260710_123457_def67890_0002，日志已写入 week07/runs/agent_runs.jsonl
```

---

### 2.3 模拟工具失败

```powershell
& "E:\Anaconda\envs\research\python.exe" week07\simple_agent.py --no-stream --simulate-tool-failure "根据课程资料，什么是 RAG？"
```

**预期检查：**

- [ ] 输出包含四段
- [ ] `是否调用工具：` 下面为 `是，但失败`
- [ ] `结论：` 包含"未获取到资料"
- [ ] `依据：` 包含模拟错误说明和 `--simulate-tool-failure` 相关信息
- [ ] 不包含任何真实章节引用（如 `#RAG 与本地检索`）
- [ ] `运行记录：` 包含 `run_id`

**示例输出片段：**

```text
结论：
未获取到资料。以下是当前可回答部分：该问题命中了本地资料检索条件，
但读取本地知识文件时发生错误...

依据：
本地工具读取失败：[模拟工具失败] 根据 --simulate-tool-failure 开关，
本次调用模拟本地知识文件读取失败。真实文件未被修改。

是否调用工具：
是，但失败

运行记录：
run_id=run_20260710_123458_ghi90123_0003，日志已写入 week07/runs/agent_runs.jsonl
```

---

## 3. 验证模拟失败后资料仍可读

模拟失败是临时性的，不会影响真实资料文件：

```powershell
# 模拟失败后再正常读取
& "E:\Anaconda\envs\research\python.exe" week07\simple_agent.py --no-stream "什么是RAG？"
```

预期：第二条命令应正常返回资料内容（`是否调用工具：` 为 `是`），证明真实文件未被修改。

---

## 4. 日志查看

每次调用后在 `week07/runs/agent_runs.jsonl` 中追加一行 JSON 记录。查看最新记录：

```powershell
Get-Content week07\runs\agent_runs.jsonl -Tail 3 | ForEach-Object { $_ | ConvertFrom-Json | ConvertTo-Json }
```

或直接用 Python 验证格式：

```powershell
& "E:\Anaconda\envs\research\python.exe" -c "
import json
with open('week07/runs/agent_runs.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        record = json.loads(line.strip())
        print(f\"{record['时间']} | {record['路由']} | {record['工具状态']} | {record['run_id']}\")
"
```

每条记录应包含以下字段：时间、run_id、用户输入、路由、是否调用工具、工具状态、来源、章节、命中词、错误信息、回答摘要、日志写入状态。

---

## 5. 补充测试

### 5.1 范围外问题

```powershell
& "E:\Anaconda\envs\research\python.exe" week07\simple_agent.py --no-stream "帮我查询今天上海的天气"
```

预期：`是否调用工具：` 为 `否`，结论说明不能联网，不读取资料。

### 5.2 英文大小写一致性

```powershell
# 大写
& "E:\Anaconda\envs\research\python.exe" week07\simple_agent.py --no-stream "什么是RAG？"
# 小写
& "E:\Anaconda\envs\research\python.exe" week07\simple_agent.py --no-stream "什么是rag？"
```

预期：两条命令均应路由到本地检索并命中 `RAG 与本地检索` 章节。

### 5.3 空输入

```powershell
& "E:\Anaconda\envs\research\python.exe" week07\simple_agent.py --no-stream ""
```

预期：`结论：` 提示输入有效问题，`是否调用工具：` 为 `否`。

---

## 6. 模型服务失败验证

课程问答依赖模型服务。模型不可用时，程序仍应返回四段式结果，但结论必须说明“模型服务不可用，未生成最终答案”，依据保留真实检索章节，且不展示异常堆栈或绝对路径。

使用无效 provider 可安全验证此分支：

```powershell
& "E:\Anaconda\envs\research\python.exe" week07\simple_agent.py --no-stream --provider invalid "什么是 RAG？"
```

预期：`是否调用工具：` 为 `是`，依据包含 `week07/local_knowledge.md#RAG 与本地检索`，结论说明模型服务不可用；工具读取失败场景仍使用第 2.3 节命令验证。

---

## 7. 常见问题排查

### 输出乱码

使用 `research` 环境和 PowerShell。必要时先执行：

```powershell
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001
```

### 提示找不到模块

确认使用的是指定解释器：

```powershell
& "E:\Anaconda\envs\research\python.exe" -c "import sys; print(sys.executable)"
```

### 日志文件不存在

检查 `week07/runs/` 目录是否存在且有写入权限。如果程序无法创建目录，会在"运行记录"段中显示失败原因。

---

## 8. 测试记录表

| 日期 | 命令 | 预期工具状态 | 实际工具状态 | 是否通过 | 备注 |
|---|---|---|---|---|---|
|  | 你能帮助我做什么？ | 否 |  |  | 直接回答 |
|  | 根据课程资料，提示词模板由哪些部分组成？ | 是 |  |  | 本地检索 |
|  | --simulate-tool-failure "根据课程资料，什么是 RAG？" | 是，但失败 |  |  | 模拟失败 |
|  | 帮我查询今天上海的天气 | 否 |  |  | 范围外 |
|  | ""（空输入） | 否 |  |  | 空输入 |
