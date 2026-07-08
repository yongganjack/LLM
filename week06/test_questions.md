# Week06 Simple Agent 测试手册

> 目的：快速验证 `week06/simple_agent.py` 的工具调用、非工具调用、边界条件和失败降级是否符合任务包要求。
> 推荐在项目根目录 `D:\Desktop\python_DL_project\LlmResearchProject` 下运行。

---

## 0. 推荐运行环境

本项目当前推荐使用 Anaconda 虚拟环境 `research`：

```powershell
cd D:\Desktop\python_DL_project\LlmResearchProject
& "E:\Anaconda\envs\research\python.exe" week06\simple_agent.py "请根据本地资料说明 Agent 的工作流程"
```

如果你已经激活了环境，也可以简写：

```powershell
conda activate research
python week06\simple_agent.py "请根据本地资料说明 Agent 的工作流程"
```

基础自动测试：

```powershell
& "E:\Anaconda\envs\research\python.exe" week06\test_simple_agent.py
```

预期：全部通过，当前应为 `27/27`。

---

## 1. 每次手工测试先看这三项

每次运行后先检查输出是否包含固定三段：

```text
结论：
...

依据：
...

是否调用工具：
...
```

再检查最后一段的值：

| 场景 | 预期 `是否调用工具` |
|---|---|
| 命中本地资料关键词且读取成功 | `是` |
| 普通通用问题 | `否` |
| 命中工具但本地资料读取失败 | `是，但失败` |
| 空输入 | `否` |

---

## 2. 冒烟测试：最少跑这 3 条

### 2.1 工具路径

```powershell
& "E:\Anaconda\envs\research\python.exe" week06\simple_agent.py "请根据本地资料说明 Agent 的工作流程"
```

预期检查：

- `是否调用工具：` 为 `是`
- `依据` 提到 `week06/local_knowledge.md` 或本地资料
- `结论` 包含 Agent 流程语义，例如“接收用户输入、判断、读取、组织回答”

### 2.2 非工具路径

```powershell
& "E:\Anaconda\envs\research\python.exe" week06\simple_agent.py "什么是 Prompt Template？"
```

预期检查：

- `是否调用工具：` 为 `否`
- `依据` 说明未调用本地工具
- 正常回答 Prompt Template 的含义

### 2.3 空输入路径

```powershell
& "E:\Anaconda\envs\research\python.exe" week06\simple_agent.py ""
```

预期检查：

- 提示“请输入有效问题或任务说明”
- `是否调用工具：` 为 `否`

---

## 3. 工具调用路径测试题

触发关键词：

```text
资料、知识库、文档、根据本地、本地资料、week06、Agent、工具、流程、依据
```

| # | 测试问题 | 预期重点 |
|---|---|---|
| 1 | `请根据本地资料说明 Agent 的工作流程` | 调用工具；回答四步流程 |
| 2 | `Agent 的工具调用条件是什么` | 调用工具；说明关键词规则 |
| 3 | `week06 项目中的 Agent 有哪些功能边界` | 调用工具；说明能做/不做 |
| 4 | `本地资料里对输出格式有什么要求` | 调用工具；说明三段式输出 |
| 5 | `请依据知识库说明失败处理方式` | 调用工具；说明工具失败降级 |
| 6 | `这个文档里写了什么内容` | 调用工具；概括本地知识文档 |
| 7 | `Agent 的四步流程分别是什么` | 调用工具；回答接收、判断、读取、组织 |
| 8 | `read_local_knowledge 工具的作用是什么` | 调用工具；说明只读本地文件 |

可复制批量运行：

```powershell
$py = "E:\Anaconda\envs\research\python.exe"
$questions = @(
  "请根据本地资料说明 Agent 的工作流程",
  "Agent 的工具调用条件是什么",
  "week06 项目中的 Agent 有哪些功能边界",
  "本地资料里对输出格式有什么要求",
  "请依据知识库说明失败处理方式",
  "这个文档里写了什么内容",
  "Agent 的四步流程分别是什么",
  "read_local_knowledge 工具的作用是什么"
)
foreach ($q in $questions) {
  "`n===== $q ====="
  & $py week06\simple_agent.py $q
}
```

---

## 4. 非工具调用路径测试题

这些问题不应读取 `local_knowledge.md`。

| # | 测试问题 | 预期重点 |
|---|---|---|
| 1 | `什么是 Prompt Template？` | 不调用工具；回答概念 |
| 2 | `你好，请介绍一下你自己` | 不调用工具 |
| 3 | `Python 中如何实现单例模式` | 不调用工具；回答编程知识 |
| 4 | `帮我写一个快速排序算法` | 不调用工具；回答代码/算法 |
| 5 | `1 + 1 等于几` | 不调用工具 |
| 6 | `请用中文翻译 "Hello World"` | 不调用工具 |

可复制批量运行：

```powershell
$py = "E:\Anaconda\envs\research\python.exe"
$questions = @(
  "什么是 Prompt Template？",
  "你好，请介绍一下你自己",
  "Python 中如何实现单例模式",
  "帮我写一个快速排序算法",
  "1 + 1 等于几",
  "请用中文翻译 ""Hello World"""
)
foreach ($q in $questions) {
  "`n===== $q ====="
  & $py week06\simple_agent.py $q
}
```

---

## 5. 边界条件测试

| # | 命令 | 预期 |
|---|---|---|
| 1 | `& $py week06\simple_agent.py ""` | 提示输入有效问题；状态 `否` |
| 2 | `& $py week06\simple_agent.py "   "` | 提示输入有效问题；状态 `否` |
| 3 | `& $py week06\simple_agent.py "Agent"` | 命中关键词；状态 `是` |
| 4 | `& $py week06\simple_agent.py "agent"` | 小写不命中；状态 `否` |
| 5 | `& $py week06\simple_agent.py "week06"` | 命中关键词；状态 `是` |
| 6 | `& $py week06\simple_agent.py "hello world"` | 不命中关键词；状态 `否` |

先设置变量再复制命令：

```powershell
$py = "E:\Anaconda\envs\research\python.exe"
& $py week06\simple_agent.py ""
& $py week06\simple_agent.py "   "
& $py week06\simple_agent.py "Agent"
& $py week06\simple_agent.py "agent"
& $py week06\simple_agent.py "week06"
& $py week06\simple_agent.py "hello world"
```

---

## 6. 失败降级测试

### 6.1 模型不可用但本地资料可读

可以用非法 provider 触发模型初始化失败：

```powershell
& "E:\Anaconda\envs\research\python.exe" week06\simple_agent.py --provider invalid "请根据本地资料说明 Agent 的工作流程"
```

预期检查：

- 仍然输出三段式
- `是否调用工具：` 为 `是`
- 不应只提示“安装 LangChain”或“启动模型服务”
- 结论仍应包含本地资料中的 Agent 流程语义

### 6.2 本地资料文件缺失

手工测试时可以临时重命名文件，测试后立刻改回：

```powershell
Rename-Item week06\local_knowledge.md local_knowledge.md.bak
& "E:\Anaconda\envs\research\python.exe" week06\simple_agent.py "请根据本地资料说明 Agent 的工作流程"
Rename-Item week06\local_knowledge.md.bak local_knowledge.md
```

预期检查：

- `结论` 包含“未获取到资料”
- `依据` 包含本地工具读取失败信息
- `是否调用工具：` 为 `是，但失败`

---

## 7. 交互模式测试

启动：

```powershell
& "E:\Anaconda\envs\research\python.exe" week06\simple_agent.py
```

建议输入顺序：

```text
/stats
请根据本地资料说明 Agent 的工作流程
什么是 Prompt Template？
/provider local
/stats
q
```

预期：

- `/stats` 能显示当前后端、设备、温度和 max tokens
- 工具问题状态为 `是`
- 普通问题状态为 `否`
- `q` 能正常退出

---

## 8. 测试记录表

| 日期 | 环境 | 命令/问题 | 预期状态 | 实际状态 | 是否通过 | 备注 |
|---|---|---|---|---|---|---|
|  | research | 请根据本地资料说明 Agent 的工作流程 | 是 |  |  |  |
|  | research | 什么是 Prompt Template？ | 否 |  |  |  |
|  | research | 空输入 | 否 |  |  |  |
|  | research | 模型不可用 fallback | 是 |  |  |  |
|  | research | 本地资料缺失 fallback | 是，但失败 |  |  |  |

---

## 9. 常见问题排查

### 输出乱码

优先使用 `research` 环境和 PowerShell。必要时先执行：

```powershell
$OutputEncoding = [System.Text.Encoding]::UTF8
chcp 65001
```

### 提示找不到 LangChain

确认使用的是指定解释器：

```powershell
& "E:\Anaconda\envs\research\python.exe" -c "import sys, langchain; print(sys.executable); print(langchain.__version__)"
```

### 工具路径没有调用工具

检查问题是否包含工具关键词之一：

```text
资料、知识库、文档、根据本地、本地资料、week06、Agent、工具、流程、依据
```

### 想先跑自动测试

```powershell
& "E:\Anaconda\envs\research\python.exe" week06\test_simple_agent.py
```
