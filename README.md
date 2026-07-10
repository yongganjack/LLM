
<p align="center">
  <h1 align="center">🤖 LLM Research Project</h1>
  <p align="center">
    <strong>大模型入门学习与实践 — 从 API 调用到智能 Agent 的渐进式项目</strong>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/LangChain-1.x-green.svg" alt="LangChain">
    <img src="https://img.shields.io/badge/Ollama-Llama3.2-orange.svg" alt="Ollama">
    <img src="https://img.shields.io/badge/DeepSeek-API-purple.svg" alt="DeepSeek">
    <img src="https://img.shields.io/badge/Progress-7%20Weeks-brightgreen.svg" alt="Progress">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
    <br>
    <a href="https://github.com/yongganjack/LLM/actions"><img src="https://github.com/yongganjack/LLM/actions/workflows/ci.yml/badge.svg" alt="CI Status"></a>
    <img src="https://img.shields.io/github/last-commit/yongganjack/LLM" alt="Last Commit">
    <img src="https://img.shields.io/github/repo-size/yongganjack/LLM" alt="Repo Size">
  </p>
</p>

---

## 📖 项目简介

一个为期 7 周的大模型（LLM）学习与实践项目，从最基础的 HTTP API 调用开始，逐步深入到 LangChain 框架、结构化输出、Agent 工具调用和本地知识检索。项目支持**本地 Ollama (Llama 3.2)** 和**云端 DeepSeek** 双后端，提供 CLI 中文问答助手、提示词对比实验、以及基于 ReAct 模式的智能 Agent。

**核心理念：先理解底层，再使用框架，最后构建智能系统。**

---

## 📑 目录

- [学习路线 (7 Weeks)](#-学习路线-7-weeks)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [核心功能](#-核心功能)
  - [双后端支持](#1-双后端支持-local--cloud)
  - [多角色预设](#2-多角色预设)
  - [交互式对话](#3-交互式对话)
  - [提示词对比实验](#4-提示词对比实验)
  - [Agent 能力](#5-agent-能力-week07)
- [关键实验结论](#-关键实验结论)
- [运行测试](#-运行测试)
- [贡献指南](#-贡献指南)
- [参考资源](#-参考资源)

---

## 🗺️ 学习路线 (7 Weeks)

| 周次 | 主题 | 核心内容 | 关键技术 |
|------|------|----------|----------|
| **Week 3** | 模型部署与调用 | 本地 Ollama + 云端 DeepSeek 双后端接入，CLI 中文问答助手 | HTTP API, SSE/JSON-lines 流式解析, GPU 加速 |
| **Week 4** | LangChain 重构 | 将 Week03 的手写 HTTP 调用迁移到 LangChain 框架 | ChatOllama, ChatDeepSeek, ChatPromptTemplate, LCEL |
| **Week 5** | 结构化输出增强 | 系统提示词自动附加 JSON 输出约定，4 级降级解析器 | JSON Output Contract, Fallback Parser, 多轮对话记忆 |
| **Week 6** | Simple Agent 原型 | 首个 LangChain Agent — 工具调用、ReAct 循环、三段式输出 | create_agent, @tool, ReAct Pattern, Fallback |
| **Week 7** | 高级资料问答 Agent | 确定性意图路由、本地关键词检索与重排、模型生成结论、JSONL 日志 | Intent Routing, Local Search & Rerank, Four-Section Output |

---

## 🚀 快速开始

### 环境要求

- **Python** 3.10+
- **Ollama** (可选，用于本地模型) — [下载地址](https://ollama.com/download)
- **DeepSeek API Key** (可选，用于云端模型) — [获取地址](https://platform.deepseek.com)

### 安装

```bash
# 1. 克隆仓库
git clone https://github.com/yongganjack/LLM.git
cd LLM

# 2. 安装依赖
pip install -r requirements.txt

# 3. (可选) 安装本地模型
ollama pull llama3.2:1b

# 4. 配置 API Key
# 编辑 config.py，替换为你的 DeepSeek API Key
# API_KEY = "sk-your-key-here"
```

### 运行

```bash
# ── Week03: 原始 API 调用 ──
python test.py                          # 本地 Ollama 调用
python cloud_api_demo.py                # 云端 DeepSeek 调用
python prompt_comparison.py             # 提示词对比实验

# ── Week04/05: LangChain 问答助手 ──
python week04/qa_assistant_lc.py "什么是机器学习？"
python week05/qa_assistant_structured.py --stream "解释 RAG"
python week04/qa_assistant_lc.py -p cloud --role teacher "什么是 Transformer？"

# ── Week06/07: Agent ──
python week07/simple_agent.py "根据课程资料，提示词模板由哪些部分组成？"
python week07/simple_agent.py --no-stream --simulate-tool-failure "什么是 RAG？"
python week07/simple_agent.py           # 交互模式
```

---

## 🏗️ 项目结构

```
LlmResearchProject/
├── README.md                           # 项目说明
├── API_REFERENCE.md                    # Ollama vs DeepSeek API 详细对照文档
├── requirements.txt                    # 项目依赖
├── config.py                           # DeepSeek API Key 配置
│
├── test.py                             # Week03: Ollama 本地调用 (流式/非流式)
├── cloud_api_demo.py                   # Week03: DeepSeek 云端调用 (流式/非流式)
├── prompt_comparison.py                # Week03: 提示词对比实验 (5个测试用例)
├── comparison_results.json             # Week03: 对比实验结果 (JSON)
├── comparison_results.md               # Week03: 对比实验结果 (Markdown 分析)
│
├── week03/                             # Week03: 原始 HTTP 调用版问答助手
│   ├── qa_assistant.py                 #   933行, 支持本地/云端/GPU/CPU/流式/交互
│   ├── langchain_migration_guide.md    #   迁移到 LangChain 的详细指南
│   └── README.md                       #   Week03 说明
│
├── week04/                             # Week04: LangChain 重构版
│   ├── qa_assistant_lc.py              #   866行, LCEL 链式调用
│   ├── langchain_smoke_test.py         #   冒烟测试
│   └── week04_demo.md                  #   学习记录与前后对比
│
├── week05/                             # Week05: 结构化输出增强
│   ├── qa_assistant_structured.py      #   JSON 输出约定 + 4 级解析器
│   ├── test_structured_output.py       #   结构化输出测试
│   └── prompt_template_output_parser_reference.md
│
├── week06/                             # Week06: Simple Agent 原型
│   ├── simple_agent.py                 #   LangChain Agent + ReAct 循环
│   ├── test_simple_agent.py            #   Agent 验收测试
│   ├── local_knowledge.md              #   本地知识库
│   └── week06_agent_development_plan.md
│
└── week07/                             # Week07: 高级资料问答 Agent
    ├── simple_agent.py                 #   确定性路由 + 本地检索 + 模型生成
    ├── test_simple_agent.py            #   问答与验收自动测试
    ├── local_knowledge.md              #   大模型入门课程资料
    ├── agent_function_spec.md          #   Agent 功能规格
    ├── test_questions.md               #   测试问题集
    ├── runs/                           #   运行日志 (JSONL)
    └── week07_agent_development_plan.md
```

---

## 🔑 核心功能

### 1. 双后端支持 (Local + Cloud)

| 特性 | Ollama (本地) | DeepSeek (云端) |
|------|--------------|----------------|
| 模型 | `llama3.2:1b` (1.2B) | `deepseek-chat` |
| 启动方式 | `ollama serve` | API Key 鉴权 |
| 流式协议 | JSON-lines | SSE (Server-Sent Events) |
| GPU 加速 | `num_gpu` 参数控制 | 云端自动 |
| 速度 | 3~9s (CPU) | 0.7~3s |
| 费用 | 免费 | 按量付费 |

一键切换：`-p local` / `-p cloud`

### 2. 多角色预设

```bash
python week04/qa_assistant_lc.py --role teacher "解释量子计算"   # 教师模式
python week04/qa_assistant_lc.py --role coder "写一个排序算法"    # 程序员模式
python week04/qa_assistant_lc.py --role doctor "什么是高血压？"   # 医学顾问
python week04/qa_assistant_lc.py --role translator "你好世界"     # 翻译模式
```

### 3. 交互式对话

支持 `/help`, `/role`, `/stream`, `/clear`, `/provider`, `/device`, `/save`, `/stats` 等会话指令，完整的多轮对话上下文记忆。

### 4. 提示词对比实验

5 组对照实验，side-by-side 对比本地与云端模型在不同提示词策略下的表现：

| 实验 | 对比维度 |
|------|----------|
| Case 1 | Zero-shot vs Few-shot (情感分类) |
| Case 2 | Standard vs Chain-of-Thought (数学推理) |
| Case 3 | No-role vs Role-based (技术解释) |
| Case 4 | Plain vs Structured Output (信息提取) |
| Case 5 | Vague vs Detailed System Prompt (创意写作) |

### 5. Agent 能力 (Week07)

```
用户输入 → 确定性意图路由 → 本地关键词检索 → 模型生成结论 → 四段式输出
                                    ↓
                              JSONL 运行日志
```

- **意图路由**: 纯规则识别问候/帮助/课程问题/范围外
- **本地检索**: 中文关键词匹配，标题命中优先 ×3，无需向量数据库
- **模型生成**: 基于检索片段生成回答，附完整来源引用
- **四段输出**: 结论 / 依据 / 是否调用工具 / 运行记录
- **JSONL 日志**: 每次运行记录时间、路由、命中词、来源等完整信息

---

## 📊 关键实验结论

来自 [prompt_comparison.py](prompt_comparison.py) 的对比实验：

1. **提示词工程对小模型（1B）边际收益有限** — Few-shot 未能纠正情感误判
2. **大模型对细微指令敏感** — DeepSeek 能区分"最多150字"、"不要包含其他内容"等精确约束
3. **结构化输出（JSON）是两个模型表现最一致的场景**
4. **创意写作最能体现模型代差** — DeepSeek 能生成工整的五言绝句，Llama 1B 完全无法遵循格式

> 详见 [comparison_results.md](comparison_results.md)

---

## 🧪 运行测试

```bash
# Week04 冒烟测试 (依赖 + 连通性 + LCEL)
python week04/langchain_smoke_test.py

# Week05 结构化输出测试
python week05/test_structured_output.py

# Week06 Agent 验收测试
python week06/test_simple_agent.py

# Week07 Agent 问答与验收测试
python week07/test_simple_agent.py
```

---

## 🤝 贡献指南

欢迎贡献！请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细的贡献流程和代码规范。

提交 Issue 或 PR 前，请先查看已有的 [Issues](https://github.com/yongganjack/LLM/issues) 和 [PRs](https://github.com/yongganjack/LLM/pulls)。

---

## 📚 参考资源

- [Ollama 官方文档](https://ollama.com/docs)
- [DeepSeek API 文档](https://platform.deepseek.com/docs)
- [LangChain 文档](https://python.langchain.com/docs/)
- [Datawhale LLM Universe](https://github.com/datawhalechina/llm-universe)
- [Datawhale LLM Cookbook](https://github.com/datawhalechina/llm-cookbook)

---

## 📄 License

MIT License — 详见 [LICENSE](LICENSE) 文件

---

## ⭐ Star History

如果这个项目对你有帮助，欢迎 Star ⭐！

<p align="center">
  <sub>Built with ❤️ as a learning journey into LLMs</sub>
</p>
