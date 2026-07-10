# 🤝 贡献指南 (Contributing Guide)

感谢你对这个项目的兴趣！无论是报告 Bug、提出新功能、改进文档，还是提交代码，都是非常欢迎的。

---

## 📋 贡献方式

### 报告 Bug

如果你发现了 Bug，请通过 GitHub Issues 提交，并包含以下信息：

1. **环境信息**: Python 版本、操作系统、依赖版本 (`pip list | grep -E "langchain|requests"`)
2. **复现步骤**: 尽量提供最小可复现代码
3. **期望行为 vs 实际行为**: 清楚描述差异
4. **错误日志**: 完整的终端输出或 traceback

### 提出新功能

1. 先搜索 [Issues](https://github.com/yongganjack/LLM/issues) 确保没有重复
2. 使用 Feature Request 模板（如果有）
3. 描述你希望解决的问题和使用场景

### 提交代码 (Pull Request)

1. **Fork 本仓库**
2. **创建分支**: `git checkout -b feat/your-feature-name`
3. **编写代码**: 遵循现有代码风格
4. **运行测试**: 确保相关测试通过
5. **提交 PR**: 描述变更内容和动机

---

## 🧪 本地开发

```bash
# 1. 克隆并安装依赖
git clone https://github.com/yongganjack/LLM.git
cd LLM
pip install -r requirements.txt

# 2. 配置 API Key
# 编辑 config.py，填入你的 DeepSeek API Key

# 3. (可选) 安装本地模型
ollama pull llama3.2:1b

# 4. 运行测试
python week04/langchain_smoke_test.py
python week05/test_structured_output.py
python week06/test_simple_agent.py
python week07/test_simple_agent.py
```

---

## 📝 代码规范

- **语言**: 所有注释、文档、commit message 使用中文，代码标识符使用英文
- **命名**: 函数 `snake_case`，类 `PascalCase`，常量 `UPPER_SNAKE`
- **文档字符串**: 公共函数使用 Google 风格或 NumPy 风格 docstring
- **类型注解**: 尽量为公共函数参数和返回值添加类型注解
- **Commit Message**: 使用约定式提交格式

```
feat(week07): 新增确定性意图路由
fix(week05): 修复流式模式 token 计数为 0 的 bug
docs: 更新 README 添加使用示例
refactor(week04): 提取 _extract_usage 为独立函数
```

### Commit 类型

| 类型 | 说明 |
|------|------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 文档变更 |
| `refactor` | 代码重构（不改变行为） |
| `test` | 测试相关 |
| `chore` | 构建/工具/依赖变更 |
| `style` | 格式变更（不影响逻辑） |

---

## 📂 项目结构约定

```
weekXX/                      # 每周新内容放在独立目录
├── *.py                     # 主脚本
├── test_*.py                # 测试文件
├── *_plan.md                # 开发计划
├── *_spec.md                # 功能规格
├── local_knowledge.md       # 本地知识库
└── 本周文件说明文档.md       # 每周文件说明
```

- **不要**在 `config.py` 中提交真实的 API Key（已在 `.gitignore` 中排除）
- **优先复用**已有模块而非复制代码（如 Week07 复用 Week04 的 token 提取逻辑）
- **测试文件**应覆盖正常路径 + 失败降级路径

---

## 🔍 评审流程

1. 提交 PR 后，GitHub Actions 会自动运行基础检查
2. 维护者会进行代码评审，可能会提出修改建议
3. 评审通过后合并到 `master` 分支

---

再次感谢你的贡献！🎉
