---
title: How to make a how-to
description: 如何在 MOSS 项目中创建和维护一篇 how-to 知识文档。覆盖格式、结构、验证闭环。面向 AI 协作者和人类开发者。
---

# How to Make a How-To

## 背景

how-tos 是 MOSS 项目的知识库，存放在 `src/ghoshell_moss/cli/how_tos/` 目录下。
它服务于四件事：

1. **AI 读取** — 通过 `moss how-tos read <path>` 命令行按需获取知识，避免把全部知识塞进 context
2. **AI 更新** — AI 协作者在开发过程中发现知识空白或过时，可以直接修改这些文档
3. **AI 验证** — 每篇文档末尾有"文档目标"，AI 在别的上下文里可以据此判断文档是否有效
4. **人类阅读** — 人类开发者同样可以用命令行或直接打开文件来查阅

how-tos 遵循 **code as prompt** 思想：文档不重复描述代码，而是引导读者用 `moss codex` 等工具直接反射源码。

在你写一篇新的 how-to 之前，先确认：
- 用 `moss how-tos list` 看看是否已有相关文档
- 用 `moss how-tos recall "<你的问题>"` 让 agent 做语义召回

## 格式

### 文件名

用 `-` 分隔的小写英文，描述你要教的事。例如：

```
how-to-create-a-channel.md
how-to-setup-zenoh-bridge.md
```

目录用来组织领域。如果是一个新领域，创建子目录，并在子目录里放 `README.md` 作为领域概述。

### YAML Frontmatter

文件开头必须有 YAML frontmatter，包含两个字段：

```yaml
---
title: 清晰的中文或英文标题
description: 一句话描述文档内容。这是 AI agent 做语义召回的关键字段，写清楚能大幅提高匹配率
---
```

`description` 的质量直接影响 `moss how-tos recall` 的召回准确度。
把"读者在什么场景下需要这篇文档"写进去。

### 正文

Markdown 格式。遵循下面的结构约定。

## 结构

每篇 how-to 建议包含以下章节。不是强制的，但越贴近这个结构，文档越好维护。

### 1. 背景 / 概述

说明这篇文档解决什么问题，适用的场景。

**约束**：尽量用 code-as-prompt 方式提供背景。不要大段复制源码，而是指引读者用 moss 工具自行查看：

```bash
# 查看相关模块的接口
moss codex get-interface ghoshell_moss.core.blueprint.channel_builder

# 查看相关概念
moss concepts blueprint channel_builder
```

这样当源码变化时，文档不会过时，读者看到的始终是最新的接口。

### 2. 步骤

**这是最关键的章节**。按顺序列出完成任务的每一步操作。打磨要点：

- 每一步是可独立验证的动作，不是模糊的描述
- 命令用代码块包裹，读者可以直接复制执行
- 每一步之后如果有预期结果，简要说明
- 遇到分歧点时，给出判断条件，而不是假设读者走某条路径

### 3. 示例

**约束**：优先引用项目中已有的实现或单元测试，而不是在文档里写完整示例：

```bash
# 查看已有的 channel 实现作为参考
moss codex get-source ghoshell_moss.channels.speech_channel

# 查看相关测试
moss codex get-source tests.ghoshell_moss.core.channels.test_py_channel
```

如果现有代码没有合适的示例，可以在文档里写几行关键代码片段。
但如果示例需要详细展开（超过 20 行），应该把完整示例写到目标模块的测试文件或 `__init__.py` 的 docstring 里，然后文档引用它。

### 4. 常见问题

汇集已知的坑和解决方案。每个问题一个 `###` 标题，格式：

```markdown
### 问题：xxx 报错了

原因：...

解决：...
```

## 文档目标

这是文档的"验收标准"。写完文档后，在末尾声明一个明确的目标。
**当 AI 在别的上下文里阅读这篇文档后无法完成以下目标时，应该回来优化这篇文档。**（仅限内核开发者角色）

格式：

```markdown
## 文档目标

读者按照本文档操作，应该能够：
1. 在 <路径> 下创建一个符合格式的新 how-to 文档
2. 通过 moss how-tos list 看到新文档
3. 通过 moss how-tos read <新文档> 查看内容
4. 通过 moss how-tos recall "<相关查询>" 能召回新文档
```

目标必须是可验证的、具体的操作结果，不能是"理解了 xxx"这样的模糊表述。

---

## 检查清单

写完文档后，逐项确认：

- [ ] 文件名用 `-` 分隔的小写英文，有 `.md` 后缀
- [ ] YAML frontmatter 包含 `title` 和 `description`，description 写明了适用场景
- [ ] 背景部分引用了相关 `moss codex` 命令而非复制源码
- [ ] 步骤每一条是可执行的操作
- [ ] 示例优先引用了项目已有代码或测试
- [ ] 文档目标具体、可验证
- [ ] 在合适的位置引用了其他 how-to 文档（用相对路径）

## 文档目标

读者按照本文档操作，应该能够：
1. 在 `src/ghoshell_moss/cli/how_tos/` 下创建一个符合格式的新 how-to 文档
2. 新文档通过 `moss how-tos list` 可见
3. AI 通过 `moss how-tos recall "<新文档主题>"` 能召回该文档
4. 新文档结构包含背景、步骤、示例、常见问题和文档目标五个章节
