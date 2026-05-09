---
title: How to make a how-to
description: 如何在 MOSS 项目中创建一篇新的 how-to 文档, 让 AI 和人类都能通过知识库找到它
---

# How to Make a How-To

这篇文档解释如何在 MOSS 项目中创建一篇新的 how-to 文档。

## 步骤

1. 在 `src/ghoshell_moss/cli/how_to/` 目录下创建一个 `.md` 文件
2. 文件名应该描述你要教的事情，用 `-` 分隔，例如 `how-to-create-a-channel.md`
3. 文件开头加 YAML frontmatter，至少包含 `title` 和 `description`：

```yaml
---
title: 清晰的标题
description: 一句话描述这个文档的内容, 这是给 AI 做召回用的关键信息
---
```

4. 用 Markdown 写正文。结构自由，但建议包含：背景、步骤、示例、常见问题
5. 如果是一个新主题领域，考虑创建子目录，并在子目录里放 README.md 作为领域概述

## 为什么

- `description` 字段是 AI agent 做召回时的关键依据，写清楚能大大提高被匹配到的概率
- 文件名就是 locator，简洁有描述性最好
- 这个知识库是反身性的——how-to 文档教你怎么写 how-to 文档

## 示例

参考这篇文档自身的格式：`how-to-make-how-to.md`
