---
title: MOSS How-To 知识库
description: MOSS 项目的操作指南和知识积累, 由 AI 和人类协作编写, 通过 moss how-tos 命令访问
---

# MOSS How-To 知识库

这里存放 MOSS 项目的操作指南。每个文档是一个 how-to，解释如何完成一个具体任务。

## 如何使用

```bash
# 列出所有 how-to — AI 处理 how-to 相关任务时，始终先跑这条了解全局
moss how-tos list

# 查看一个
moss how-tos read how-to-make-how-to.md

# 语义召回 (需要配置 ANTHROPIC_SMALL_FAST_MODEL)
moss how-tos recall "怎么创建一个新的 how-to 文档?"
```

## 如何贡献

写一个新的 .md 文件放在这个目录下。建议包含 YAML frontmatter (title + description)。README.md 是目录自身的说明。
