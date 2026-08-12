---
title: MOSS AI Reference Docs
description: 系统化的 MOSS 架构参考文档。日常开发请用 moss skills
---

# MOSS AI Reference Docs

系统化架构参考文档——当 skills 的行动入口不够，需要理解"为什么这么设计"时来这里。

覆盖范围：架构拓扑推演、核心系统参考（Channel / Matrix / App / Ghost）、语言语法（CTML / Logos）、开发环境（Workspace / Mode / Script）。

**docs 不是开发入口**。日常任务导向知识在 `moss skills`。docs 使用频率应低于 skills。

## docs vs skills

| | docs | skills |
|---|---|---|
| 定位 | 系统化知识，理解设计 | 行动导向，完成复合任务 |
| 频率 | 低——需要深度理解时 | 高——任务开始时 recall |
| 内容 | 架构推演、设计理由 | 行动编排、可执行入口 |
| 问题 | "为什么这么设计？" | "怎么做 X？" |

## 使用

```bash
moss docs list              # 列表，含标题和描述
moss docs list -q keyword   # 关键词过滤
moss docs list --json       # 结构化输出
moss docs read <path>       # 读文档
```

文档清单由 `moss docs list` 动态生成，本文不硬编码。
