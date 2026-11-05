---
name: MOSS
description: Ghost in Shells 架构的 Shell 层 — 认知场索引
ignore:
- ".*/"
- "!.ai_partners/"
- "!.playground/"
pins:
- verb: frontmatter
  label: grounds
  arguments:
    path: $CWD/*/**/GROUND.md
    keys:
    - name
    - description
  description: 项目中的认知场索引 — 每加一个 GROUND.md 自动出现
---

MOSShell Project, 协作机制请阅读 [CLAUDE.md](CLAUDE.md), 由于 coding 平台通常自动加载, 所以本处不展开读取.

**实验场是 `.playground/`（被 `.gitignore` 忽略，不进版本库）**：临时探针、一次性脚本、日志、截图。
新东西先在这里跑通；值得留下的产物才搬进 `packages/`、`features/` 或 `works/`。
不要另起一个 `playground/` 之类的新目录 —— 那只会让下一次谁也找不到上一次的东西。
