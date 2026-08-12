---
$id: moss-project
name: MOSS
description: Ghost in Shells 架构的 Shell 层 — 认知场索引
pins:
- verb: frontmatter
  label: fields
  arguments:
    path: $GROUND/**/GROUND.md
    keys:
    - name
    - description
  description: 项目中的认知场索引 — 每加一个 GROUND.md 自动出现
---

# MOSS

Ghost in Shells 架构的 Shell 层 (Model-oriented Operating System Shell)。

认知入口: `moss start`。核心抽象: `moss --ai codex architecture`。
子场索引见 `fields` pin — 每个 GROUND.md 自动出现。
