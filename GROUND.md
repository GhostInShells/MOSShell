---
$id: moss-project
name: MOSS
pins:
- verb: frontmatter
  label: fields
  arguments:
    path: $GROUND/**/GROUND.md
    keys:
    - $id
    - name
  description: 项目中的认知场索引 — 每加一个 GROUND.md 自动出现
---

# MOSS

Ghost in Shells 架构的 Shell 层。工程入口: `moss start`。
