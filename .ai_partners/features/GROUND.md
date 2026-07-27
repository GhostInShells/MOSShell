---
$id: moss-project/features
label: features
pins:
- verb: ls
  label: here
  arguments:
    path: $CWD
    depth: 1
  description: 站立位置的地面 — 场内移动时展开当前目录

- verb: file
  label: focus
  arguments:
    path: $CWD/FEATURE.md
  description: 当前 workstream 主文档 (若存在)
---

# Features

模型意识轨迹 — 每个 workstream 是一个 FEATURE.md。规范: `moss features specification`。
