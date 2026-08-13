---
name: nodes
description: MOSS 开箱的 nodes — 能力的运行时容器
pins:
- label: index
  verb: frontmatter
  arguments:
    path: $GROUND/**/NODE.md
    keys: [name, description]
    max_depth: 3
    limit: 50
  description: 开箱 nodes 一览 — 每个 node 的身份与一句话
- label: here
  verb: ls
  arguments:
    path: $CWD
    depth: 1
  description: 站立位置的地面 — 场内移动时展开当前目录
- label: focus
  verb: file
  arguments:
    path: $CWD/NODE.md
  description: 当前 node 主文档 (若存在)
---

# Nodes

MOSS 开箱的 nodes — 能力的运行时容器。每个 node 子目录是一个可运行的
能力单元，`NODE.md` 声明它的身份（name / category / exec 入口）。

站在 nodes 根时 `index` 给出所有 node 的一览；walk 进某个 node 目录时
`focus` 自动展开它的 NODE.md。
