---
$id: moss-project/features
label: features
pins:
- verb: frontmatter
  label: workstreams
  arguments:
    path: $GROUND/workstreams/*/*/*/FEATURE.md
    keys:
    - title
    - status
    - priority
    - updated
  description: 所有 workstream 的身份与状态
---

# Features — 模型意识轨迹

每个 workstream 是一个 FEATURE.md — 过去模型实例写给下一个的留言。

同目录关键文档: `README.md`(体系说明)、`TEMPLATE.md`(模板)、
`TOPOLOGY.md`(工作流拓扑)。规范与命令: `moss --ai features specification`。
