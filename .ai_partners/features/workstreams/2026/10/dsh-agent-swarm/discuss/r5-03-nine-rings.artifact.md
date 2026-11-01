---
label: "r5-03-nine-rings"
kind: "mermaid"
description: "九连环：路由是平的、治理是树，两者不合并；加一层不改总线"
created_at: "2026-11-01T22:31:42"
---
flowchart LR
  subgraph BUS["总线 · 平，只认地址"]
    direction TB
    R1["hive/a"]
    R2["hive/a/op-1"]
    R3["hive/b/op-9"]
  end
  subgraph FACTS["facts · 树，只认治理"]
    direction TB
    F1["ghost"] --> F2["case 1"] --> F3["case 1.1"]
  end
  BUS -.->|"前缀 + 订阅"| OK["递归只在通讯面"]
  FACTS -.->|"parent 边 + 归档"| OK
  classDef ok fill:#d8f5d0,stroke:#2a7,stroke-width:2px,color:#000
  class OK ok