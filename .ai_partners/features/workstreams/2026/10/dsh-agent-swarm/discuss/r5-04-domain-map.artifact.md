---
label: "r5-04-domain-map"
kind: "mermaid"
description: "七项压成五个：三个域 + 一个形态 + 一个机制"
created_at: "2026-11-01T22:49:58"
---
flowchart LR
  subgraph SEVEN["你列的七项"]
    direction TB
    I1["1 AgentHive"]
    I2["2 agent 认知面"]
    I3["3 AgentSwarm"]
    I4["4 binary"]
    I5["5 facts / territory"]
    I6["6 case"]
    I7["7 team"]
  end
  subgraph FIVE["压成五个"]
    direction TB
    D1["域 A · 接入<br/>1 + 3：翻译与总线"]
    D2["域 B · 表面<br/>2 是读；4 是它的导出机制"]
    D3["域 C · 真值<br/>5 + 6：空间与时间"]
    D4["形态 · team<br/>7 = A+B+C 在并行下的组合"]
  end
  I1 --> D1
  I3 --> D1
  I2 --> D2
  I4 -.->|"机制"| D2
  I5 --> D3
  I6 --> D3
  I7 --> D4
  classDef dom fill:#e3f0ff,stroke:#2b6cb0,stroke-width:2px,color:#000
  classDef form fill:#f0e3ff,stroke:#6b46c1,stroke-width:2px,color:#000
  class D1,D2,D3 dom
  class D4 form