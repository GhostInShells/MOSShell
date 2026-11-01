---
label: "r5-09-confluence"
kind: "mermaid"
description: "confluence 的形状：父线多一格普通 commit，带一条指向子线 tip 的引用；没有内容要合，只有事实要上报"
created_at: "2026-11-02T00:01:49"
---
flowchart TB
  subgraph PP["父 case 的线"]
    P0["…"] --> P1["confluence commit<br/>note + 引用 child@tip"]
  end
  subgraph CA["子 case A 的线"]
    A0["checkpoint"] --> A1["完结"]
  end
  subgraph CB["子 case B 的线"]
    B0["checkpoint"] --> B1["完结"]
  end
  A1 -->|"上报"| P1
  B1 -->|"上报"| P1
  P1 -.->|"父可沿引用深读子线"| A1
  P1 -.->|"同"| B1
  N["不是 merge：没有内容要合，只有事实要上报"]
  P1 --- N
  classDef note fill:#fff3cd,stroke:#b8860b,stroke-width:2px,color:#000
  class N note