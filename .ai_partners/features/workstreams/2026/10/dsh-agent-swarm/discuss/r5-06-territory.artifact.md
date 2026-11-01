---
label: "r5-06-territory"
kind: "mermaid"
description: "territory = 一个地址 + 三个时态：过去与未来是源，现在是投影；三个角色是读法不是权限"
created_at: "2026-11-01T23:14:42"
---
flowchart LR
  P["过去 · 源<br/>append-only<br/>git log / memento 线"]
  N["现在 · 投影<br/>ground = law + inventory<br/>渲染出来的，不写入"]
  F["未来 · 源<br/>意图<br/>features：workstream / surface / stage"]
  P -->|"推导"| N
  F -->|"约束"| N
  subgraph ROLES["三个读法 · 不是权限"]
    U["使用者<br/>读：现在"]
    A["架构师<br/>读：过去 + 未来 · 写：未来"]
    W["worker<br/>读：过去 + 未来 · 写：现在（施工）"]
  end
  U -.-> N
  A -.-> F
  W -.-> P
  W -.-> N
  classDef src fill:#e3f0ff,stroke:#2b6cb0,stroke-width:2px,color:#000
  classDef proj fill:#fff3cd,stroke:#b8860b,stroke-width:2px,color:#000
  class P,F src
  class N proj