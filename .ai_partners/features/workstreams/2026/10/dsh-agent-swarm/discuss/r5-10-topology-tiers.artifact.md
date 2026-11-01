---
label: "r5-10-topology-tiers"
kind: "mermaid"
description: "gateway 之后的拓扑：三档入口，判据是「要不要本地类型对象」，不是活得久不久"
created_at: "2026-11-02T00:44:51"
---
flowchart TB
  Q["一个进程要怎么跟总线说话？"]
  Q -->|"要有本地类型对象<br/>0.48–0.85s"| P["project / matrix 级<br/>bootstrap 出契约与 channel 代理"]
  Q -->|"只要叫一声、拿个结果<br/>~10ms"| G["gateway 级<br/>读 .moss 文件 → http → 总线"]
  Q -->|"只发出去、不要回执<br/>10–40ms"| W["wire 级<br/>裸 zenoh put"]
  P --> Z["同一张 zenoh 网"]
  G --> Z
  W --> Z
  N["gateway 不改变 wire ⊂ project ⊂ matrix 的包含关系<br/>它改变的是「客户端必须进去」这件事——现在可以停在门口"]
  Z --- N
  classDef z fill:#e3f0ff,stroke:#2b6cb0,stroke-width:2px,color:#000
  classDef n fill:#fff3cd,stroke:#b8860b,color:#000
  class Z z
  class N n