---
label: "r5-11-proxy-vs-gateway"
kind: "mermaid"
description: "反向代理 vs 网关：判别式是「谁拥有路由表」；两者可组合，不冲突"
created_at: "2026-11-02T00:44:52"
---
flowchart LR
  subgraph RP["反向代理"]
    direction TB
    R1["知道：上游在哪"]
    R2["不知道：上游内部有什么"]
    R3["404 由上游判"]
    R4["转发，不改协议语义"]
  end
  subgraph GW["网关"]
    direction TB
    G1["知道：内部有什么（manifest）"]
    G2["manifest 由声明聚合而来"]
    G3["404 自己同步判"]
    G4["翻译：http ↔ 总线"]
  end
  X["判别式<br/>谁拥有路由表"]
  RP --- X
  GW --- X
  Y["组合方式<br/>gateway 只听 localhost<br/>要对外 → 前面挂反向代理<br/>LAN 与 TLS 归代理，token 归 gateway"]
  RP --- Y
  GW --- Y
  classDef y fill:#d8f5d0,stroke:#2a7,stroke-width:2px,color:#000
  class Y y