---
label: "r5-02-four-points"
kind: "mermaid"
description: "四个点位 = 一个陌生平台能不能被总线化的验收表；ACP 只稳占两个"
created_at: "2026-11-01T22:31:41"
---
flowchart TB
  subgraph TEST["验收：一个陌生平台凭什么算「能上总线」"]
    direction LR
    P1["① kick off"] --> P2["② checkpoint"] --> P3["③ keep going"] --> P4["④ dismiss"]
  end
  ACP["ACP 只稳占 ① 与 ④<br/>② 有事件、没有闸门<br/>③ 中途进不去"]
  BASH["只有 bash 的 agent：四项都凑得出来<br/>kick off = 起进程 · checkpoint = 进程退出<br/>keep going = facts 重建 · dismiss = kill"]
  TEST --> ACP
  TEST --> BASH
  classDef weak fill:#ffe0e0,stroke:#c00,color:#000
  class ACP weak