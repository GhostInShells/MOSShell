---
label: "r2-20"
kind: "mermaid"
description: "四层：动机 -> 名册（发现面）-> Prototype（stubs + 章程，自带知识面）-> Instance（领地 + 展示面 + 控制面）；每层一条判据"
created_at: "2026-10-29T21:35:53"
---

flowchart TB
  L0["0 · 动机：线性转并行 + 可遗忘的场"]
  L1["1 · 名册：治理发现面 —— 谁在、谁活着、谁能干什么"]
  L2["2 · Prototype：stubs + 章程 + 自带知识面，用来初始化"]
  L3["3 · Instance：真实领地，可运行；有展示面 + ghost 控制面"]
  L0 --> L1
  L1 --> L2
  L2 --> L3
  L3 -->|"长期迭代，反过来长出新的 prototype"| L2
  subgraph J["每层一条判据（可证伪）"]
    direction TB
    J1["L1：新增一个 prototype / instance，要不要改名册？不改 -> 边界对"]
    J2["L2：初始化出来的实例，能不能脱离原型作者继续长？"]
    J3["L3：杀掉实例，能不能从 facts 重建？"]
  end
  L1 --> J1
  L2 --> J2
  L3 --> J3
