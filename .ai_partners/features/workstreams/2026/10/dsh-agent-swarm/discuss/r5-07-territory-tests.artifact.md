---
label: "r5-07-territory-tests"
kind: "mermaid"
description: "四条验收：百年 / 分形 / 独立 / 缺席；过不了的那一条就是要补的最小东西"
created_at: "2026-11-01T23:14:44"
---
flowchart TB
  Q["一块目录凭什么算 territory？"]
  Q --> T1["① 百年<br/>只给目录、不给任何运行时<br/>三个面都读得出来"]
  Q --> T2["② 分形<br/>子目录单独拷走<br/>它自己还是一块"]
  Q --> T3["③ 独立<br/>删掉所有 session<br/>一个字节都不变"]
  Q --> T4["④ 缺席<br/>没有未来面的目录<br/>仍然是一块，只是没有未来"]
  T1 --> R["四条全过才叫 territory<br/>过不了的那一条 = 要补的最小东西"]
  T2 --> R
  T3 --> R
  T4 --> R
  classDef ok fill:#d8f5d0,stroke:#2a7,stroke-width:2px,color:#000
  class R ok