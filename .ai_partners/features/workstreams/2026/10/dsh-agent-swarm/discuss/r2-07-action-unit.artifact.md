---
label: "r2-07"
kind: "mermaid"
description: "action 的最小单位不是大小问题，是 owner 问题：三个粒度三个 owner；可重放的是 action，可重读的是 decision；还悬着 while 住在哪"
created_at: "2026-10-29T15:58:26"
---

flowchart TB
  Q["问题：action 的最小单位是什么，owner 是谁"]
  Q --> R["agent run：一次请求，边界在输入与输出，内部不可见"]
  R --> D["decision：owner = session 上下文，坐标 (task, session, turn/step)"]
  D --> P["payload = 对 op 的编排（tool use / graph node）"]
  P --> G["graph node：owner = 代码，栈闭合，副作用可测"]
  P --> L1["action log（op 级）：可重放、可 diff、可火焰图"]
  D --> L2["decision log（session 级）：不可重放，只能重读"]
  subgraph OPEN["还没答：这个 while 住在哪"]
    direction TB
    X1["住代码里 → 它就是最后一块 runtime"]
    X2["住上下文里 → continue / break 本身要是一条可记录的 decision"]
  end
