---
label: "r2-12"
kind: "mermaid"
description: "调研对照：A2A 的 task 由被叫方创建、AX 用单写者换续跑；我们由发起方持有案、执行者可换，终态由验收宣告"
created_at: "2026-10-29T16:54:14"
---

flowchart TB
  Q["问题：A2A 的 task、AX 的 actor、我们的案，差在哪"]
  Q --> A["A2A v1.0：task 由被叫方创建，终态由被叫方宣告"]
  Q --> X["AX：单写者 controller + event log + snapshot，续跑"]
  Q --> M["我们：案由发起方持有，执行者可换，终态由验收宣告"]
  A --> A1["借：Agent Card 声明面 / Parts 字段名即类型 / A2A-Version 协商"]
  A --> A2["不借：任务归属在被叫方 → 执行者不可替换"]
  X --> X1["借：bursty 经济学——等审批时不该养着 actor"]
  X --> X2["不借：单写者换来续跑，我们换的是可替换"]
  M --> M1["族内透明、族间不透明；验收才是终态"]
