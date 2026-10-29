---
label: "r2-21"
kind: "mermaid"
description: "一份 blueprint 三个投影（人 / ghost / agent），三个投影都不手写；翻译层数 = 进程边界数，多一层就多丢一次类型"
created_at: "2026-10-29T21:45:34"
---

flowchart TB
  Q["问题：blueprint 怎么同时服务三个用户，又不长出翻译层"]
  B["Blueprint：一份声明"]
  B --> P1["人的投影：结构 + 状态 + 历史"]
  B --> P2["ghost 的投影：地址 + 契约 + 投递语义"]
  B --> P3["agent 的投影：反射出来的签名 + 压缩面"]
  P1 --> R["三个投影都不手写，都是 Blueprint 的函数"]
  P2 --> R
  P3 --> R
  R --> C["判据：一个事实只能有一个声明点；三个面之间不能有只有一方知道的事实"]
  Q --> T["翻译层数 = 进程边界数"]
  T --> T1["blueprint <-> module A/B：同进程，零翻译，类型活着"]
  T --> T2["ghost <-> instance：一个进程边界，一次翻译，就是总线"]
  T --> T3["再加 http / bash = 第二次翻译，只有非 python 消费者才值得"]
