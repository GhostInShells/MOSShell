---
label: "r2-08"
kind: "mermaid"
description: "术语澄清：一个名词扛了四个轴——归属/状态/执行/控制；控制 op 分三类，动的东西不同（会话 / run / 卷宗）"
created_at: "2026-10-29T16:16:58"
---

flowchart TB
  Q["问题：这些词各回答哪个轴"]
  Q --> A1["归属：这活归谁 → 目录树上的位置"]
  Q --> A2["状态：断线后从哪重建 → 卷宗（最小有状态单元）"]
  Q --> A3["执行：现在谁在跑 → run（一次 session 绑定）"]
  Q --> A4["控制：谁能对它做什么 → op"]
  A2 --> C1["一个卷宗，一生可以有多个 run"]
  A3 --> C1
  A4 --> O1
  subgraph OPS["控制 op 分三类，动的对象不同"]
    direction TB
    O1["注入：inject 追加 / steer 改方向 → 动运行中的会话"]
    O2["生命周期：pause / resume / cancel / rebuild → 动 run"]
    O3["事实：refresh / rollback → 动卷宗，不动 session"]
  end
