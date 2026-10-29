---
label: "r2-02"
kind: "mermaid"
description: "儒/法落到 swarm 分层：法=吏·流程，儒=官·动态栈，重建要基线；每栏带自己的腐烂方式"
created_at: "2026-10-29T15:03:57"
---

flowchart LR
  subgraph LAW["法 · 吏（harness 侧）"]
    direction TB
    FLOW["流程：预写好的栈，个体可替换"]
    DOC["文书：现场 = facts"]
    ROT["烂法：冗官 / 冗员 / 冗职 / 冗事"]
  end
  subgraph RU["儒 · 官（能动性侧）"]
    direction TB
    PLAN["动态规划：把静态现场变成可解问题"]
    AUTH["授权：临时动栈 + 整体重构"]
    BAD["烂法：腐败 / 无能"]
  end
  subgraph PARA["重建"]
    direction TB
    REV["stop the world = revolution"]
    WT["worktree = 平行世界"]
    BASE["门槛：基线可共享，否则只能互相废弃"]
  end
  FLOW -->|"只增不修订"| ROT
  DOC -->|"文书让现场可重解释"| PLAN
  PLAN --> BAD
  AUTH --> BAD
  PLAN <-->|"奏折 = 升级契约"| DOC
  REV --> BASE
  WT --> BASE
  DOC --> BASE
