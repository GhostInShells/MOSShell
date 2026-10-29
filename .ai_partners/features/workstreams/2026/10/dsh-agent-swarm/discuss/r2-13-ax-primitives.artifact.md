---
label: "r2-13"
kind: "mermaid"
description: "AX 四原语对位：Task=run、Workspace=workspace、Gateway=op 入口、Model=供给；让出「怎么活着」，不让「发生过什么」和「谁持有案」"
created_at: "2026-10-29T16:54:53"
---

flowchart LR
  subgraph AX["AX 的四个声明原语"]
    direction TB
    T["Task = 一次有状态的 actor 会话"]
    W["Workspace = 沙箱工作目录"]
    G["Gateway = 入口"]
    M["Model = 模型供给"]
  end
  subgraph US["我们的东西"]
    direction TB
    R["run = 一次承办（可让出）"]
    WD["workspace = agent 可写目录（已有，KD5）"]
    OP["op = 地址 + 契约（我们的入口）"]
    MT["matter = 案（不能让）"]
    F["facts = 共享真相（不能让）"]
  end
  T -->|"对位"| R
  W -->|"对位"| WD
  G -->|"对位"| OP
  M -->|"由 runtime 供给"| R
  R -->|"承办"| MT
  MT -->|"住在"| F
  subgraph LINE["让出的那条线"]
    direction TB
    L1["让出：怎么活着——挂起/恢复/资源边界/多路复用/冷启动"]
    L2["不让：发生过什么（日志所有权）、谁持有案"]
  end
  R --> L1
  F --> L2
