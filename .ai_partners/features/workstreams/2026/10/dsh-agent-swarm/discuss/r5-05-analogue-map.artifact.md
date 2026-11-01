---
label: "r5-05-analogue-map"
kind: "mermaid"
description: "对标物不在 agent 行业，不等于没有；它们分别在 OS 和人类组织里"
created_at: "2026-11-01T22:50:00"
---
flowchart TB
  Q["这一项在 agent 行业里有没有对标物？"]
  Q --> A["有 / 半有<br/>1 AgentHive：每个产品一个 adapter<br/>3 AgentSwarm：都在发明自己的寻址<br/>7 team：绑在共享 session 日志上"]
  Q --> B["没有，但 OS 里有<br/>2 认知面 = ps / top / journalctl<br/>4 binary = CLI 可执行物<br/>5 territory = 文件系统 + git"]
  Q --> C["没有，但人类组织里有<br/>6 case = 工单 / 公文"]
  A --> P["行业把该松绑的都绑进了 harness<br/>所以我们没有对标物是**可预测的**"]
  B --> P
  C --> P
  P --> R["判据：若某项既无对标物、又不能由「解除绑定」解释<br/>那才是我们错了"]
  classDef ok fill:#d8f5d0,stroke:#2a7,stroke-width:2px,color:#000
  class P,R ok