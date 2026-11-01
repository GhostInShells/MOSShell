---
label: "r5-13-three-trees"
kind: "mermaid"
description: "最重要的拓扑变更：case / team / agent 三棵树靠 moss bin 解耦，只在 kickoff 那一点相遇"
created_at: "2026-11-02T01:34:53"
---
flowchart TB
  subgraph OLD["旧：一个 harness 里的状态机"]
    direction TB
    H["harness 进程"] --> R["roster"]
    H --> TB["task board"]
    H --> PM["peer messages"]
    H --> L["agent loop"]
  end
  subgraph NEW["新：三棵独立的树"]
    direction TB
    C["case · 轨迹 + bins"]
    T["team · roles / groups / log + bins"]
    A["agent · instruction + session 绑定"]
  end
  K["唯一耦合点 · kickoff<br/>--wd <某个目录> + 一句 prompt"]
  C --> K
  T --> K
  A --> K
  K --> P["一次进程：跑一轮，退出"]
  P -.->|"写回"| C
  P -.->|"写回"| T
  N["三者不必互相知道<br/>可以是不同 skill 目录下的一堆 bin"]
  P --- N
  classDef old fill:#f2f2f2,stroke:#999,color:#333
  classDef core fill:#e3f0ff,stroke:#2b6cb0,stroke-width:2px,color:#000
  class OLD old
  class K,P core