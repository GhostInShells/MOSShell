---
label: "r2-01"
kind: "mermaid"
description: "R2 剪枝一览：一个问题、一个分叉、三栏（讨论附注，非结论）"
created_at: "2026-10-29T14:53:07"
---

flowchart TB
  Q["这一轮要回答的：并行之后，调度住在哪"]
  Q --> R{"住在 runtime 里？"}
  R -->|"是"| H["harness：任务推进耦合运行时"]
  R -->|"否"| S["seam：只读 facts、发起动作、把结果写回"]
  subgraph CUT["本轮剪掉"]
    direction TB
    C1["op 面向 develop 阶段的 agent"]
    C2["运行时资源规划"]
    C3["cluster / launcher 当故障域"]
  end
  subgraph KEEP["本轮留下"]
    direction TB
    K1["runtime = seam"]
    K2["栈回模型上下文 + checkpoint"]
    K3["OS 优先：能力靠拾取"]
    K4["横向切割：加一档不改接口"]
  end
  subgraph OPEN["仍悬着"]
    direction TB
    O1["读放大"]
    O2["可调试性"]
    O3["单写者 vs 多读者"]
    O4["mindflow 的落点"]
  end
