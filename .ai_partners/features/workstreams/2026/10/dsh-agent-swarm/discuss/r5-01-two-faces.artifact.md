---
label: "r5-01-two-faces"
kind: "mermaid"
description: "一个 model runtime 的三个面：入口 / 中间（loop+session）/ 控制。协议只在两端有手"
created_at: "2026-11-01T22:31:40"
---
flowchart TB
  A["① 入口<br/>harness / mindflow<br/>prompt · turn · impulse"]
  B["② 中间那段<br/>loop：踢出去之后、下一脚之前谁说了算<br/>session：上下文与历史"]
  C["③ 控制<br/>tools / channels / CTML"]
  A --> B
  B --> C
  C -->|"结果 / 事件"| B
  D["ACP 够得着的只有两处：prompt、update / cancel<br/>摸不到中间"] -.->|"所以 automation-only"| B
  classDef gap fill:#fff3cd,stroke:#b8860b,stroke-width:2px,color:#000
  class B gap