---
label: "r5-12-case-inception"
kind: "mermaid"
description: "case 的 branch 目录就是 agent session 的 inception 目录：沙箱边界、工具面、上下文注入三合一"
created_at: "2026-11-02T00:44:53"
---
flowchart TB
  B["case 的 branch 目录"]
  B --> S["agent session 的 cwd<br/>dsh 沙箱边界 = SessionHeader.cwd（实测）"]
  S --> T["工具面：binary 放进目录、上 PATH<br/>不靠 harness 注入"]
  S --> C["上下文：branch view 注入<br/>父子关键上下文也走注入"]
  S --> K["移交 = 换一个 session 指向同一目录<br/>不搬任何状态"]
  P["父 case 的 branch"] -.->|"注入 view，不靠路径穿越"| C
  classDef core fill:#e3f0ff,stroke:#2b6cb0,stroke-width:2px,color:#000
  class B core