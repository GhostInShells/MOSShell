---
label: "r5-08-case-lifecycle"
kind: "mermaid"
description: "case 的一生 = 一条 branch 的一生：建分支 / commit 步进 / fork 并行 / confluence 上报 / 归档留坐标"
created_at: "2026-11-02T00:01:47"
---
flowchart LR
  K["kick off<br/>建 branch + CASE.md"]
  C["checkpoint<br/>commit（一步一锚点）"]
  F["并行<br/>从当前 commit fork 子 branch"]
  X["confluence<br/>子线向父线上报，不是 merge"]
  A["归档<br/>branch 留坐标<br/>commit 目录搬走"]
  K --> C
  C -->|"步进"| C
  C --> F
  C --> X
  F --> X
  X --> A
  C -->|"直接完结"| A
  classDef pt fill:#e3f0ff,stroke:#2b6cb0,stroke-width:2px,color:#000
  class K,C,F,X,A pt