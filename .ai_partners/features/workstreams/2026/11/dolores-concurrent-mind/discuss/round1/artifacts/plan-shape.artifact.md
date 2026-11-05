---
label: "plan-shape"
kind: "mermaid"
description: "三个 store + 三个 runtime：main 只写 plan，scheduler 只调度，clone 只执行。"
created_at: "2026-11-04T01:19:45"
---
flowchart LR
    MAIN["main<br/>interaction loop<br/>only plan writer"]
    SCH["scheduler<br/>fold · ready · claim · spawn"]
    CL["clone<br/>reply | session"]
    PLAN[("plan<br/>nodes · edges · ops")]
    MEM[("memento<br/>branches · commits")]
    SESS[("session log<br/>anchors")]
    VIEW["plan view<br/>(projection)"]

    MAIN -->|ops| PLAN
    SCH -->|fold| PLAN
    SCH -->|claim + spawn| CL
    CL -->|seed| SESS
    CL -->|commits| MEM
    MAIN -->|anchor| SESS
    PLAN -.->|projection| VIEW
