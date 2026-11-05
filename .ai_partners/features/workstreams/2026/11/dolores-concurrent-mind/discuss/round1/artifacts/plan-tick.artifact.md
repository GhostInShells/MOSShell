---
label: "plan-tick"
kind: "mermaid"
description: "scheduler tick：fold → ready → select → claim → spawn → backprop。"
created_at: "2026-11-04T01:20:07"
---
flowchart LR
    OPS["ops"] --> FOLD["fold"]
    FOLD --> READY["ready set"]
    READY --> SEL["select k<br/>value/cost + UCT"]
    SEL --> CLAIM["claim (vloss)"]
    CLAIM --> SPAWN["spawn"]
    SPAWN --> RES["result"]
    RES --> BACK["backprop"]
    BACK --> FOLD
