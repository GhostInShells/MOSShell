---
label: "plan-node-fsm"
kind: "mermaid"
description: "节点状态机：ready 才能被 claim；失效一律 cancelled。"
created_at: "2026-11-04T01:19:59"
---
stateDiagram-v2
    [*] --> pending
    pending --> ready: deps bound + when + accept
    ready --> claimed: claim
    claimed --> running: spawn
    running --> done: release ok
    running --> failed: release err
    running --> cancelled: patch invalidates
    pending --> parked: human
    parked --> pending: resume
    done --> archived
    failed --> archived
    cancelled --> archived
