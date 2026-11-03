---
label: "plan-build"
kind: "mermaid"
description: "建造顺序：10 个竖切；第一个是 plan store + reply clone。"
created_at: "2026-11-04T01:20:41"
---
flowchart LR
    S1["1 plan store"] --> S2["2 reply clone"]
    S2 --> S3["3 scheduler"]
    S3 --> S4["4 session clone"]
    S4 --> S5["5 invalidation"]
    S5 --> S6["6 comms"]
    S6 --> S7["7 surface"]
    S7 --> S8["8 compact + confluence"]
    S8 --> S9["9 economics"]
    S9 --> S10["10 split / dig"]
