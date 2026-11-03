---
label: "plan-spawn"
kind: "mermaid"
description: "clone spawn：brief 由 deps 编译；reply 或 session；结果回 release。"
created_at: "2026-11-04T01:20:15"
---
flowchart TB
    N["node"] --> BRIEF["brief<br/>anchor + deps + task + accept"]
    BRIEF --> Q{"kind"}
    Q -->|reply| R["bypass run<br/>one turn"]
    Q -->|session| S["fork branch<br/>seed from anchor<br/>workspace"]
    R --> RES["{exit, out, err}"]
    S --> RES
    RES --> REL["release · backprop"]
