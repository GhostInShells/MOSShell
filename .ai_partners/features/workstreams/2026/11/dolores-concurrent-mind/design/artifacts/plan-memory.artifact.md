---
label: "plan-memory"
kind: "mermaid"
description: "记忆：state changed → checkpoint；compact 可调用；archive → confluence → 父提交吸收。"
created_at: "2026-11-04T01:20:24"
---
flowchart LR
    RUN["run"] --> CK{"state changed?"}
    CK -->|yes| CM["commit<br/>checkpoint"]
    CK -->|no| CONT["continue"]
    CM --> CP["compact()<br/>commit + new epoch"]
    CP --> WARM["old session warm"]
    CL["archive"] --> CONF["confluence<br/>→ parent inbox"]
    CONF --> ABS["parent commit absorbs<br/>(atomic rename)"]
