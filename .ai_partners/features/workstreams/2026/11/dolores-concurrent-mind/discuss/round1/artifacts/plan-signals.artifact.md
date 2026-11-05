---
label: "plan-signals"
kind: "mermaid"
description: "通讯：clone 事件走 mindflow 总线；按 target 寻址；commit 只广播 title。"
created_at: "2026-11-04T01:21:33"
---
flowchart LR
    CL["clone"] -->|"started / progress / final"| BUS[("mindflow bus")]
    BUS -->|"target = main"| MAIN["main"]
    BUS -->|"target = branch"| CL2["clone"]
    HOOK["commit hook"] -->|"coord + title"| BUS
    BUS -->|"titles"| SUB["subscribers"]
