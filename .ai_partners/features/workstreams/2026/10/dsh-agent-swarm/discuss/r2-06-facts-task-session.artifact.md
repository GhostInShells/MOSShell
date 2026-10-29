---
label: "r2-06"
kind: "mermaid"
description: "facts→task→session 三重投影：状态宿主是 task；三档存储按「恢复单位」分，能 merge 的进 git，不能的进 db"
created_at: "2026-10-29T15:44:19"
---

flowchart TB
  Q["问题：facts / task / session 三重投影，谁持有状态，各自落在哪一档"]
  Q --> F["facts：唯一真相"]
  F -->|"提炼"| T["task.md：最小有状态单元，绑定 skill / knowledge / feature"]
  T -->|"在 append-only session 里还原"| S["session：智力理解后的上下文"]
  S -->|"执行"| A["action log：op → action 的记录"]
  A -->|"回写"| T
  subgraph STORE["三档存储：恢复单位不同"]
    direction TB
    D1["裸文件：不可恢复，只允许丢"]
    D2["git：按快照恢复，可 merge"]
    D3["jsonl / DB：按重放 + 备份恢复，不可 merge"]
  end
  T --> D2
  S --> D3
  A --> D3
  subgraph HOST["状态宿主"]
    direction TB
    H1["有状态的是 task，不是 session"]
    H2["判据：解绑 session 后，能否只凭 task 目录重建「接下来做什么」"]
    H3["skill 是 task 的资产，不是容器"]
  end
  T --> H1
