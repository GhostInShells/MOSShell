---
label: "r5-14-team-shape"
kind: "mermaid"
description: "team 这一版：append-only 三份数据、没有游标、读面只给最近 N 条、唤醒靠可丢的队列提示"
created_at: "2026-11-02T01:34:54"
---
flowchart LR
  subgraph TD["team 目录 · append-only"]
    direction TB
    R["roles.jsonl<br/>role → dir / session 绑定 / 可选 address"]
    G["groups.jsonl<br/>group → members + 自解释"]
    L["groups/g/log.jsonl"]
  end
  P["post(from, to?, ref?, body)"] --> L
  P --> Q["给相关 role 入队一条提示<br/>可丢"]
  Q --> K["kickoff 那一轮"]
  RD["read · 只给最近 N 条<br/>没有 --since"] --> L
  K --> RD
  K --> I["role 的目录<br/>instruction + ops 指针（不是副本）"]
  RD --- NC["没有游标<br/>忘了就重读"]
  classDef nc fill:#fff3cd,stroke:#b8860b,stroke-width:2px,color:#000
  class NC nc