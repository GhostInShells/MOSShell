---
label: "r2-14"
kind: "mermaid"
description: "Session 上下文：三层注入的生命周期不同（前两层进 KV 前缀，第三层只发差量）；等待是三个原语，审批三态必须分开；scope 必须是签发的凭据"
created_at: "2026-10-29T17:16:08"
---

flowchart TB
  A["Agent Instruction：族级常量，每 session 一次"]
  B["matter instruction：协议常量，每 session 一次"]
  C["matter content：案级，每轮只发差量 + 指针"]
  A --> L
  B --> L
  C --> L
  L["循环：执行 context"]
  L --> E["插入事件 / 生命周期打断"]
  E --> W["wait / wait_until / wait_for —— 超时也是一种结果"]
  W --> AP["审批 = 带否决权的等待：approve / deny / timeout 三态必须分开"]
  L --> U1["主动上行：agent push（可丢，只是提示）"]
  L --> U2["被动上行：外部 pull（可从 facts 重建）"]
  E --> S["scope：必须是总线签发的凭据；拿不到就拒绝"]
