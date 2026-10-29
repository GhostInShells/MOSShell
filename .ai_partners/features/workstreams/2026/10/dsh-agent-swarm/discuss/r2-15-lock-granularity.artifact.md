---
label: "r2-15"
kind: "mermaid"
description: "锁不能挂在进程树上（job 化会切断父子）；锁键是 (matter, run)，进程只是持有者；意图锁 + 资源锁两层，顺序固定，pull 只读"
created_at: "2026-10-29T17:22:58"
---

flowchart TB
  Q["问题：锁挂在进程树上，还是挂在案上"]
  Q --> P["现状：sh 一个进程，它调用的 op 是子进程"]
  Q --> L["两种锁，两种失败模式"]
  P --> P1["job 化会切断父子（disown / setsid / reparent 到 init）"]
  P1 --> P2["所以进程树不是稳定的锁键"]
  L --> L1["意图锁：键是 (matter, run)，TTL + 心跳，记录落在 facts"]
  L --> L2["资源锁：flock 文件 / 契约，短的、硬的、OS 保证"]
  L --> L3["顺序固定：matter -> 契约 -> 文件，否则死锁"]
  L --> L4["pull 必须只读；要写就另开一条 op 走锁"]
  L1 --> J["后台 job 与前台 pull 共存的前提：一个持锁、一个只读"]
