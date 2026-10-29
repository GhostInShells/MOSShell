---
label: "r2-10"
kind: "mermaid"
description: "总线上移：matter/op/scope/facts 在上，执行运行时在下；成立与否看「新增一个运行时要不要改中心」，三条不变量"
created_at: "2026-10-29T16:39:07"
---

flowchart TB
  subgraph BUS["通讯总线（进程 A）· 必须在上面"]
    direction TB
    M["matter 树：父子关系 + 地址"]
    O["op：地址 + 契约 + 投递语义"]
    S["scope：matter + session + turn"]
    F["facts：通过 matter 引用"]
  end
  subgraph RT["执行运行时 · 在下面，可替换"]
    direction TB
    R1["dsh harness（进程 B）"]
    R2["ghost / agent graph"]
    R3["任意开源项目 + adapter"]
  end
  O -->|"广播 / 定向确认分发"| R1
  O --> R2
  O --> R3
  R1 -->|"子进程调用 skill op"| O
  subgraph INV["成立的三条不变量"]
    direction TB
    I1["总线无状态：全部知识能从 facts 重建"]
    I2["op 必须声明投递语义：至多一次 / 至少一次 + 去重键"]
    I3["scope 只传身份；存活走心跳；投递到死 run 要有定义结果"]
  end
  M --> I1
  O --> I2
  S --> I3
