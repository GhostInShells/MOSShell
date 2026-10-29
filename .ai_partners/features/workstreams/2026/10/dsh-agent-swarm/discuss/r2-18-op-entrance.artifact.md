---
label: "r2-18"
kind: "mermaid"
description: "op 的入口该是解释器还是协议：方案一把解释器地址放进易失通道；方案二把发现收窄成一个端口文件，且不禁止 python 客户端"
created_at: "2026-10-29T17:46:32"
---

flowchart TB
  Q["问题：op 的入口是解释器，还是协议"]
  Q --> P1["方案一：swarm 给一个 python，路径随 matter 提示带进去"]
  Q --> P2["方案二：op 退化成 bus 协议，curl 本地端口"]
  P1 --> A1["优点：编程灵活、moss 环境发现仍有效"]
  P1 --> A2["问题：解释器地址进了易失通道（模型上下文），换机器即失效"]
  P2 --> B1["优点：完全协议化、与 moss 解耦、感知面控制面同构"]
  P2 --> B2["代价：发现收窄成一个端口文件；但不禁止 python 客户端"]
  B2 --> C["分流判据：要票的 op 走总线；纯本地且不要票的走文件系统"]
  P2 --> D["总线已经有了：zenoh 层级 key + 通配 + liveliness，缺的只是 HTTP facade"]
  D --> E["facade 就是一个 moss node：curl 可达，真身在总线上"]
