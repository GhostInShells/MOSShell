---
label: "r2-05"
kind: "mermaid"
description: "op 的形状：script 有地址、skill.md 是压缩面、decorator 反射成命令；scope 要两种机制——env 传身份，租约管锁"
created_at: "2026-10-29T15:31:40"
---

flowchart TB
  Q["问题：op 是什么形状"]
  Q --> L1["script：在 OS 里有地址，bash 就是驱动器"]
  Q --> L2["skill.md：压缩面 + 声明，不读脚本体也知该不该用"]
  Q --> L3["decorator：把声明反射成 channel 命令"]
  Q --> L4["驱动者：模型 / 人 / 任意 harness"]
  L1 --> SC["scope 怎么传、锁、重建"]
  SC --> P1["pass：env 传身份——便宜，但 env 只能传身份"]
  SC --> P2["lock：必须能死掉（租约 / heartbeat），env 锁不住"]
  SC --> P3["rebuild：facts + env 还原同一 scope"]
  L4 --> J1["判据：op 若要知道自己在被谁驱动才能跑，它就不是 OS 层的"]
