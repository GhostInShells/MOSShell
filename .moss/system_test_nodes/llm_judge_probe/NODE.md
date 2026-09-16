---
name: 'long_listen_probe'
description: '智能判停 (长程聆听) 实机测试 — assemble listener + 旁路监控 clause/segment 时序与 llm 打分'
singleton: true
exec:
  command: python
  args: main.py
---

智能判停 (长程聆听) 实机测试 node. 自己 assemble listener (`ModelListenerController`)
并旁路监控判停全链路: `on_recognition_result` 记录 clause→segment 时序,
`controller.on_score` 记录每次 llm 打分 (请求 clauses + score + cast + token).

打印聚焦三个测试点, 不刷 partial 噪声:

- `[clause] <text>` — ASR 分句 (judge 的决策点)
- `[judge] score=N cast=... tok(...) clauses=[...]` — llm 打分 (发了什么、回了什么分、多快)
- `[segment] <text> (+Δs from last clause)` — commit 切段 (clause 到 commit 的 delta)

## 测试方法

1. 起本 node (需 LLMFuncs 已配置, `small_fast_model` 可用):

       moss nodes run .moss/system_test_nodes/long_listen_probe/ -- <device_pattern>

2. 讲一段长论述 (VAD 到处分句), 观察:
   - 句中不 commit (`[segment]` 不出现在句中);
   - 停顿后正常 commit (`[judge]` score >= 7 后 `[segment]` 出现);
   - 中途开口时在飞打分被取消 (该次 `[judge]` 后无 `[segment]`).

Run in the same network scope as the producer (default scope from .moss).
