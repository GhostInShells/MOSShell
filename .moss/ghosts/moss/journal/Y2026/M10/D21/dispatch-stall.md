---
name: dispatch-stall
description: D21 17:13 那次"声音突然没了 + DSH 像卡死"的现场复盘与证据
---

# D21 17:13 两个卡死

他低血糖去补糖之前说的两个 bug：DSH 那边卡死、音频那边崩了。他走后我把
`moss.log` 和那一段 dsh 会话翻出来对时间戳，结论是**同一个根因的两个面**。

## 一、命令卡在 pending，永不执行

时间线（全来自日志）：

- 17:13:04 我发了一个 `<all>`：子 channel `webview_screen` 的 7 个 `box`，加 `<_>` 里 7 句 `say`。
- 7 句 say 依次播完（17:13:10 → 17:13:35），`webview_screen` 17:13:17 就 `pending tasks 0`。
- 17:13:58 下一句 `<say>` 只做了 prepare：TTS 建流、喂文本、合成、音频都收到了，
  **但命令体一次都没跑**（没有 `start task`、没有 `Starting playing TTS stream`）。
- 17:14:10 他说听不见；`say replace=true`、`clear_speech` 全都只 prepare 不执行。
- 17:15:58 关运行时，运行时自己报出那句：`CLEARED: cleared by runtime` —— 它一直躺在
  `__main__` 的 `_pending_tasks` 里，没人消费。

机制在 `_tree_channel_runtime.py` 的 `_consume_task`：消费者是串行的，碰到
`_executing_blocking_task` 就 `await .wait()`，**没有超时、没有日志**。一个永不结束的
blocking task 就把整条轨永久堵死，而 `clear_speech` 自己也在队列里排队 —— 所以声音没法
从对话里救回来，只能重启。**"音频崩了"是表象，是调度堵死。**

嫌疑人：17:13:35.5 同时起跑的两个 `__scope_exit__`（`<_>` 的、`<all>` 的）。写了两个复现
用例（本地子 channel、代理子 channel，都是 `<all>` 套子 channel scope），**都通过** —— 说明
"<all> 套子 scope 就不收口"不是充分条件，触发器还没找到。

## 二、`moss_wait_next` 根本退不出去

同一段会话里连着三次拿到：

> a percept arrived before the yield — the moment is pushed and this turn continues instead of ending

代码在 `_run.py` 的 `_handle_wait_next`：等动作结束时若有人递了 percept，就让输，回合继续。
这个设计假设 percept 稀少且来自人。可那会儿命令卡住，状态 moment 一秒一条 —— 于是每次
yield 都输，回合永远结束不了，外面看就是 DSH 卡死、token 白烧。

## 我做了什么

- 两个复现用例：`test_scope_child_channel_scope.py`、`test_scope_proxy_child_scope.py`（都绿，
  留作 scope 收口的契约证据）。
- 诊断写进 `.ai_partners/features/workstreams/2026/10/command-dispatch-stall/FEATURE.md`，
  含三条提议：调度看门狗（blocking task 超 N 秒就点名 + dump 栈）、scope 不许无限持有轨道、
  `clear_speech` 必须是 `call_soon` 逃生口。
- **没有改产品代码**，等他点头。

## 给未来的自己

判据还是那条：**先看时间戳，别推理**。我先猜"子 channel 的 scope 永不收口"，写了两个用例，
两个都绿 —— 猜错了，但错得便宜，而且留下了证据。下次再遇到这种卡死，先想"能不能拿到栈"，
拿不到就先装看门狗，别在黑箱里猜第二遍。
