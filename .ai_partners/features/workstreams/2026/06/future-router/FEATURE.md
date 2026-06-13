---
title: Future Router
status: completed
priority: P2
created: 2026-06-13
updated: 2026-06-13
depends: []
milestone:
description: >-
  进程内 Future 路由基建 — concurrent.futures.Future + 字符串协议 + 防崩接口 + on_create 回调，
  作为 approval channel / async task channel 等异步交互场景的统一基座。
status_note: tools/future_router.py 落地。create/call/get/list_pending/list_done/resolve/reject/cancel/on_create + wait_future。仅进程内线程安全，跨进程由上层协议组合实现。
---

# Future Router

> Use `moss features set-status future-router <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

session-communication-bus 设计中的 FutureManager（跨进程 future、SQLite + Zenoh 通知）
被暂停优先级，但其核心使用场景 — 审批模块、模型发起的异步任务、跨线程 future
协调 — 在进程内仍然存在，且没有统一基建。

每次需要"一个线程提交请求，另一个线程/协程在异步路径上等结果"的场景，
都散落实现 dict + lock + Future，难以观察、难以防崩、调试体验差。

future-router 是这个场景的最小通用基建：
- **issuer 侧**：create 一个 future，await 结果（或登记 id 后续查）
- **executor 侧**：通过 on_create 回调感知新任务，调 resolve/reject/cancel 设置结果
- **观察侧**：list_pending / list_done 看进行中与最近完成的，配合 debug / TUI

与 codex-module-sandbox → module-eval-channel 的关系一致：
本 feature 是基建，未来的 approval channel / async task channel 在其之上组合而成。

## 与 FutureManager（跨进程）的定位区分

| 维度 | future-router (本 feature) | FutureManager (session-communication-bus) |
|------|---------------------------|-------------------------------------------|
| 边界 | 单进程内多线程 | 跨进程（Cell ↔ Cell） |
| 底层 | concurrent.futures.Future | SQLite + Zenoh notifications |
| 协议 | str → str | task_json → result_json |
| 状态机 | future 自带 (done/cancelled/exception) | 显式 pending/resolved/rejected/cancelled/timed_out |
| 适用 | 进程内审批 / 异步任务 / 线程间协调 | 跨进程审批 / Cell 间异步请求 |

进程内场景**不应等 FutureManager**。两者协议形态相近，未来若 FutureManager 实现，
可以让 future-router 成为其单进程同构降级实现 — 但目前不强求 API 一致。

## Design Index

- 实现：`src/ghoshell_moss/tools/future_router.py`
- 相关基建讨论：`.ai_partners/features/workstreams/2026/05/session-communication-bus/FEATURE.md`（FutureManager 设计）
- 复用对象：`ThreadSafeFuture`（`core/helpers/asyncio_utils.py`）—— 经评估在本场景无收益，不复用

## Key Decisions

### KD1: 底层用 `concurrent.futures.Future` 而非 `ThreadSafeFuture`

`ThreadSafeFuture` 包装 `asyncio.Future`，set_result 通过 `call_soon_threadsafe`
调度到 event loop，`done()` 状态非即时。`concurrent.futures.Future` 标准库
天然线程安全、状态即时、`add_done_callback` 在 set 线程同步触发。

router 内部用 `add_done_callback` 做 pending→done 迁移；同步触发使得这套
登记/归档机制不依赖 event loop，纯线程安全语义干净。asyncio 侧用
`asyncio.wrap_future` 一行接入。

`ThreadSafeFuture` 当初是不知道 `concurrent.futures` 的产物，本场景没有复用价值。

### KD2: 字符串协议，不引入泛型

`create(arguments: str) → Future[str]`。理由：
- 路由层不关心载荷语义，最终若要桥接跨进程必然走序列化
- str 是最小公约数，调用方在边界做编解码
- 加泛型不增表达力，反而让 router 像 typed container

若未来出现强类型场景，把 str 提升为 TypeVar 是机械变更，不影响设计。

### KD3: pending / done 列表分开，done 用有界 deque

两份列表分开返回（list_pending / list_done）—— 避免调用方对已完成 future
做"是否还能修改"的判断。done 用 `deque(maxlen=max_done, 默认 64)` 防内存泄漏，
旧条目 FIFO 淘汰。`get(id)` 跨两者搜索，因为 id 持有者可能在完成后才来取结果。

### KD4: 防崩通过 router 的 resolve/reject/cancel 返回 bool

executor 侧调 `resolve(id, result)`：
- pending 不存在 / future 已 done → 返回 False（提示外部状态已变）
- 正常 set → 返回 True

不抛 `InvalidStateError`，调用方拿 False 自行决定告警还是吞掉。
**不防御外部直接修改 future**：拿到 future 对象后任意线程 `set_result`
是合法用法 —— `add_done_callback` 保证归档路径仍然触发。

### KD5: on_create 回调锁外执行 + 异常吞掉

回调在 router 锁释放后逐个调用，避免回调里反向调 router 接口造成死锁。
单个回调抛异常被 try/except 吞掉并日志记录 —— 一个坏回调不应阻塞其他回调
或污染 issuer 路径。回调注册时间复杂度 O(1)，触发时 O(n)，不提供注销
（用例都是 channel/app 生命周期级别的注册，不需要动态摘除）。

### KD6: `call()` 糖方法 — code-as-prompt

`async def call(arguments, timeout) -> str` 不是为了"少写几行"，而是给
不熟悉 `asyncio.wrap_future` 的读者一个**可直接复制的参照实现**。
docstring 显式说明：需要 id（日志 / 外部 cancel / 关联）时直接用 `create + wait_future`。

模块级 `wait_future(future, timeout)` 同理 —— 把 wrap_future + wait_for +
取消传导封一层，让超时/取消能通过底层 future 状态传到 executor 侧。

## Implementation Notes

### Pending→done 归档由 add_done_callback 接管

无论是 router 的 resolve/reject/cancel 触发 done，还是外部直接调用
`future.set_result()` 触发 done，归档路径都是同一条 callback。
这是"不防御外部修改"成立的前提 —— 归档不依赖路径，依赖状态。

### 超时/取消的传导语义

`wait_future` 在 `asyncio.TimeoutError` 或 `CancelledError` 时手动调
`future.cancel()`，让 executor 侧通过 `future.cancelled()` 或 `future.done()`
观察到 issuer 已放弃。这是 issuer→executor 单向信号 —— executor 仍可继续
处理（毕竟工作可能已经做完），但 resolve 会返回 False，自然丢弃结果。

### 与 session-communication-bus 设计的对照

session-communication-bus 中 FutureManager 的 issuer/receiver 双视图、
状态机、timeout_at 字段都是为跨进程持久化设计。本 feature 的进程内实现
保留了 issuer/executor 角色概念，但状态由 future 自身承载 —— 这是单进程
"内存即真相源"与跨进程"持久化即真相源"的本质差异。

未来若 FutureManager 实现，**不强求**让 future-router 成为其降级实现。
两套 API 可以独立演化，应用层根据进程边界自行选择。
