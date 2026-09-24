---
title: Zenoh Backpressure Governance
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P2
created: 2026-09-24
updated: 2026-09-24
depends: []
milestone:
description: >-
  全面治理仓库内 zenoh egress 的背压 / 静默丢包问题：集中配置 + 统一的防御面。
---

# Zenoh Backpressure Governance

> Use `moss features set-status zenoh-backpressure-governance <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

仓库大量用 zenoh 做 transport（signal / output / stream / topic / parameter / qa /
operator / presence / bridge），但 egress 侧没有线程卸载。触发点是一次对
`matrix_channel.py` 的线程阻塞排查：mesh 的 cell event → `send_signal` → `session.put`
落在 event loop 线程上，担心背压会 stop the world。

核实后发现两个更准确的威胁，都不是"卡死 loop"：

1. **默认拥塞控制是 DROP，不是 BLOCK**（zenoh 1.9.0 `CongestionControl.DEFAULT = DROP`）。
   put 在队列满时**静默丢弃**，发送方无感知。这是"静默丢消息"。
2. **drop 起点很低**：`batch_size` 默认 64KB，每优先级传输队列 1–4 批 → 每条链路每个
   priority lane 的 in-flight 缓冲约 64–256KB。一个视觉帧 / 一坨 TTS chunk 就能顶到。

所以问题不是"背压卡死 loop"（DROP 下不会），而是"**静默丢**"（不可接受）+"**若未来
切 BLOCK 则 stop the world**"（需要卸载）。治理目标是：让 zenoh egress 的背压 / 丢包
**有界、可见、可配置**，而不是每处调用点各自裸写。

## 已核实事实（下一个实例勿重复推导）

- `zenoh.CongestionControl.DEFAULT = DROP`；全仓 grep 无任何 `congestion_control=` /
  `BLOCK` 显式设置，所有 put 走 DROP。
- egress `.put(` 调用点约 14 处：`zenoh_session.py` 3（add_signal/output/pub_stream_delta）、
  `zenoh_qa.py` 2、`zenoh_parameters.py` 2、`zenoh_topics.py` 2、`zenoh_operator.py` 1、
  `zenoh_service_terminal.py` 1、`bridges/zenoh_bridge` 2、`zenoh_presence.py` publish。
- 配置咽喉 = `matrix/networks/zenoh_network.py:_create_zenoh_session`，是生产 runtime 的**唯一**
  session 实例化点。所有消费方（session/topic/parameter/qa/operator/presence/mesh/hub）都走
  IoC `force_fetch(zenoh.Session)` 拿同一个 session，无人绕开。唯一的缺口是 `extra` 透传通道
  **存在但从未被填过**——即集中 transport config 能力未启用，不是"被绕过"。
- `zenoh_topics.py:532` 和 `bridges/zenoh_bridge/_suite.py:18` 的 `zenoh.open(zenoh.Config())`
  是 test suite（`TopicServiceSuite` / `BridgeTestSuite`，`suite_for_test.py` 谱系），只在测试里
  跑，不在生产 wiring 里。
- 入栈侧已优雅丢包：`zenoh_mesh.py` 用 janus `put_nowait` + drop，不阻塞 loop。
- query 路径已 to_thread：`liveliness().get` / `session.get` 都已 `asyncio.to_thread`。
- 主 session 的 `zenoh.open` 也已 to_thread（`zenoh_adapter.py:108`）。

## Key Decisions

<!-- Record each meaningful design choice. This is what the next AI incarnation reads first. -->

1. **集中防御分两层，先 config 后 wrapper。** config 只能调队列大小 / batch / 带宽 /
   batching，**不能**改默认拥塞控制（DROP→BLOCK 是 per-put/per-publisher 的，不是 config
   默认）。因此：
   - 第一层（纯 config，零调用点改动）：在 `zenoh_network.py` 集中注入
     `transport/unicast/queue_size`、`batch_size`、`max_bandwidth`、`batching.backoff`。
   - 第二层（若要 BLOCK 语义或让 drop 可见）：共享的 session wrapper / put 漏斗。
2. **是否切 BLOCK 是语义决策，不是补防御。** 保持 DROP 时，正确防御 = ingest 限频
   （`_dispatch_event` 按 address 窗口合并）+ 让丢包可见（计数告警）。切 BLOCK 才需要
   单写线程 + 有界队列卸载。
3. **启用而非补全配置咽喉。** 生产 session 已收口到 `zenoh_network.py`（单一实例化，
   全走 IoC）。真正的工作是把 `extra`（或等价的 typed 字段）填上 transport 调优，而不是
   去"收口 opens"——生产路径本来就没有裸 open。

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->

- DROP 是无声的：zenoh put 侧没有"我丢了 N 条"的回执，让丢包可见需要在发送侧自建计数
  或改用 BLOCK。
- 队列容量 = `batch_size × N批`，量级几百 KB。控制面消息（signal/output/topic/param/qa，
  均 <1KB）在 loopback host 上基本顶不到；真正打爆的是 bulk 媒体（`pub_stream_delta` /
  TTS 音频 / 视觉帧 / 推图）。
- "远端可喂"：本机 session 的 config 防不住远端 cell 的洪水，洪水淹的是**入栈**回调链，
  那是 ingest 限频的职责，不是 config 的。
