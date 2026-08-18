---
title: Shell Trajectory — 观测轨迹取代上下文监控
status: in-progress
priority: P1
created: 2026-08-19
updated: 2026-08-19
depends: []
milestone:
description: >-
  ShellTrajectory 取代旧观测面 (ContextMonitor / ShellContext / InterleavedThinkingToolset),
  以 pull 型帧轨迹承载上下文缓存经济学下的观测: 帧 = events + facade delta + dynamic messages.
---

# Shell Trajectory

> Use `moss features set-status shell-trajectory <status> -m "note"` to update state.

## Motivation

interleaved thinking 主流化 + 前缀 KV 缓存经济学, 要求调整上下文策略。旧观测面三件套
(`host/context_monitor.py` ContextMonitor + `core/concepts/shell_context.py` ShellContext ABC
+ `host/interleaved_thinking.py`) 无法对齐目标, 被 ShellTrajectory 取代。

核心认识: 观测是**持久的** (跨请求、跨 compact), 上下文组装是**每请求的**。旧设计把
观测嵌进每请求的上下文构建器, 是错的; ShellTrajectory 把它拔出来做持久层。

## Key Decisions

1. **帧三分 + pull (drain)**。一帧 = shell events + facade delta + dynamic messages,
   模型主动拉取。旁路是对现实的妥协: 全双工模型不需要, 但当代模型不能边思考边插入
   返回值, ordered 思考→行为→观察必须拼帧。

2. **本期不做 hot 逻辑** (后置)。context messages 进历史 (dsh 融合无法隐藏 context)。
   历史 = compact → 重建 trajectory (epoch) + 每帧拉取。

3. **facade delta = per-channel 文本 diff + 墓碑**。增/改重发新 facade, 删发
   `<channel removed/>`。`_make_facade_body` 统一组装 (failure 短路), 逐块对比早退。

4. **时间戳分层**。channel facade 块不带时间戳 (durable 需字节稳定, 且批量 refresh 下
   大面积重复); 时间戳放 frame 层 (ephemeral)。message 用 `D19 00:01:17+8` 短格式
   (日 + 时分秒 + 时区), 年/月由 frame 的 "today" 承担。

5. **now 语义**。frame `at=` = 发送时刻 (now), message 时间戳 = 事件发生时 (冻结事实)。
   请求重试必须重新 `project(now=新时间)`, 否则模型误把上次发送时刻当 now。

6. **ack 纪律**。peek 非破坏, commit 才 drain + 推进 baseline。只有消费方能判断消费
   成功 — 模型请求失败不能丢事件 (at-least-once)。

## 取代与撤销 (dead ends)

- 两个旧 workstream **删除**: `channel-meta-dyn-static` 与 `context-cache-engineering`
  (同一命题的演化, 从未 completed), 完整历史见 git log。
- ContextMonitor (`host/context_monitor.py`, 33 tests) 无法对齐目标 → 取代。
- `shell_context.py` (ShellContext ABC 契约) 删除。
- `diff_facade` 的 `created` 相等快跳过: 存疑保留 (假定运行时不会原地改 meta)。

## 悬置 (cut scope, 已记录)

- hot 逻辑 (降级三态机、warm/hot 分类)。
- 裸事件 drain (interleaved thinking 中流)。
- 装线 (articulate 循环的 now 注入、MCP 接线)。
- CTML 判词统一 (moss_static / moss_dynamic 旧词需与轨迹叙事对齐)。

## Implementation Notes

- 事件 index 记账: `_append_event` 独占计数器, 回调不碰游标。
- `on_channel_metas_generation` 回调 + discard 句柄 (set 存回调)。
- `facade_body` 的 states 必须走 `state_text()` (str); 直接 `states_message()` (Message)
  会 join TypeError。
