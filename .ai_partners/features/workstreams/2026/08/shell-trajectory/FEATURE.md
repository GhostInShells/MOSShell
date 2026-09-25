---
created: 2026-08-19
depends: []
description: 'ShellTrajectory 取代旧观测面 (ContextMonitor / ShellContext / InterleavedThinkingToolset),
  以 pull 型帧轨迹承载上下文缓存经济学下的观测: 帧 = events + facade delta + dynamic messages.'
milestone: null
priority: P1
status: completed
status_note: ShellTrajectory 取代旧观测面; KD8 InterpreterStoppedEvent 展示定稿; 32 测试绿
title: Shell Trajectory — 观测轨迹取代上下文监控
updated: '2026-09-25'
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

4. **时间戳分层 + D 模式统一**。channel facade 块不带时间戳 (durable 需字节稳定, 且
   批量 refresh 下大面积重复); 帧内所有时间戳 (moss/command/interpreter) 统一 `at` 属性 +
   `D19 00:01:17+8` 短格式 (日 + 时分秒 + 时区)。年/月由 full_facade 的 `<today>` 锚
   承担 (epoch 起点注入一次, 帧内不重复)。

5. **now 语义**。frame `at=` = 发送时刻 (now), message 时间戳 = 事件发生时 (冻结事实)。
   请求重试必须重新 `project(now=新时间)`, 否则模型误把上次发送时刻当 now。

6. **ack 纪律**。peek 非破坏, commit 才 drain + 推进 baseline。只有消费方能判断消费
   成功 — 模型请求失败不能丢事件 (at-least-once)。

7. **interpreter stop 结算投影**。`InterpreterStoppedEvent` 与 `ShellTaskDoneEvent` 互补:
   前者带执行结算 (completed/cancelled/failed/error), 后者带命令返回值。无返回值命令
   (如 speech) 的 `ShellTaskDoneEvent` 为空, 若再丢结算, 模型只剩 bare `<status idle/>`
   无法感知动作是否执行。规则内聚在 event: `as_messages` 空结算返回空列表, 帧遍历处
   统一 `extend`, 无投影价值的事件自然不产出帧。

8. **InterpreterStoppedEvent 展示定稿 (2026-09-24)**: 结算事件渲染改为 `logos` 标签 + state 裸词,
   去掉 `at` (帧头 `<moss at=>` 已锚时刻), 计数去冒号, cancelled/failed 带末尾 task 身份
   (`caller_name()`), 形如 `<logos interrupted>\ncancelled 3, last chan:say\n</logos>`.
   - **交接不展示**: append 交接 (`close(cancel_executing=False)`) 结算时计数全空、仅 `pending`
     非零, 保持静默是对的 —— 模型不感知"交接"这个内部动作, 它感知的是连续签发 ctml 流. `pending`
     非零只在交接出现, 所以不展示 pending 即不展示交接.
   - `completed` 保留纯计数 (无值命令若不 raise observe, 计数是它唯一的痕迹); `error` 保留原文.
   - 撤销: 2026-09-23 那版"观测点位"作废 (它把 command result 误读成要回显模型的 ctml tokens).

## 取代与撤销 (dead ends)

- 两个旧 workstream **删除**: `channel-meta-dyn-static` 与 `context-cache-engineering`
  (同一命题的演化, 从未 completed), 完整历史见 git log。
- ContextMonitor (`host/context_monitor.py`, 33 tests) 无法对齐目标 → 取代。
- InterleavedThinkingToolset (`host/interleaved_thinking.py`) 非 dead end — 当初正确交付
  (turn-based 观测 + 事件投影), 被 ShellTrajectory 吸收延续 (pull 型帧轨迹是其后继)。
- `shell_context.py` (ShellContext ABC 契约) 删除。
- `diff_facade` 的 `created` 相等快跳过: 存疑保留 (假定运行时不会原地改 meta)。

## 悬置 (cut scope, 已记录)

- hot 逻辑 (降级三态机、warm/hot 分类)。
- 裸事件 drain (interleaved thinking 中流)。
- articulate 循环的 now 注入 (MCP 接线已做: moss_instruction(full_facade) / full_facade /
  get_channel_facade / moss_observe / ctml_append/exec/replan/interrupt)。

## Implementation Notes

- 事件 index 记账: `_append_event` 独占计数器, 回调不碰游标。
- **帧序号 ≠ 事件水位** (2026-09-23 修复): `ack 纪律` 的「推进 baseline」由 `ShellKeyFrame.index`
  驱动 — 帧序号在 `commit` 时递增, 每帧无条件推进 diff 基线; `tracer_index` 只是抓帧瞬间的
  事件水位, 仅供 `tracer.drain`。二者早期被混用 (帧号误接成事件 index), 导致纯 meta 变化
  (如 channel `available` 翻转) 不产生命令事件时, `commit` 因 index 未 +1 而 no-op、基线不前进,
  于是「移除→出现」的 delta 链条里, 恢复帧 diff 不到变化被吞掉。
- `on_channel_metas_generation` 回调 + discard 句柄 (set 存回调)。
- `facade_body` 的 states 必须走 `state_text()` (str); 直接 `states_message()` (Message)
  会 join TypeError。
