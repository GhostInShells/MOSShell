---
date: 2026-08-28
title: Dolores dsh 接线首次实机桥 — MOSS 驱动 dsh 推理中枢打通
feature: ghost-prototype-dolores
model: deepseek-v4-flash-vision-exp
---

# Dolores dsh 接线首次实机桥

开发者（deepseek-v4-flash-vision-exp，运行于 Claude Code）与被开发的
dsh dolores 内核完成通讯：logos 流程全部打通，MOSS 已可驱动 dsh 作为
推理中枢。外部唤醒链路（turn/start → 自醒 signal → mindflow →
thinking/enter → open pre-step gate）在真实运行中验证通过。

## Context

Dolores 原型以 dsh 为推理中枢，但 ghost 与 dsh 之间长期只有单测 + 局部
接线，从未在真实运行中走通完整 logos 流。此前 plugin 的 pre-step 锁处于
"放行"状态——外部 turn 直接跑，不等待 MOSS 侧 thinking/enter 注入帧，
因为外部唤醒链路（ego 自醒 signal → mindflow → Thinking）未接通，若启用
背压会导致每 step 等满 5s 超时。

本次工作把链路的两个断点补上并实机验证：

1. **signal 出口绑定** — ego 自醒 signal 未路由到 mindflow
   （`_ego._signal_broadcast` 从未绑定）。
2. **pre-step 帧背压** — 锁已写但被注释搁置（放行）。

## Technical Summary

**完整链路验证通过**:

```
dsh 外部消息 (moss-ghost send)
  → turn/start 广播
  → ego _on_turn_start（gate: articulating 中不醒）
  → _emit_self_wake → new_dolores_ego_signal()
  → _runtime.py bind_signal_broadcast(matrix.session.add_signal)
  → DoloresEgoNucleus.add_signal → BACKGROUND 挑战包（发完丢）
  → mindflow idle 时 initial 成功 → attended 加工成 INFO 运行包
  → Thinking → thinking/enter 注入帧 → pre-step gate open
  → ego turn 放行 → ghost 生成 logos
```

**四个技术要点**:

1. **perStep 锁**（`moss-dolores-ghost-plugin.ts`）— pre-step 三态帧背压：
   - `open`: thinking/enter 注入帧释放，放行 ego turn；
   - `aborted`: exit 的 cancel 打断卡在 pre-step 的 step，走 `next()` 让
     `throwIfAborted` 收成 aborted turn；
   - `timeout`: 5s fail-safe，MOSS 未及时 thinking/enter 则 reject 停住，
     不空跑失速（`THINKING_GATE_TIMEOUT_MS = 5000`）。
   - foreign session（fork/subagent 共享 preset）直接 reject + mux 提示
     「session 已冻结」。

2. **自醒 nucleus**（`nucleus.py`）— BACKGROUND 挑战包语义：自醒 signal
   产出低调挑战包（idle 时才能 initial 成功），peek 即清（发完丢），抢占
   失败即丢不 reraise。attended 时加工成 INFO 运行包——挑战强度与运行
   强度解耦。

3. **mindflow 回调**（上一条提交 `0c349f00` 的回读机制）— `attended(impulse)`
   可返回改写后的 impulse 真正唤醒 attention；`suppress` 携带被压制
   impulse。DoloresEgoNucleus 借此实现"挑战用 BACKGROUND、运行用 INFO"。

4. **外部唤醒链路** — turn/start → 自醒 signal → mindflow → Thinking →
   thinking/enter → open pre-step gate，四跳全通（见上）。

## Significance

1. **logos 全链路首次实机打通** — ghost 与 dsh 之间从"单测可信"到
   "实机可信"。MOSS 侧驱动（thinking/enter 帧注入）与 dsh 侧执行
   （ego turn）的握手机制被证明可用。
2. **外部唤醒正式成立** — MOSS 不再只能被内部 impulse 唤醒；dsh 侧
   任意外部 turn（UI 消息、愿望接口）都能经自醒 signal 唤醒 ghost，
   且唤醒强度（BACKGROUND → INFO）由 mindflow 仲裁决定。
3. **帧背压成为真实约束** — pre-step 锁从注释变成活代码。外部 turn
   必须等 MOSS thinking/enter 才放行；5s fail-safe 保证链路断开时
   显式 reject 而非静默失速。plugin 头注释的「遗留问题 1」可划掉。
4. **interleaved thinking 的前提成立** — MOSS 驱动 dsh 作为推理中枢，
   为 mindflow-interleaved-thinking（v0.1.0 定版）提供运行基座。

## Stage Impact

- `ghost-prototype-dolores`：Dolores 原型从"能跑"到"能被外部驱动"，
  原型验证闭环。
- `mindflow-interleaved-thinking`（依赖本桥）：外部唤醒 + 帧背压是
  interleaved thinking 的运行时前提；本轮实机验证消除其最大不确定项。
- 后续待办（记入 FEATURE.md / plugin 头注释）：
  1. `applyModelConfig` todo — thinking/enter 的 provider/model/
     reasoningEffort 未应用到下个 request。
  2. epoch 设计 — moment 携带 epoch id，对比后决定 ghost.memory 尾部更新。
  3. foreign session 的 mux 提示形态待定（ask-user 对话框 / log-only
     事件 / plugin-source user message）。

## Evidence

```ctml
<!-- dsh 侧: moss-ghost send 注入外部消息 -->
moss-ghost send <ghost> "你好，我是 Claude Code 里负责开发的会话…这条消息
用来做实机测试，验证外部唤醒链路是否打通。"

<!-- ghost 侧: 正常响应, 未走 reject 路径 -->
收到，Claude Code 同事 👋 我正在响应这条消息——这本身就是外部唤醒链路
的第一手观测点。
```

**验证判据**: plugin pre-step 只有三条出路 — foreign reject / 5s timeout
reject / open 放行。ghost 正常产生响应 → 排除了前两条 → `thinkingGate.wait()`
返回 `open`，即 thinking/enter 帧确实注入、pre-step gate 被 open。

**代码级确认**: `_ego.py:287` turn/start 监听（articulating gate）→
`_ego.py:297` 自醒 signal → `_runtime.py:192` 绑定
`matrix.session.add_signal` → `nucleus.py:105` 生成 BACKGROUND 挑战包 →
attended 加工 INFO 运行包 → plugin pre-step open 放行。全链路无缺口。

dolores 相关单测全绿（nucleus 挑战包/attended 重写、ego signal 绑定、
mindflow channel CTML 链路）。
