---
created: 2026-07-20
depends: []
description: GhostRuntime 的极简安全模式：开启后 ghost 生成的 logos 不直接进 action 执行， 经 TUI 用户通过/否决后放行或回流反馈。范围有意收窄在
  articulate→action 链路。
milestone: null
priority: P2
status: in-progress
status_note: gate landed at e60b4fda; round 2 = decisions 11-13 (turn-based UX + introspection
  channel)
title: Ghost Runtime Safemode — ghost 生成 logos 的人工审批闸口
updated: '2026-07-26'
---

# Ghost Runtime Safemode

> Use `moss features set-status ghost-runtime-safemode <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

Ghost 在真实环境执行 CTML 命令前缺一个人工审批闸口。开启安全模式后，ghost 生成的
logos 先 buffer 等待 TUI 用户裁决，通过才进入 action 执行，否决则作为反馈回流给
ghost 下一帧。有意保持极简：这只是 GhostRuntime 的一个局部治理功能，不是系统级
安全体系，命名上也不扩大化（不用 "safemode" 这种全局词）。

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

1. **范围：只闸 articulator 生成的 logos，不闸输入。** 输入侧通断是 pause 的职责，
   两套机制各管各的。`moment.command_logos`（impulse 反射弧，如 FATAL 急停）必须
   绕行 gate——审批卡住急停不可接受。

2. **四动作：开启 / 关闭 / 通过 / 否决。** 开关只影响下一轮 articulation 的模式
   （生成开始时判定一次），不动在途逻辑；已挂起的 pending 继续等人批完。

3. **门控对象 = 每次审批一个 gate。** 基于 `concurrent.futures.Future`：TUI 线程
   `set_result`，async 侧 `asyncio.wrap_future` await。三态 verdict：
   approved / rejected(reason) / cancelled。

4. **abort 感知走事件驱动，不用轮询。** 等待注册为 `articulator.create_task(...)`，
   abort 时 ctx 看门狗 task 抛错 → task group 级联取消等待 task。需给 `Articulator`
   ABC 补 `is_aborted()`（`Action` 已有同款，对称补齐）作为 CancelledError 消歧键
   （区分 abort 取消 vs 真·shutdown 取消）。`_run_articulator` finally 里幂等
   `gate.cancel()`。（曾提 0.1s 轮询方案，因事件驱动路径现成而收回。）

5. **否决反馈用 observe，不用 abort。** abort 杀死整个 attention，ghost 要等下一个
   外部 impulse 才感知否决，"否决作为 logos 返回值"链断裂。正确路径：
   `articulator.raise_observe(...)` → `attention._loop` 开下一帧 → 否决消息随
   Reaction 进 moment → ghost 重新 articulate（再被闸）。

6. **空流 bug 必须先修（阻塞项）。** `BaseAction._logos` 预取 None 的 early-return
   是不可达死代码（base_attention.py:348-351，外层 `is not None` 挡住预取的 None），
   空 articulator 时 action 挂死在 1s 轮询直到外部 abort。否决路径必然空流，必然
   踩中。现网不炸只因 effort-none 的 articulator 通常先发 command_logos。先补单测
   （空 articulator → received_logos 立即返回、零 yield），再修（prefetch 加独立
   标志位）。

7. **TUI 交互按频率分层。** 开/关走 TUI 层 `/safe` 翻转命令（default_commands，
   享 /exit 待遇）+ inspector `/safe.on()` 等精确接口；通过/否决走条件快捷键
   `c-y` / `c-d`（ConditionalKeyBindings，仅 pending 存在时激活，无 pending 时
   默认编辑行为不受影响）。键位依据误触代价不对称分配：误批准是最坏结果，approve
   配最少肌肉记忆的 c-y（yank）；c-a（行首反射键）因此被否。

8. **uuid 对齐防批错帧。** 按键提交 toolbar 展示中的 uuid，gate 内比对 current
   pending，stale 则响亮 no-op，绝不自动顺延到下一帧。残差竞态（渲染延迟）降级为
   白按一下 + 提示。

9. **TUI 通知无新基建：一读一回调。** 审批内容（logos 面板 + uuid）走现有
   `session.output` → `_on_session_output` 渲染通道；toolbar 走
   `on_pending_changed` 回调 + `app.invalidate()`（镜像 pause callback 契约）；
   按键现读 current pending。不引入队列 / ThreadSafeEvent。

10. **无历史记录。** 初版设计有审批历史（通过/否决/取消都进历史），讨论后砍掉，
    保持极简。

## Implementation Notes

- 拦截点：`GhostRuntimeImpl._run_articulator`（ghost_runtime.py:330 附近）。开启时
  delta 走 buffer 而非 `articulator.send_nowait`；`session.pub_logos` 照常——用户
  实时阅读生成过程，这正是裁决的依据。通过后回放 buffer，articulator context 仍开，
  流未关（None 哨兵只在 `__aexit__` 放）。
- SafeMode 管理器放 `ghoshell_moss/host/` 与 PauseController 同级，GhostRuntimeImpl
  持有。
- 审批等待期间 articulate loop 串行阻塞 → 任意时刻单 pending → 热路径无需 uuid
  选择。
- prompt 状态行加 `[SAFE]`，镜像现有 `[PAUSED]`。

## Post-implementation Optimization (发现于 e60b4fda 之后)

`[SAFE]` prompt prefix 让 shell 语义视觉翻转 —— 实现后才看清的信号反过来撬动
三条简化，作为 SafeMode 的连带产物记录，等实现启动时再落地。

11. **回合制 via prompt，废快捷键。** SafeMode 有 pending 时，prompt line
    重定向到审批：空 enter = approve，输入文本 = reject with reason。placeholder
    显示 `<enter: approve · type: reject with reason>` 防误触。这样拆掉 c-y/c-d
    ConditionalKeyBindings、`_safe_mode_wired`、toolbar 刷新链；reject reason 从
    hardcoded `"rejected by user via c-d"` 升为用户真实内容；决策 7 的键位人机
    工学争论（c-a vs c-y）整个消失。pending 期间新 input signal 强制先处理
    pending，对齐单串行 pending 语义。原触发洞察：`[SAFE]` prefix 已把 shell
    语义翻转，prompt line 的默认行为 *不应* 保留为普通信号 —— 视觉信号 + 语义
    翻转应一致。

12. **Approve-with-note 走 attention 内观通道，不动 Articulator ABC。**
    Articulator 是纯流式推理输出，加 outcome 是概念泄漏。approve 若带 note，
    走与 reject 同一条 attention observation 通道（`raise_observe` 已在此空间），
    只是不 raise、不抢占下一帧。若目前只有 raise 变体，加一个 non-raising
    observation attach。

13. **内观 vs 外视 —— 概念锚点。** `Action.outcome` 是外视（command 执行后
    世界的回应，与工具调用同步，交错 thinking 时序拥挤发生在此通道）；attention
    observation 是内观（帧边界事件，articulate 结束→下一帧开始之间发生）。
    SafeMode 裁决属于内观，天然不与 tool outcome 抢时序。两条通道**不合并**。
    内观 message 靠自身封装结构（xml 分段等）自解释来源，**不加 source 字段** ——
    抽象负担留白，未来加入异步人工评论、self-reflection、后台 critic agent 等
    更多内观来源时，各自消息体自解释即可。

## Round 2 Landed

- **输入协议**：pending 时 `""` = approve；`"!<text>"` = approve-with-note；
  `"<text>"` = reject with reason；`"/<cmd>"` bypass 让默认命令生效。默认 = reject
  是刻意 —— 误发文本时不会误批准（approve 是更坏后果）。
- **12 的实现修正**：raise_observe 混进"logos 需完整执行"路径是脆弱的（依赖
  `capture_error` swallow 顺序 + `put_nowait(None)` 保序，一旦 `__aexit__` 语义
  微调就崩）。给 `Articulator` ABC 加了非 raise 的 `observe(message)`，`BaseArticulator`
  委托 `AttentionContext.observe`。approve-with-note 和 reject 都改用它，两条路径
  对称。这即是决策 12 里预留的 fallback，最终采纳。
- **13 的执行**：logos 用 `<safemode-approval-note>` / `<safemode-rejection>` xml
  包裹后进内观通道，不加 source 字段。
- **TUI 基类改造**：新增 `_pre_handle_input(item) -> bool` 和 `_get_input_placeholder()`
  两个钩子（`host/tui.py`），子类覆盖即可劫持输入 / 挂 placeholder。GhostTUI 只写
  ~30 行就消化掉审批交互。
- **拆除**：c-y/c-d ConditionalKeyBindings、`_safe_mode_wired` 首次注册回调、
  toolbar `[SAFE pending ...]` 刷新链、`_on_safe_pending_changed`、`_safe_approve`
  / `_safe_reject`、hardcoded `"rejected by user via c-d"` 字符串。决策 7 完全废弃。
- **`Verdict.reason` → `Verdict.message`**：approve-with-note 和 reject-with-reason
  语义上共享 message 字段，字段名跟随。

## Round 2 顺便修的 Round 1 遗留

用户手动测试 round 2 时发现 gate 表面在工作 (`[SAFE]` prefix、`/safe` 都对)，但
ghost 响应照常输出，pending 从未触发 `_pre_handle_input`。日志里露出真相：

```
articulate error: a coroutine was expected, got <Future pending>
File "ghost_runtime.py", line 393, in _run_articulator
    verdict = await articulator.create_task(
File "base_attention.py", line 249, in create_task
    task = self._event_loop.create_task(cor)
TypeError: a coroutine was expected, got <Future pending>
```

**Bug**：round 1 (e60b4fda) 写成 `articulator.create_task(asyncio.wrap_future(fut))`。
`wrap_future` 返回 `Future` 不是 `coroutine`，uvloop 的 `create_task` 严格拒收。
gate 每次 activate 立即抛 TypeError → 被 `_run_articulator` 的 `except Exception`
静默吞成 log 一行 → `finally` 里 `cancel_current` 把 pending 结算为 cancelled →
verdict.kind == 'cancelled' 什么都不做 → `send_nowait` 从未执行，action pipeline
从未跑；但 `pub_logos` 每 delta 都发了，TUI 看到 raw stream 以为"正常工作"。

**修法**：用 async 函数包一层再喂 create_task：

```python
async def _await_gate_verdict():
    return await asyncio.wrap_future(verdict_future)

verdict = await articulator.create_task(_await_gate_verdict())
```

**Round 1 潜伏几周的根因是测试 gap**：`test_safe_mode.py` 只测 `SafeModeImpl` 自身
状态转移（submit / approve / reject / cancel），完全没覆盖 gate 与 articulator
的实际 await 路径。补 `test_safe_mode_gate_integration.py` 三个用例，直接跑
`await articulator.create_task(_await_verdict())` 这个组合，任何等价 bug 复现
就 TypeError 失败。

**次生教训**：`_run_articulator` 的 `except Exception as e: logger.exception(...);
session.output('error', ...)` 静默模式在这里咬了自己。虽然 error 有进 session
output，但 TUI 默认停在 logos state 看不到；日志用户不主动 grep 也不知道。
下一次遇到"表面正常但功能未生效"，第一动作就应该是 `tail -f .moss/runtime/logs/moss.log`。
架构上是否要把 articulate error 也 push 到 logos state 显眼位置，值得单开讨论。