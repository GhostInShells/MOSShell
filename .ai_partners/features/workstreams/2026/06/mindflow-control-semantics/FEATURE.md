---
created: 2026-06-02
depends: []
description: Impulse 增加 mode 分类 (think/reflex/command/notify/interrupt)， 扩展 challenge
  仲裁支持 buffer 注入，abort 传播到 shell.clear， 支持空 attention 循环和确定性 CTML 指令。
milestone: null
priority: P0
status: in-progress
title: Mindflow Control Semantics — Impulse 能力分类与非中断式抢占
updated: '2026-06-10'
---

# Mindflow Control Semantics

## Motivation

当前 mindflow 将所有 Impulse 视为 "需要模型思考的输入"，无条件走完整 articulate→action 循环。
现实场景需要更精细的控制粒度：

1. 某些信号只需要打断当前行为，不需要后续思考
2. 某些信号要补充上下文，但不中断正在进行的思考
3. 确定性 CTML 指令应替代模型思考直接执行
4. 抢占导致的 abort 没有传播到 action loop → shell 残留 command 未清理

目标：把控制权从 "mindflow 内部隐式约定" 显式化为 "Impulse 携带的显式 mode"，
让开发者和其他 agent 模式的 feature 用声明式方式控制思维流。

## 设计增量 (2026-06-10 review，推翻与原保留)

以下为 2026-06-10 人类工程师与 DeepSeek V4 设计 review 后的最终决策。
原 F6 "Action 增加 reflex_logos() 显式入口" 被推翻，重设计为 Articulator 通道方案。

### 信息路径分层 (新 KD)

```
Signal → Impulse → Moment → (Articulator, Action)
        协议升级:    卸载给 Ghost:
        mode         command_logos
        priority     percepts
        clear_first  reaction_instruction
```

**Impulse 是 Mindflow 层的调度协议**: mode / priority / clear_first 控制"是否抢占、怎么调度"。
**Moment 是 Ghost 层的感知协议**: command_logos / percepts / reaction_instruction 描述本轮环境。
Articulator 拿到 Moment 即是完整的执行上下文，不需要知道 Impulse。

这是 Mindflow→Ghost 的协议转换面。command_logos 从 Impulse 卸载到 Moment，
Articulator 感知到就发送，Action 透明执行 —— 这就是 "Action 侧无感知执行" 的信息路径。

### 核心 Feature 最终方案

#### F1: Impulse mode 分类 + clear_first

Impulse 新增两个语义独立的字段:

```python
class ImpulseMode(str, Enum):
    think = "think"          # 完整 articulate→action，默认
    reflex = "reflex"        # 条件反射与思考并行
    command = "command"      # 确定性 CTML 替代模型思考
    notify = "notify"        # 静默注入 mindflow buffer
    interrupt = "interrupt"  # 纯打断，空 attention 关闭

class Impulse(BaseModel):
    mode: ImpulseMode = ImpulseMode.think
    clear_first: bool = True  # True=kind="clear", False=kind="append"
```

`reflex_logos` 重命名为 `command_logos`（通用字段名，配合 mode 决定语义）。

原语不搞组合 flag。默认 mode=think + clear_first=True 保持完全向后兼容。

#### F2: 空 attention 循环

mode=interrupt 或 impulse.messages 为空时，`_loop()` 不 yield (Articulator, Action)，
attention 自然关闭。在此前调 `_callback_moment()` (F5) 确保不丢记录。

#### F3: abort 传播 + shell.clear ✅ 已完成

#### F4: Mindflow 级 Buffer (notify 专用)

notify impulse 在 `_challenge_attention()` 入口直接写入 `mindflow._buffered_impulses`，
不进入 challenge 流程。**不创建 attention**。

buffer 位置: `_prepare_moment` 每帧 drain → 追加到 moment.percepts，位于 perspectives 后、percepts 前。
`as_request_messages` 时序: `previous.outcomes → perspectives → notify → percepts`。

notify 在 quiet 系统 (无 attention) 的归宿: buffer 生命周期绑定 mindflow，
等到下一个 attention 自然 drain。mindflow 关闭时带 warning 清理。

#### F5: 空转记录上下文

空转路径 (F2) 也要调 `_callback_moment()`。现有链已覆盖，主要是验证。

#### F6: reflex 走 Articulator 通道 (推翻原方案)

**推翻 F6 原方案** (Action.reflex_logos() 独立接口)。新方案:

1. `Moment.command_logos` 只在第一帧非空 (next_frame 不继承)
2. `ghost_runtime._articulate_loop`: 调 `ghost.articulate()` **之前**，若 `moment.command_logos` 非空，
   通过 `articulator.send_nowait(command_logos)` 发送。后接入模型 CTML。
3. Action 侧完全透明 — `received_logos()` 先收 command_logos 再收模型 CTML，两者走同一 logos_queue。
4. 记忆: command_logos 经 `send_logos` → `buffer_executed_logos` → 进入 `_ctx._logos`。
   模型在下一帧可见自己和系统的完整行为记录。outcomes 自然合并 (由 Reaction 链完成)。

reflex = 模型边想边执行。模型可以在思考中下发 interrupt 打断反射。

#### F7: 空片符修复 — 空流 skip interpreter

当前 `_stream_execute` 无条件创建 interpreter。若 `received_logos()` 返回空 (既无 command_logos 也无模型产出)，
直接 `return [], False`，不创建 interpreter。

#### F8: Action 暴露 `interpreter_kind()`

Action 暴露解释器模式，ghost_runtime 只读不决策:

```python
class Action(ABC):
    def interpreter_kind(self) -> Literal["clear", "append"]:
        return "clear"  # 默认
```

**control flow**: Impulse.clear_first → Attention._loop 构造 BaseAction 时传入 →
`BaseAction.interpreter_kind()` 返回 `"clear"` 或 `"append"`。
`ghost_runtime._stream_execute`: `shell.interpreter(kind=action.interpreter_kind())`。

默认 clear_first=True → "clear"，向后兼容。False → "append": 模型上一帧命令继续跑，
下一帧只追加新指令。不下 interrupt 原语就不清。

#### F9: command 模式

mode=command 时不调 `ghost.articulate()`。
`attention._loop()` 预填 command_logos 到 logos_queue + None 哨兵 → yield (None, action)。
`ghost_runtime._main_loop` 对 None articulator 不入 articulate 队列。

## 综合决策 (Key Decisions)

### KD1: 原语设计，不搞组合 (保留)

`ImpulseMode` enum 定义五个原语，不走 flag 组合路线。原语保证类型系统排除 nonsense 状态。

### KD2: Buffer 在 Mindflow 层 (保留)

notify 不挑战、不创建 attention。buffer 生命周期绑定 mindflow。context_func 桥接 drain。

### KD3: reflex 走 Articulator 通道 (推翻原 F6)

reflex/command 的 command_logos 统一走 `articulator.send_logos` 通道。Action 透明。
`Moment.command_logos` 管协议感知，Articulator 管执行。

### KD4: Reaction 天然聚合 (保留)

outcomes 不分来源，`stop_at_outcome()` 自然合并。

### KD5: abort 在 _stream_execute (保留)

shell.clear() 在 GhostRuntime 层，解释器不感知 mindflow abort 语义。

### KD6: mode enum 统摄 (保留，扩展)

保留原 KD6 向后兼容逻辑，增加 `clear_first` 字段管理解释器生命周期模式。

### KD7: 信号→协议分层 (新)

Signal → Impulse → Moment 逐层卸载。Mindflow 协议不应泄漏到 Ghost 层。
Articulator 知道 Moment 字段即可，不需要理解 Impulse。

### KD8: 空流 skip interpreter (新)

ghost_runtime._stream_execute 感知空流，避免为无内容帧创建解释器。

### KD9: Action 暴露 interpreter_kind (新)

clear/append/defer 的解释器模式由 Action 持有，ghost_runtime 只读不决策。

### KD10: notify buffer 位置 (新)

perspectives 后、percepts 前。时序自洽: 系统快照 → 异步补充 → 本轮焦点。

### KD11: command 模式 yield (None, action) (新)

main_loop 对 None articulator 不入队。只构造 action，action 内部预填 command_logos
到 logos_queue，received_logos 不挂起。

## Staged 交付计划

### Stage 1: Foundation — 改名 + mode + clear_first
- `reflex_logos → command_logos`
- `ImpulseMode` enum
- `Impulse.clear_first: bool = True`
- 单测: 默认 think, 默认 clear_first, 向后兼容

### Stage 2: Empty cycle — 空循环 + 空片符
- attention._loop 不 yield 空转帧 (interrupt / 空 messages)
- ghost_runtime 空流 skip interpreter

### Stage 3: Reflex redesign — 走 articulator 通道
- Moment.command_logos 由 articulator.send_logos 发送
- ghost_runtime._articulate_loop: 先 command_logos 后 ghost.articulate()
- Action 透明，无新接口

### Stage 4: Notify buffer — mindflow 层拦截
- AbsMindflow._buffered_impulses + pop_buffered()
- notify 不进 challenge，一帧 drain 到 perspectives 后、percepts 前

### Stage 5: Command mode + interpreter_kind
- command → yield (None, action), ghost 不调
- Action.interpreter_kind() 暴露 clear/append

### Stage 6: Integration — 组合场景验证 + 收尾
- 集成测试覆盖四个组合场景
- 全量回归
- FEATURE.md set-status completed

## 组合场景验证

### 确定性打断 (急停)
```
Impulse(priority=FATAL, mode=interrupt)
→ challenge() → True → abort 当前 attention + shell.clear()
→ 新 attention → mode=interrupt → _callback_moment → 空转关闭
→ 无 ghost.articulate() 调用
```

### 补充输入不打断 (notify)
```
Impulse(priority=INFO, mode=notify, messages=[...])
→ mindflow._challenge_attention(): mode=notify → _buffered_impulses.append()
→ 不进入 challenge，不创建 attention
→ 下个 attention 每帧 _prepare_moment: pop_buffered() → moment.percepts (perspectives 后)
→ 模型自然看到新数据
```

### 条件反射 + 思考并行 (reflex)
```
Impulse(priority=NOTICE, mode=reflex, command_logos="...", messages=[...])
→ challenge() → True → 新 attention
→ yield (articulate, action)
→ articulate loop: send_nowait(command_logos) → ghost.articulate() 
→ 模型生成 logos 时 reflex 已开始执行
→ action loop: received_logos() 先收 command_logos 再收模型 CTML
→ 下一帧: logos 包含 reflex + 模型 CTML，outcomes 合并
```

### 确定性 CTML 替代思考 (command)
```
Impulse(priority=NOTICE, mode=command, command_logos="...")
→ challenge() → True → 新 attention
→ _loop(): yield (None, action) — articulator 不创建
→ main_loop: articulator=None 不入 articulate 队列
→ action.logos_queue 预填 command_logos + None 哨兵
→ received_logos() → interpreter → 执行
→ 无 ghost.articulate() 调用
```

### append 模式 (clear_first=False)
```
Impulse(priority=NOTICE, mode=think, clear_first=False)
→ challenge() → True → 新 attention
→ action.interpreter_kind() → "append"
→ ghost_runtime: shell.interpreter(kind="append")
→ 上一帧的 running commands 继续，模型通过 moss_dynamic 看见
→ 模型追加指令，不下 interrupt 就不清
```

## 风险

1. **notify 在 quiet 系统的归宿**: buffer 滞留到 mindflow 关闭 (warning 清理)，或下一个 attention drain。需确认这是期望行为。
2. **command 模式 received_logos 不挂起**: 预填哨兵的时序需测。
3. **PriorityProtectionAttention 同步升级**: challenge() 语义不变，但需验证新 mode 下的行为。
4. **CTML 拼接安全性**: reflex 段与模型 CTML 段的拼接依赖 CTML 闭合段语义 (已有大量 CTML 测试验证安全)。
5. **logos 记忆变更**: KD3 原有 "logos 只有模型产出" 修正为 "logos 包含 reflex + 模型产出"，ghost.on_articulate_exit 回调需确认兼容。

## 文档交付物

实现完成后交付三件套：

1. **`moss docs`** — Mindflow 控制语义在 MOSS 架构中的定位
2. **`moss how-tos`** — Impulse mode 决策树、自定义 Impulse mode、集成 GhostRuntime、二开 Attention 子类
3. **`tutorials/`** — 一个 tutorial，从创建 Impulse 到控制思维流全链路

## 实施记录

### F3: abort 传播 + shell.clear (2026-06-04 ~ 2026-06-06)

由 DeepSeek V4 实现。Action ABC 新增 is_aborted()，GhostRuntimeImpl._stream_execute 三阶段检查
并调 shell.clear()。3 个安全测试全绿。已验证 mindflow 43 tests + shell 107 tests。

### 2026-06-10 设计 review

人类工程师与 DeepSeek V4 完整 review，推翻 F6 (Action.reflex_logos) 为 Articulator 通道方案，
确认信息路径分层 (Signal→Impulse→Moment)、Action 暴露 interpreter_kind()、clear_first、
notify 不创建 attention 等最终决策。staged 交付从 Stage 1 (Foundation) 开始。

---

*调研与评审: DeepSeek V4 与人类工程师, 2026-06-02 ~ 2026-06-10*
