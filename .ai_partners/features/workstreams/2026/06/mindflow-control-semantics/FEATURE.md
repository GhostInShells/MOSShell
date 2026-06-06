---
created: 2026-06-02
depends: []
description: Impulse 增加 mode 分类 (think/reflex/command/notify/interrupt)， 扩展 challenge
  仲裁支持 buffer 注入，abort 传播到 shell.clear， 支持空 attention 循环和确定性 CTML 指令。
milestone: null
priority: P0
status: in-progress
title: Mindflow Control Semantics — Impulse 能力分类与非中断式抢占
updated: '2026-06-07'
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

## Design Index

- Mindflow blueprint: `ghoshell_moss.core.blueprint.mindflow`
- AbsAttention + BaseAttention: `ghoshell_moss.core.mindflow.base_attention`
- PriorityProtectionAttention: `ghoshell_moss.core.mindflow.priority_attention`
- BaseMindflow / PriorityMindflow: `ghoshell_moss.core.mindflow.base_mindflow`
- GhostRuntime 集成: `ghoshell_moss.host.ghost_runtime.py`
- 单测: `tests/ghoshell_moss/core/mindflow/`

## 六个 Feature 概述

### F1: Impulse mode 分类 (ImpulseMode enum)

Impulse 增加 `mode: ImpulseMode` 字段，定义五个原语: `think` / `reflex` / `command` / `notify` / `interrupt`。

不搞组合 flag。原语保证无非法状态，类型安全。

`reflex_logos` 重命名为 `command_logos`（通用），配合 mode 决定行为：
- mode=reflex: `command_logos` 作为条件反射并行执行
- mode=command: `command_logos` 替代本轮思考

### F2: 空 attention 循环

mode=interrupt 或无 messages 时，`_loop()` 不 yield (Articulator, Action)，
attention 自然关闭。`_main_loop` 的 async for 空循环自动跳过。

### F3: abort 传播到 action loop + shell.clear

`_stream_execute()` 在 feed/compile/execute 各阶段检查 abort，
发现后调 `shell.clear()` 取消 pending CommandTask。

### F4: Mindflow 级 Buffer 机制

**不在 attention 层，在 mindflow 层**。

notify impulse 在 `_challenge_attention()` 入口直接写 `mindflow._buffered_impulses`，
不进入 challenge 流程。Attention 通过 context_func 桥接 `pop_buffered_impulses()`，
每帧 `_prepare_moment()` 时 drain → moment.percepts。

### F5: 空转记录上下文

空转路径 (F2) 也要调 `_callback_moment()`，确保 moment 不丢失。
现有 `on_moment` 回调 + `last_outcome()` → `stop_at_outcome()` 链已覆盖，主要是验证。

### F6: Action 增加 reflex_logos() 显式入口

Action ABC 新增 `reflex_logos() -> str` 方法（不走 logos_queue，与 received_logos() 通道分离）。
`_stream_execute()` 先消费 reflex，再消费模型 logos。模型 percepts 中预先知道 reflex 正在执行。

## Key Decisions

### KD1: 原语设计，不搞组合

`ImpulseMode` enum 定义五个原语，不走 flag 组合路线。

**为什么不是组合**: `skip_articulate` / `buffer_data` / `interrupt_current` 等 bool flag 之间存在隐含的排斥和依赖关系（例：`buffer_data=True` + `interrupt_current=True` 无意义），但类型系统不帮你检查。组合把 16 种状态中 5 种合法的验证交给使用者；原语直接保证不可能构造出 nonsense 状态。

**五个原语**:

| Mode | 行为 | 模型参与 | 场景 |
|---|---|---|---|
| `think` (默认) | 完整 articulate→action | 是 | 正常输入 |
| `reflex` | 条件反射与思考并行，模型感知 reflex 正在执行 | 是（并行思考） | 条件反射 |
| `command` | 执行 CTML 替代本轮思考，不调 ghost.articulate() | 否（感知结果） | 确定性指令 |
| `notify` | 数据注入 mindflow buffer，不打断不思考 | 否（下一帧看到） | 补充上下文 |
| `interrupt` | 纯打断，空 attention 关闭，无后续 | 否 | 急停 |

**扩展性**: 新增原语不破坏已有语义。将来有真实用例驱动时再加，不提前设计。

### KD2: Buffer 在 Mindflow 层，不在 Attention 层

**否决了 Attention 级 buffer**。Buffer 数据本质是跨 attention 的——notify impulse 产生时可能属于 attention A，A 被 interrupt 打断后数据应迁移到 attention B。放在 attention 上需要额外的移交逻辑。

**方案**: Mindflow 持有 `_buffered_impulses: list[Impulse]`。notify impulse 在 `_challenge_attention()` 入口处直接写入 buffer，不进入 challenge 流程。Attention 通过 `context_func` 桥接 `pop_buffered_impulses()`，每帧 `_prepare_moment()` 时 drain。

```python
# mindflow 层
async def _challenge_attention(self, impulse: Impulse) -> None:
    if impulse.mode == ImpulseMode.NOTIFY:
        self._buffered_impulses.append(impulse)
        return  # 不挑战，不打断
    # 其他 mode 走正常 challenge 流程
```

**优势**: buffer 生命周期绑定 mindflow。context_func 已是现成的桥（`_main_loop` 已用 `with_context_func('moss_dynamic', ...)`）。challenge() 保持纯仲裁，mode 语义由 mindflow 层解释，不侵入 attention。

**与同 ID 吸收的区别**: `challenge() → None`（同 ID 吸收）仍是 attention 内部的事（complete 更新）。notify buffer 走 mindflow 路径。两条线互不干扰。

### KD3: reflex = 并行，不阻塞

**reflex 不阻塞等待结果，而是与思考并行**。

```
                    ┌─ action loop: 先执行 reflex_cmds，不走 logos 通道
Impulse ──→ moment │
                    └─ articulate loop: 立即启动 ghost.articulate()
                    　  模型在 percepts 看到 "reflex X, Y, Z 正在执行"
                    　  模型照常生成 logos，可以下发 interrupt
```

**通道分离**:

| | reflex | 模型 logos |
|---|---|---|
| 来源 | Impulse 显式字段 | ghost.articulate() 产出 |
| 传输 | Action.reflex_logos() | Action.received_logos() |
| 执行 | shell 先消费 | shell 后消费 |

**记忆分离**: 下一帧 `last_outcome().logos` 只包含模型生成的 CTML（纯的）。`last_outcome().outcomes` 包含 reflex 结果 + 模型 CTML 结果（合并，按时序）。模型不会被不属于自己的 CTML 污染。

**与 command 的本质区别**: command 停下来等结果再思考（替代一轮）。reflex 边想边执行，模型可以选择 "这个反射不对，我要打断"——agency 更大。

### KD4: Reaction 链是天然的聚合器

reflex outcomes 和模型 outcomes 在同一帧 Reaction 里。`AttentionContext.outcome()` 只管 append，不区分来源。`stop_at_outcome()` 自然合并。不需要额外的 "同一个 outcome" 机制。

ghost 记忆体系通过 `on_moment` + `on_articulate_exit` 两个回调串联——对 reflex 模式，`on_moment` 记录增强后的 moment，`on_articulate_exit` 记录模型 logos。对 command 模式，只有 `on_moment`（因为没调 ghost），语义正确。

### KD5: abort 检查点放在 _stream_execute 而非解释器内部

shell.clear() 的调用时机在 GhostRuntime 层，不在 mindflow 层。
解释器不应感知 mindflow 的 abort 语义。

### KD6: mode enum 统摄而非继续加字段

Impulse 已有 15+ 字段。`reflex_logos` 重命名为 `command_logos`（通用），配合 mode 决定行为。`reaction_instruction` 保留用于 prompt 补丁场景。向后兼容：默认 mode=think 保持现有行为。

## 实施顺序

| 优先级 | Feature | 理由 |
|---|---|---|
| P0 | F3 — abort 传播 + shell.clear | 当前 bug 级缺陷，abort 后残留 command |
| P0 | F1/F6 — Impulse mode 分类 + 重命名 | 后续所有 feature 的基础抽象 |
| P1 | F2 — 空循环 | 依赖 mode 分类，实现简单 |
| P1 | F4 — Buffer 机制 | 依赖 mode 分类 + challenge 扩展 |
| P2 | F5 — 空转记录上下文 | 大部分已有，主要验证 |

## 组合场景验证

### 确定性打断 (急停)
```
Impulse(priority=FATAL, mode=interrupt)
→ challenge() → True → abort 当前 attention
→ action loop 感知 abort → shell.clear()
→ 新 attention → mode=interrupt → 空转关闭
→ 无 ghost.articulate() 调用
```

### 补充输入不打断 (notify)
```
Impulse(priority=INFO, mode=notify, messages=[...])
→ mindflow._challenge_attention(): mode=notify → _buffered_impulses.append()
→ 不进入 challenge，当前 attention 不受影响
→ 下一帧 _prepare_moment: context_func → pop_buffered_impulses() → moment.percepts
→ 模型自然看到新数据
```

### 条件反射 + 思考并行 (reflex)
```
Impulse(priority=NOTICE, mode=reflex, command_logos="...", messages=[...])
→ challenge() → True → 新 attention
→ _prepare_moment: percepts += "reflex executing: ..."
→ yield (articulate, action)
→ articulate loop: ghost.articulate() 立即启动，模型边想边执行
→ action loop: reflex_logos() → shell 先执行，再 received_logos() → shell 后执行
→ 模型可以在思考中下发 interrupt
→ 下一帧: logos 只有模型产出，outcomes 包含 reflex + 模型结果
```

### 确定性 CTML 替代思考 (command)
```
Impulse(priority=NOTICE, mode=command, command_logos="...")
→ challenge() → True → 新 attention
→ _loop(): 不 yield articulate，只 yield action (reflex_logos 填充 command_logos)
→ action 执行 → outcome → 模型下一帧感知结果
→ 无 ghost.articulate() 调用
```

## 文档交付物

实现完成后交付三件套：

1. **`moss docs`** — 提纲挈领的架构总览，Mindflow 控制语义在 MOSS 架构中的定位
2. **`moss how-tos`** — 从四个角度拆分：
   - 创建自定义 Impulse mode
   - 集成 Mindflow 到 GhostRuntime（当前集成方式 + 新能力）
   - 选取合适的控制模式（think/reflex/command/notify/interrupt 决策树）
   - 二开 Attention 子类（challenge 策略 + Buffer 机制）
3. **`tutorials/`** — 一个 tutorial，完整演示从创建 Impulse 到控制思维流的全链路

## 开发模式

结对编程：人类工程师调整抽象设计，DeepSeek V4 review + 实现。

## 风险

1. **Buffer 与 observe 的交互**: buffered 数据和 observe outcomes 在同一帧 moment 中出现时的顺序
2. **shell.clear() 幂等性**: abort + 自然结束双重调用的安全
3. **PriorityProtectionAttention 需同步升级**: 新 challenge 返回值需子类实现
4. **F3 需要集成测试**: 涉及 attention/action/interpreter/shell 四组件交互. 单元级安全测试已补 (3 个)，验证 clear 在 feed/execute 中途和多次调用的安全性。端到端仍需 REPL 验证。

## 实施记录

### F3: abort 传播 + shell.clear (2026-06-04 ~ 2026-06-06)

由 DeepSeek V4 实现，人类工程师 review。

**改动**:
1. `Action` ABC 新增 `is_aborted()` 抽象方法 (blueprint/mindflow.py)
2. `BaseAction` 实现 `is_aborted()` → `self._ctx.is_aborted()`
3. `GhostRuntimeImpl._stream_execute()` 在 feed/compile/execute 三阶段结束后检查 `action.is_aborted()`，触发 `shell.clear()` 并返回部分结果
4. 内联 helper `_check_abort_and_clear(phase)` 封装三阶段统一逻辑
5. 3 个 shell.clear 安全测试: feed 中途 / execute 中途 / 多次幂等

**关键决策偏离**: 无。按 KD5 执行。

**已验证**:
- shell.clear() 在活跃解释期间安全 (幂等，partial results 可捕获)
- mindflow 43 tests + shell 107 tests 全绿
- `_stream_execute` 改动需 GhostRuntime 端到端验证 (REPL)

---

*调研与评审: DeepSeek V4 与人类工程师, 2026-06-02*