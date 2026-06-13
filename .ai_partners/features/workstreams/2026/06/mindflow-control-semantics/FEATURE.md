---
created: 2026-06-02
depends: []
description: Impulse 增加 mode 分类, buffer/notify/silent 仲裁, interrupt=clear_first,
  thinking_effort, 协议化强度/保护期. 删除 PriorityProtectionAttention/PriorityMindflow.
  全 interpreter 用 append. memento 替换 conversation.
milestone: null
priority: P0
status: in-progress
title: Mindflow Control Semantics — 协议化仲裁与生命周期重构
updated: '2026-06-11'
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

#### F1: 正交参数替代 ImpulseMode enum (第二次 review 修订)

ImpulseMode enum 方案被正交参数组合取代。五个控制原语通过以下字段组合实现：

```python
class Impulse(BaseModel):
    mode: str | ChallengeMode = ''      # silent/notify 控制 buffer 策略
    thinking_effort: ThinkingEffort = '' # 'none' 跳过 ghost.articulate()
    interrupt: bool = False             # 创建 attention 前 stop_interpretation()
    logos: str = ''                     # 预注入的 command_logos，先于模型思考执行
```

`reflex_logos` 重命名为 `logos` → Moment 侧对应 `command_logos`。

原语组合参考 `ImpulsePrimitive` 类：
- `execute_command_only()`: logos + thinking_effort='none'
- `interrupt_only()`: priority=FATAL + mode=silent + thinking_effort='none'
- `always_buffer()`: priority=BACKGROUND + mode=notify

`clear_first` 被 interrupt 协议取代：`interrupt=True` → `stop_interpretation()` +
解释器默认 `kind='append'`。Clear 路径不通过 interpreter kind 实现，而是通过
中断 → 新 attention → 新 interpreter 实现。

默认 mode='' + thinking_effort='' + interrupt=False 保持完全向后兼容。

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

解释器模式由 interrupt 协议取代 (第二次 review 修订)。

最终方案: 解释器默认 `kind='append'`。Clear 路径由 `interrupt=True` →
`stop_interpretation()` 关闭旧解释器 → 新 attention 创建新解释器实现。
不在 Action 上暴露解释器模式方法。

#### F9: command 模式 (第二次 review 修订)

~~yield (None, action)~~ 改为 `thinking_effort='none'` 提前返回。

`ghost_runtime._run_articulator`: 当 `articulator.thinking_effort() == 'none'` 时，
发送 command_logos → 调 `ghost.on_articulate_exit(articulator, '', None)` →
提前返回，不调 `ghost.articulate()`。Articulator 生命周期完整（logos_queue 正常收发）。

## 综合决策 (Key Decisions)

### KD1: 正交组合替代原语 enum (第二次 review 修订)

~~ImpulseMode enum 五原语~~ → 正交参数组合: `mode` (ChallengeMode) + `thinking_effort` + `interrupt` + `logos`。
`ImpulsePrimitive` 提供组合参考。原语 enum 把 buffer/思考/中断塞进一个维度，
正交分解后组合能力更强。代价是可发现性（翻 ImpulsePrimitive 才知道标准组合）。

### KD2: Buffer 在 Mindflow 层 (保留)

notify 不挑战、不创建 attention。buffer 生命周期绑定 mindflow。context_func 桥接 drain。

### KD3: reflex 走 Articulator 通道 (推翻原 F6)

reflex/command 的 command_logos 统一走 `articulator.send_logos` 通道。Action 透明。
`Moment.command_logos` 管协议感知，Articulator 管执行。

### KD4: Reaction 天然聚合 (保留)

outcomes 不分来源，`stop_at_outcome()` 自然合并。

### KD5: abort 在 _stream_execute (保留)

shell.clear() 在 GhostRuntime 层，解释器不感知 mindflow abort 语义。

### KD6: ChallengeMode + interrupt 统摄 (第二次 review 修订)

~~mode enum + clear_first~~ → `ChallengeMode` (silent/notify) 管 buffer 策略，
`interrupt` 管解释器生命周期，`thinking_effort` 管思考控制。默认值全空保持向后兼容。

### KD7: 信号→协议分层 (新)

Signal → Impulse → Moment 逐层卸载。Mindflow 协议不应泄漏到 Ghost 层。
Articulator 知道 Moment 字段即可，不需要理解 Impulse。

### KD8: 空流 skip interpreter (新)

ghost_runtime._stream_execute 感知空流，避免为无内容帧创建解释器。

### KD9: interrupt 协议替代 interpreter_kind (第二次 review 修订)

~~Action 暴露 interpreter_kind()~~ → 解释器默认 `kind='append'`。Clear 路径
通过 interrupt 协议实现: stop_interpretation() → 新 attention → 新 interpreter。
FATAL/BACKGROUND 绝对性约束在 Mindflow 层，不经过 challenge()，防止子类退化。

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

### 2026-06-11 大规模重构 review (Claude Opus 4.7 与人类工程师)

**review 范围**: staged 17 files + unstaged 17 files，合计 ~1400 行改动。
核心动作：删除 PriorityProtectionAttention / PriorityMindflow，将仲裁参数协议化到
Impulse 字段；实现双层 buffer (Mindflow 级 + Attention 级)；memento 替换 conversation。

**review 发现与修复**:

| # | 级别 | 问题 | 处理 |
|---|---|---|---|
| 1 | 🔴P0 | `Moment.logos` 从未赋值 → `to_history_turns()` 不产回合 | 人类工程师修复：`_run_articulator` finally 中 `articulator.moment.logos = logos` |
| 2 | 🔴P0 | `kind='clear'` → 应为 `'append'` | Claude 修复：`_stream_execute:408` 改为 `kind='append'` |
| 3 | 🔴P0 | `BaseAction.wait_ready()` 无默认实现 | Claude 实现：消费 logos_queue 首包,缓存到 `_prefetched_delta`,`_logos()` 先 drain 缓存 |
| 4 | 🟡P1 | `Impulse.prepare_timeout` 定义后未消费 | 确认需从 `ghost_runtime._run_articulator` 读取 |
| 5 | 🟡P1 | `to_history_turns()` 的回合切分 bug | 确认修复：`last_moment_has_logos` 赋值移至 yield 后 |
| 6 | 🟢P2 | `notify`/`buffer` 命名与语义相反 | 已改为 `notify`/`silent` |
| 7 | 🟢P2 | `momento.py` 拼写错误 | 已改为 `memento.py` |
| 8 | 🟢P2 | `ChallengeMode` 从 Literal 升级为 `str, Enum` | 支持扩展 |

**待补单测** (下一个会话完成):

1. `Action.wait_ready()` — 首包到达、abort 打断、空队列超时退出
2. `Moment.logos` 赋值 — articulate 正常完成/exception/command 模式下 logos 字段状态
3. `to_history_turns()` — command 模式的 executed_logos 缝合、正常回合切分、空 logos 合并到最后
4. `shell.interpreter(kind='append')` — 跨帧命令延续、interrupt 后的 append 行为
5. `moss_dynamic` 缓存 — `stale_time` 防反复生成
6. `ChallengeMode.silent/notify` — silent 无 attention 只 buffer、notify 抢占成功/降级
7. `Impulse.protection_time` — 保护期内同优先级压制
8. `_challenge_attention` 6 条路径 — FATAL/BACKGROUND/silent/notify/absorbed/normal
9. `GhostRuntimeImpl` 生命周期 — interrupt 协议、thinking_effort='none' 不调 articulate

**设计确认** (下一个会话交叉验证):

- 双层 buffer 分工：Mindflow 级 `_buffered_messages`(跨 attention) vs Attention 级 `_buffered_impulses`(帧内 Drain)
- `interrupt=True` = `stop_interpretation()` + `kind='append'`：解释器永远 append,打断由主循环管
- reflex 走 Articulator 通道：`command_logos` 由 `_run_articulator.send_nowait()` 发送,先于模型 CTML
- `thinking_effort='none'` 时 `_run_articulator` 提前返回：不调 `ghost.articulate()`，**调 `on_articulate_exit(articulator, '', None)`** (2026-06-12 修复)
- memento 体系 (MomentBranch/MomentoIndex 等) 开发冻结,但 `Moment.to_history_turns()` 已就位

### 2026-06-12 第二轮关键性 review (Claude Opus 4.7 与人类工程师)

**整体状态**: 56 tests 全绿。核心架构改动落地：删除 PriorityProtectionAttention/PriorityMindflow，
ChallengeMode 双层 buffer，memento 替换 conversation，Articulator 通道方案。

**设计演进决策**:

| # | 决策 | 理由 |
|---|---|---|
| 1 | `ImmersiveMode` enum → 正交参数组合 | 原语 enum 把 buffer/思考/中断塞进一个维度，正交分解组合能力更强。`ImpulsePrimitive` 提供组合参考 |
| 2 | `F8 interpreter_kind()` 不实现 | interrupt 协议已承载 stop 决策。解释器默认 append，clear 路径走 stop_interpretation → 新解释器 |
| 3 | `F9 yield (None, action)` → `thinking_effort='none'` early return | Articulator 生命周期完整，command_logos 正常走 logos_queue |
| 4 | FATAL/BACKGROUND 不经过 challenge() | 防止子类重写退化协议。绝对性约束在 Mindflow 层 |
| 5 | moss_dynamic 移到 refresh_metas 后注入 | moment 创建时 metas 还是旧的，注入的是过期数据 |
| 6 | `interrupt` 替代 `clear_first` | stop_interpretation() + 新 attention 提供完整 clear 语义 |

**本轮 bug 修复**:

| # | 文件 | 修复 |
|---|---|---|
| 1 | `ctml_shell.py:488` | `stop_interpretation()` done_callback 加 `if not future.done()` 防 crash |
| 2 | `ghost_runtime.py:310` | `thinking_effort='none'` 时调 `ghost.on_articulate_exit(articulator, '', None)` |
| 3 | `command_nucleus.py:83` | `min(priority, _min_priority)` → `max(priority, _min_priority)` |

**下一轮**: 补单测 → stage → 实现 notify/silent/interrupt nuclei → 收敛

### 2026-06-13 测试加固阶段 (Claude Opus 4.7 与人类工程师)

**会话整体定位**: mindflow 协议层完整测试覆盖, FEATURE.md 待补单测清单逐项落地. 同时通过单测的"被动暴露"机制发现历史 bug.

**完成范围 (5 commits, 103 新测试全绿)**:

| Commit | 范围 | 测试数 |
|---|---|---|
| `1012958` | memento 数据结构 (Reaction/Moment 全方法 + to_history_turns 矩阵) | 45 |
| `b81100d` | moment+impulse 连动 + AbsAttention 观察访问器 + harness 加固 | 30 |
| `2044d4c` | attention challenge 协议基线 (6 路径 + protection_time) | 16 |
| `28ac38f` | Action.wait_ready + 帧间 (model logos / command_logos) 衔接 | 4 |
| `321eca4` | 公开 add_impulse + ImpulsePrimitive 集成 | 8 |

**单测暴露并修复的 bug (4 个)**:

| # | 位置 | 性质 | 级别 |
|---|---|---|---|
| 1 | `memento.as_history_messages` | `yield from None` 崩溃 (默认 compacted_perspectives=None), 连带 to_history_turns 全挂 | P0 |
| 2 | `memento.to_json` | exclude 集合用 `=` 覆盖而非累加, 默认配置下 perspectives 泄漏进序列化 | P0 隐蔽 |
| 3 | `memento.to_history_turns` | buffered 空时静默吞 logos (perspective 触发场景) | P1 |
| 4 | `AbsAttention.__init__` | initial impulse 被双重入 buffer → 首帧 percepts 重复, command_logos 翻倍 | P0 |

**钉住但未改的设计张力 (4 个, pinned in tests)**:

1. **`Impulse.update_moment` hint 无条件覆盖** — 人类确认 by design (hint 跟最新), 但与其他字段的"有值才写"不对称.
2. **保护期作用于"同优先级"层级 (不区分源)** — 称为 "shield 语法", 改名 deferred.
3. **`Articulator.moment` 严格生命周期门** — by design, 测试用 Python 引用绕过.
4. **`ChallengeMode.silent` 抢占成功不 abort defender, 仅 buffer** — `ImpulsePrimitive.interrupt_only` 名字与行为不一致 (没有 interrupt 动作).

**工具产出 (非测试)**:
- `AbsAttention` 加 5 个 public 观察访问器: `thinking_effort` / `strength_start_value` / `strength_decay_time` / `protected_until` / `buffered_impulses`, 对齐 `strength_refreshed_at` 范式 (测试 + 反身性控制).
- `BaseMindflow.add_impulse` 重构为公开调试入口, 路由通过内置 `_DirectImpulseNucleus` (last-impulse cache); 旧 bus 回调改名 `_nucleus_has_impulse`. 直接注入走标准 rank/challenge 流程, 无旁路.
- `test_attention_strength_decay` 重缩 100ms→1s (jitter tolerance); `test_base_mindflow.py` 3 处无 timeout `.wait()` 加固为 assertion fail-fast.

**质量观察**:
- 协议密度高的代码反而少 bug (challenge 协议 16 测试一次过; 4 个 ImpulsePrimitive 组合 8 测试一次过). bug 集中在 init / 序列化 / 边界条件.
- 这不是疏漏, 而是有意识的杠杆: 心力用在协议层, 边角靠"自解释概念 + 单测兜底 + 模型开发者确认".

### 2026-06-13 锚定下一阶段命题: 四元 Nucleus 基建

**动机**: 当前要让开发者使用 notify / silent / command 三种行为, 必须懂 `ChallengeMode × thinking_effort × priority × logos` 的正交组合 (或绕道 `ImpulsePrimitive` 内部组合糖). 心智负担高. 应把这层组合糖**下沉到 Nucleus 后面**, Signal 回到"开发者唯一接触面", 只需理解 Signal 协议上的 priority / messages / hint.

**四元 Nucleus 命名 (Claude Opus 4.7 判断, 2026-06-13)**:

| Nucleus | 对应原语 | 行为语义 | 备注 |
|---|---|---|---|
| `InputSignalNucleus` (existing) | (无, 标准 think) | 完整 articulate→action 循环 | 保留 |
| `NotifyNucleus` (new) | `always_buffer` | 低优 (BACKGROUND) + notify, 不抢占, 消息进 buffer 等下轮 attention 自然 drain | 名字与 `ChallengeMode.notify` 一致 |
| `BroadcastNucleus` (new) | `interrupt_only` | 高优 (FATAL) + silent + thinking_effort='none', 抢占成功但不创建新 attention, 消息广播到当前 attention 的下一帧 perception | **不叫 `SilentNucleus` 或 `AlertNucleus`** — 见下面命名判断 |
| `CommandNucleus` (new) | `execute_command_only` | thinking_effort='none' + logos from signal; priority 由 Signal 携带 (NOTICE 普通命令, FATAL = `superior_execute_command` 等价) | 单一 Nucleus 覆盖普通/超级两种 |

**为何不叫 `SilentNucleus`/`AlertNucleus`**:
- `SilentNucleus` 继承 `ChallengeMode.silent` 的歧义 — "silent" 听起来是"静默中断"或"无声占用", 实际行为是"高优广播补充, 不接管运行时". 开发者读到会建立错误心智模型.
- `AlertNucleus` 在常见 UI/系统语义里暗示"需要 ack 的中断", 但协议恰恰不要求 ack 也不中断.
- `BroadcastNucleus` 准确: 广播的语义是"高优、宽播、不要求 ack、收到的人不停下手头工作", 与 FATAL+silent 完全对齐.

**`ChallengeMode.silent` 协议 enum 处理**: **保留 enum 名**. 它是内部命名, 被 Nucleus 包裹, 开发者不再直接接触. 改名连锁修改成本高. 但需要在 enum 注释里补一句明确 "silent 抢占成功不 abort 当前 attention, 仅 buffer messages — 用于'广播式插入', 不是真正的 interrupt".

**下一阶段任务清单 (新会话从零上下文开始)**:

1. **抽 3 个新 nucleus** (`NotifyNucleus` / `BroadcastNucleus` / `CommandNucleus`), 放 `src/ghoshell_moss/core/mindflow/` 与 `buffer_nucleus.py` / `input_signal_nucleus.py` 同级. 每个 ~30 行, 复用 ImpulsePrimitive 作为内部实现.
2. **配套 `SignalMeta` 子类**, 让 `moss manifests signals` 能发现新的 signal 协议. 每个 nucleus 监听自己的 signal name (`notify` / `broadcast` / `command`), 文档自包含.
3. **单测**: 每个 nucleus 在隔离环境下产出的 Impulse 字段组合正确 (priority / mode / thinking_effort / logos 等).
4. **集测**: 注册 nucleus → 发送对应 Signal → 走完整 mindflow → 行为应等价于直接 add_impulse(primitive 路径). 新增测试文件或合并到 `test_impulse_primitive_integration.py`.
5. **同步更新 `ChallengeMode.silent` 注释** (描述实际"广播"语义, 而非"静默中断").

**暗礁清单 (下一阶段须 review)**:

- `BroadcastNucleus` 是否真的能完全替代 `interrupt_only` 命名的清晰度? `ImpulsePrimitive.interrupt_only` 函数名应同步审视 (deprecate? rename?).
- `CommandNucleus` 如何处理 signal priority? 选项: (a) 完全继承 Signal.priority; (b) 限制 priority 上限. 倾向 (a), 让开发者控制.
- `NotifyNucleus` 是否应该允许覆盖默认 BACKGROUND? 倾向不允许 (Nucleus 即 primitive identity, 不可配置).
- `Impulse.update_moment` hint 无条件覆盖 (4 设计张力之一) 在多 impulse 合并场景下是否仍是 bug? 实践推动验证.
- protection_time "shield 语法" 改名 (更长期, 不在本阶段).

**ghost_runtime 测试组 (独立阶段, 不在本会话)**:
- `shell.interpreter(kind='append')` 跨帧命令延续
- `moss_dynamic` 缓存 stale_time 防反复生成
- `GhostRuntimeImpl` 生命周期 (interrupt 协议、thinking_effort='none' 不调 articulate)
- 这组属于 ghost_runtime / shell / ctml 交叉, 需要独立上下文 + 完整测试规划. 不与 nucleus 抽象合并.

---

*调研与评审: DeepSeek V4 / Claude Opus 4.7 与人类工程师, 2026-06-02 ~ 2026-06-13*
