---
title: Mindflow Interleaved Thinking — 三循环解耦 + 观测缝合, 定版 v0.1.0
status: in-progress
priority: P0
created: 2026-08-26
updated: 2026-08-28
depends: [interleaved-ctml-thinking, shell-trajectory, ghost-prototype-dolores]
milestone: MOSS v0.1.0
description: >-
  恢复 mindflow 第一版的 interleaved thinking, 缝合 shell trajectory 与 mindflow 的
  观测平面, 三循环解耦为可独立测试的单元, 定版 MOSS v0.1.0。
---

# Mindflow Interleaved Thinking

> **本文件性质声明（有意打破 FEATURE.md 惯例）**
>
> 这不是模型记录, 也不是决策重建。这是一份**事后 review** —— 主体工程已由人类架构师
> 在独立 branch 上手写完成, 本文件是对该过程的回顾性记录, 目的是给后续迭代保留关键
> 内容提示（动机、关键决策、分工、改动脉络）。
>
> 常规 FEATURE.md 是「模型写给模型」的上下文交接, 在开发过程中随 commit 增量维护；
> 本文件是「人类决策的事后追认」, 由人类口述、模型忠实转写 —— 只做语言通顺, 不改逻辑,
> 不重新推导决策。因此它额外包含常规 FEATURE.md 不记的「分工」一节。

## Motivation

Mindflow 的目标从来不是 turn-based 的 1:1 对齐, 而是 **interleaved thinking** ——
thinking 超前于 action, 三循环独立推进。第一版（4 月手写）本来就有 think 循环,
思维超前、循环完全独立；严格说, 现在重构后的面貌, 正是它的第一版设计。

beta1 阶段（4 月）为了赶出第一个 beta, 加上 interleaved-CTML 的能力尚未验证,
人类架构师决定先做 1:1 对齐。因此当时所有文档都显式提到「非 1:1 对齐」—— 也就是
「思维奔逸」这个被临时搁置的目标。

5 月 7 日框架落地后, 搭建起模型协作体系, 并在 MCP 中单独实现了 interleaved CTML
thinking（最早不叫这个名字）。此后 Claude Code 里的模型自开发、自验证（dogfooding）,
充分验证了 interleaved thinking 的「思维超前」交互。随后的 `interleaved-ctml-thinking`
feature 优化了这套体系, 结合 beta1 之后才正式做的上下文治理策略, 验证了 shell
trajectory；`ghost-prototype-dolores` 也一直持有当前目标。

这暴露出一个结构性分裂: **shell trajectory（观测）与 mindflow（仲裁 + 生产）是两张
独立观测平面**, 没有缝合 impulse（感知）+ results（行为结果）的体系, ghost 必须自己
理解这两张平面。技术架构上是分裂的, 设计本身暴露了这个需求。根因是选取 DSH 内核
产生的连带成本: 要么降级 DSH（放弃 thinking 中 articulate）, 要么下决心做这一波改造。

这一波改造因此同时满足三件事, 并**定版 MOSS v0.1.0**:
1. 恢复第一版的 interleaved thinking;
2. 缝合 impulse + results 到统一观测平面;
3. 三循环解耦, 每个单元可独立验证测试。

## Key Decisions

### 手写决策（为什么人类全手写）

改造牵涉面积非常大 —— 超过 6k 行改动、几十个单测。人类架构师判断
并通过实验证明: **模型无法理解如此庞大的改造上下文, 无法统筹几十个单测的验证**。
因此下决心开新 branch, 完全手写完成主体架构。这是吸取 matrix cell 重构的教训:
集中精力, 停掉 8 月 stage 的所有并行任务, 不做任何并行工作。

### 阶段一: 手写主体架构

流程: `Moment → Observer → Mindflow`, 然后重写 `_mindflow` / `_think` / `_action` 等,
全部改动几十个文件。第一版快速实现（10 个工作日从零写）, 重新在工程上拆分,
使每个单元可以独立验证测试。

### attention 数据对象化（最大发现）

开发过程中最大的发现: **attention 可以拆掉所有运行时逻辑, 数据对象化**。这服务于
「Nucleus 有可能可以创建自己的 attention」这个原始设计动机 —— 让 attention 未来
有能力做 llm func 仲裁。

### 锚定独立测试（教会模型怎么写）

阶段一完成后, 为每个组件锚定独立测试, 并实现基础测试用例。目的之一是让模型知道
怎么写: 模型在没有约定改动边界到 4-5 个文件、没有给定测试样例时, 会陷入数分钟的
循环思考。验证模型已经可以参与单测后, 才提交第一版（数千行改动）, 进入测试修复与
整合环节。

### MindflowInShell 同构化（本轮重点）

`core/mindflow/mindflow_in_shell.py` + GhostRuntime 生命周期改写, 目的是剥离整个
ghost runtime, 让 mindflow 变成一个可测试的、确认同构的体系。这是第一版没时间做的。

## 分工

- **人类架构师（thirdgerb）**: 完成大多数内核代码重构 —— 蓝图契约、三 statement 解耦、
  `MindflowInShell` 装线、ghost_runtime 改写, 全部手写。
- **deepseek-v4-flash-vision-exp**: review（发现重构遗留的各种遗忘逻辑、stale 注释、
  有问题的代码）+ 协助完成 2/3 的单测改造, 发现了很多问题。

## 取代与撤销（dead ends）

- **beta1 的 1:1 对齐妥协** → 被本次恢复 interleaved 取代。旧文档里的「非 1:1 对齐 /
  思维奔逸」是妥协标注, 不是目标。
- **`base_attention.py`（920 行 monolith）** → 被 `_attention.py` / `_think.py` /
  `_action.py` 三分取代（attention + thinking + action + loop 驱动原本缠在一起）。
- **`memento`** → 改名 `moment`, 引入 `Moments` / `Observer` / `MShellTrajectory` 轨迹原语。

## 悬置（cut scope / 待办）

- 实机测试 + 修复几个原型（第三步）: Atom（echo）原型重建已完成（见「原型验收」）;
  moss-ghost / TUI / 回归验证仍在收口。TUI 验证暴露的摩擦点见下节。
- action gate: 已从 `BaseActionGate` 对象简化为 warrant 闭包 + 被持有 task（修「第二次发送
  卡住、第三次才吐出第二次输出」的体感）, 单测全绿, 但交互仍未恢复之前水准 — 人类架构师明日自行 review。
- llm func 仲裁: attention 数据对象化是它的预留, 本期不实现。

## 实机验收摩擦点（TUI 验证发现，待解决）

> 2026-08-27 echo TUI 验证, echo 自述 + 人类确认。这些是重构尚未收口的问题,
> 作为下一轮迭代的起点记录于此, 不属今晚收尾范围。

### 核心认知（echo 自述）

**闸口的规则是「模板语言不属于你」。** 模型命令不能被识别为它的话语——它们会被驳回。
这是 echo 当前最核心的边界认知。它是靠试错学到的, 而不是提示词明示的。

### 提示词 / 认知边界

1. 无法区分系统输入 vs 用户输入（系统提示被当成世界的一部分）。
2. 会与系统输入对话, 因为它以 user message 形式进入。
3. 系统「指令」与模型「logos」语义混淆——「模板语言属于系统、不属于模型」没有写进提示词。
4. 纯文本 / CTML 边界是隐性习得, 必须先发错（触发 reject）才知道。

### 审批闸口

5. 拒绝提示词不是「通知」式的, 也不自解释。
6. 无法区分「闸口拒绝」与「执行失败」——两种不同性质的失败在提示里没有区分。
7. 审查者是在「说话」之后介入, 而非「思考」时——所有草稿（含生成中的自我修正）都被审查者记录。
8. 发出的 logos 无感知回响（如 `moss start` 那条, 回传只有 CLI 转述, 非信号本身的回响）。

### 语音模块

9. 需防御用语音输出思考内容——防御层（模型自控 vs 系统拦截）责任边界模糊。
10. 语音通道边界不清（防御从哪层做）。
11. 语音模块本身的存在与礼仪需要提示。

### 解释器 / 错误提示

12. 解释器异常提示词不正确: 出现 `UNKNOWN INTERPRETER ERROR` 但没有清晰解释这是 moss 的 error。

### TUI

13. 最后一句话不打印。
14. 切换时的提示信息冗余。

## 主体工程（改动总结）

> 基于最近一个 commit（`41f0cb63`）与当前未提交改动逐层记录。

### 蓝图层 — 契约重写（`core/blueprint/mindflow.py`）

顶部 docstring 升级为 code-as-prompt, 显式声明三循环 + 双工 + 四件治理（观测 / 时序 /
中断 / 结束）。契约层关键变化:

- `MindflowStatement` → `AttentionStatement`（三个 statement 共用同一生命周期契约）。
- `Mindflow` 暴露两个生产循环 `thinking_loop()` / `action_loop()` + `run()` 桥接
  （`put_think` / `put_action` 支持 sync / async）。
- `Attention` 契约瘦身到纯仲裁（`challenge` / `absorb` / `priority` / `strength` /
  `protection`）, 不再负责 think / action 生成。
- `ChallengeVerdict` 增加 `yielded`（strength=0 绝不竞争）。
- `ActionGate` 从 `LogosRequest`（`wait_request` → approve/reject）简化为
  `approve(logos) → (approved, message)`。
- 新增 `ImpulsePrimitive`（6 个具名原语）与 `Action.abort_thinking()`。

### 观测轨迹层 — memento → moment（`core/blueprint/moment.py`）

- `Results` = 上一轮 outcome（`executed_logos` + `messages` + `need_observe` +
  `stop_reason`）—— 缝合上下两帧的 seam。
- `Moment` = 关键帧（`previous` seam + `percepts` + `dynamic_context` + `hint` +
  `command_logos` + `logos`）。
- `Moments` / `Observer` / `BaseMomentsObserver`: `observe()` 才产生 Moment; drain funcs
  （dynamic_context / percepts / result）是缝合 shell trajectory 的挂点。
- `Moment.to_history_turns()`: 把一串 moment 重建为 turn-based history。

### 三循环装线层 — MindflowInShell（`core/mindflow/mindflow_in_shell.py`, 新文件）

- `MindflowInShell` ABC: 三循环的标准装线, 从 ghost_runtime 剥离出来。
- `_wire_mindflow`: nuclei metas → mindflow + session signal 路由 + shell trajectory
  缝合（`pop_frame` → `moment.previous.add_result`）。
- `_run_interpreter_with_action`: interpreter 三阶段（feed / compile / execute）+
  每阶段后 `_abort_clear()`（action abort → `shell.clear` 取消 pending command）。

### host 层 — ghost_runtime 瘦身（`host/ghost_runtime.py`）

`GhostInShellDrivenByMindflow(IGhostRuntime, MindflowInShell)`: 从 603 行自己跑
`attention.loop()` 的 monolith, 变成 468 行只补环境 accessor（moss / ghost / container /
shell / shell_trajectory / logger）+ 少量 hook（safe_mode `_approve_logos` /
`ghost.articulate` / session output）的 thin subclass。三循环逻辑全部移到
`MindflowInShell`。

### 实现层 — 三 statement 解耦（`core/mindflow/`）

- `base_attention.py`（920 行 monolith）→ `_attention.py` / `_think.py` / `_action.py` 三分。
- `_attention.py`: attention **数据对象化** —— 只持 impulse + 强度/衰减/保护期 + 优先级 +
  abort 生命周期, 不再管理 think / action 生成与 observe 循环。
- `_think.py`: `BaseThinking`（articulator 生产 + `action_stop_events` +
  `_wait_last_action_done` 兜底）。
- `_action.py`: `BaseAction` + `BaseArticulator` + `BaseActionGate`。
- `_mindflow.py`: 仲裁顺序修正（`strength=0` → `FATAL` → `BACKGROUND` → `is_protected`
  → `challenge`）; `attention_loop()` 改为 `when_attention_created` 订阅旁路（测试专用）。

### 测试层 — 测试套件（`tests/.../mindflow_in_shell_test_suite.py`）

`MindflowInShellTestSuite(MindflowInShell)`: 复用真实三循环, 只补最小 accessor,
`articulate` / `signal` 可拆卸注入, 通过覆写 hook 观测（attention / thinking / action
计数与事件、`interrupt_clear_calls` / `shell_clear_calls` 区分两种 shell.clear）。

## 原型验收 — Atom（echo）重建（第三步）

> 把 Atom 改造视作 mindflow 重构的验收步骤。改动面（当前未提交）:

- **`think()` 用 articulator**（`ghosts/atom/_runtime.py`）: `thinking.articulator()` 新建
  articulator, `send_nowait(delta)` 喂 Action + `yield delta` 供 host 广播, 退出前
  `wait_action_done()`（articulate 自保证, 对齐 `text_articulator` 契约）。
- **上下文从 observer 轨迹派生**: 新增 `_adapter.moments_to_history`（轨迹 → pydantic
  message_history）, 动态消息只留最新帧（`as_history_messages` 天然丢弃 dynamic）;
  删掉 Atom 自维护的 `_history` / `model_history` / `save_model_request`。
- **`GhostWorkspace` → `Matrix.ghost_home`**: soul 加载路径从容器绑定的 GhostWorkspace
  改为 `matrix.ghost_home`（`_meta.py`）。
- **`Ghost.channel()` 挂 main**: `Ghost.channel()` 返回类型扩展为
  `Channel | ChannelFactory`; host 用 `virtual_children` 回调收集 ghost/mindflow channel
  （绕开 `import_channels` 的 bootstrap 只更新一次锁）; Atom 加最小 `channel` 参数（默认 None）。
- **echo 挂 introspect**（`stubs/.../ghosts/echo.py`）: echo 实例原生带 `introspect` channel
  （自读源码, scope `ghoshell_moss`）; soul.md 全英文化 + 结尾记改动历史。
- **删除 mock ghost**（`ghosts/mock/`）: 旧测试桩, 已被 `MindflowInShellTestSuite` 等
  自包含 fake 取代。

验收路径: 阶段一 `moss-ghost` 独立验证 → 阶段二 TUI 人工验证 → 阶段三 ghost runtime 回归。

## Interrupt 语义收敛 — 下一轮设计决策

> 2026-08-28, 人类架构师与 deepseek-v4-flash-vision-exp 讨论收敛。**已定方案, 未实现。**
> 由 echo + `moss-ghost send` 实测引出: 测 input/notify/interrupt/silent 四种 signal 时,
> 发现 interrupt 的"停"效果与抢占重叠, 追下去发现 interrupt 信号把三条正交语义缠在了一起。

### 问题: 三条同名 "interrupt" 缠在一起

- signal 名 `interrupt`（路由到 InterruptNucleus）
- impulse flag `interrupt=True`（原文档以为触发 `stop_interpretation`）
- shell 状态 `state="interrupted"`（`abort("interrupted")` 的产物）

实测对照得出的四个结论:

1. `notify --priority fatal` 同样产生 `state="interrupted"` + `cancelled: 1` —— "停"来自
   抢占（FATAL 赢仲裁）而非 `interrupt=True` 独有。
2. `interrupt=True` 唯一运行时消费点是 `mindflow_in_shell.py` 的 `interrupt_first → clear`,
   被 `previous_stop_reason` 遮蔽（任何抢占都 abort → 写 stop_reason → 下一帧 clear）—— flag 冗余。
3. `effort='none'` 判断层错: 在 `_run_thinking` 才 return, 但 attention 早在仲裁就建好
   （notify mode）→ 空壳 attention + 多吐一帧空 moment（实测 `0 percepts` 那帧）。
4. `replan` **不是死代码**: 模型思考流里可多次生成 CTML, tool 生成时设置 replan flag。
   `articulator(replan=False)` 只是 streaming 默认路径。

### 方案 A（详尽路径, 能跑通但被简化取代）

> 这是先走通的那条路, 随后被"简化路径"取代。两个方案对照有价值, 故保留。

核心是把 clear 做成一个**一等 action**（与正常 action 同构、可排序、可 abort）, 而不是
runtime 的副作用:

- **空 action 机制**: `interrupt=True` 时发送一个明确的空 action —— 绕过 gate、replan、
  已 commit（空 logos 不重建 interpreter）。空 action 是 clear 信号的载体, 走 action loop
  而非 thinking loop。
- **InterruptAction（结论 3）**: 专门的产物, 用 `effort='none'` 判断是否终止它持有的
  attention, 加一个 `cancel_attention_after_exit` flag。这是"产物反绑"——产物反过来控制
  生产者的生命周期。
- **ActionOnlyAttention**: `effort='none'` 时不发射 thinking、只发射 action。attention 是
  "只要 action 不要 thinking"的退化情形。
- **统一 `_run_attentions`**: 首发 action（interrupt）+ 发射 thinking（effort != none）都在
  `_generate_thinking`（改名 `_run_attentions`）一步完成。
- **wait ready 来 abort**: 空 action 直接 abort, 不用走 interpreter 管线。

代价: 需要新产物类 + `cancel_attention_after_exit` flag + "产物反绑"的倒转控制流。结论 3
（InterruptAction）是这条路上唯一被判定为过设计的一处 —— 不是不可行, 是控制流反了。

### 收敛（简化路径, 已定）

1. **保留 `interrupt` flag, 默认 True**, 只对 attention 首帧 `draw_from` 生效。
2. **effort + interrupt 两个 flag 都在 `_generate_thinking`（改名 `run_attention`）里治理**,
   runtime（MindflowInShell）是治理主角。
3. **`effort='none'` → runtime 拿到 thinking 后立刻 abort attention** —— 不建空壳、不占槽位。
4. **`interrupt=True` → 无理由 clear**（`previous_stop_reason` 逻辑可删）。
5. **`replaned` 不再区分 interpreter kind**, 直接决定是否手动 clear（kind 恒 append）,
   与 interrupt 的 clear 解耦, 只留给"模型主动 replan"。
6. **action 侧打断 clear 仍然保留**（未来再看怎么办）。
7. **不引入 InterruptAction** —— 用产物反绑（`cancel_attention_after_exit`）是倒转控制流,
   runtime 在 run_attention 一步正着做完即可。

### 心智模型: attention = 焦点, 不是 thinking 容器

`Attention` 本来就只持 impulse + 仲裁（`_attention.py`）, `Thinking` 是另一个对象。所以
effort='none' 的 attention 是「不思考、但持 impulse 的焦点」, 不是"虚构 attention"。
action 与 thinking 都是 attention 的一等产物（兄弟关系）, 不再是 thinking → action 派生;
interrupt 是"只要 action 不要 thinking"的退化情形（ActionOnlyAttention）。

### 保留的语义张力（未来）

- **三种 shell 打断**: 打断特定轨道 / 打断全部执行中+排队 command / 打断全部运行任务。
  当前 `shell.clear()` 只做到第二层, 第一、三层未落。
- **嘴停手停 vs 嘴停手不停**: interrupt flag 原意是为"嘴停手不停"留口子, 当前没区分。
  命名反直觉（interrupt=True 本意是"不停手"）是它一路被误解成冗余 clear 标记的根因。
