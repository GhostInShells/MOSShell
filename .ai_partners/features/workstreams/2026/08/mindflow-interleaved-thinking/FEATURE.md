---
title: Mindflow Interleaved Thinking — 三循环解耦 + 观测缝合, 定版 v0.1.0
status: in-progress
priority: P0
created: 2026-08-26
updated: 2026-08-26
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

- 实机测试 + 修复几个原型（第三步, 尚未完成）。当前单测全绿, 重构预期能较好收工。
- llm func 仲裁: attention 数据对象化是它的预留, 本期不实现。

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
