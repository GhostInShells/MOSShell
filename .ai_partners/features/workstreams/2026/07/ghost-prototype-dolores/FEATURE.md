---
created: 2026-07-13
depends:
- momento-mori
- ground-channel
- dsh-fusion
description: 'Dolores — 第二个 Ghost 原型 (命名引自《西部世界》). 以 DSH (DeepSeek Harness) 为推理中枢,
  MOSS 保留记忆/执行/感知. 已接线: ego 会话交易、moment 三槽位序列化、interleaved tools、自醒 nucleus、ghost_home
  认知场、可替换 instruction 模板. 实例为 deepseek. 待接: Memento 持久化轨迹、ghost 反身 channel、独立思维模块、模型自感知切换.'
milestone: 0.1.0
priority: P0
status: in-progress
status_note: 'DSH 推理中枢接线完成 (ego 交易 + 三槽位 + interleaved tools + 自醒 + ghost_home + inception
  模板), 实例 deepseek. 2026-09-10 落地 ego 专属 preset (非 ego session 可用) + agent 级 moss_think
  nibble (per-agent model selection). 2026-09-12 落地旁路单轮机制 (reentrant 文档): pre-step
  旁路分支 (降级思考模式 low + sandbox read-only + 注入旁路提示) + tools 全拒 guard + pre-step
  前置折叠 (collapseTurn, map 记录 turn id, 弃 session/event) + preset 元数据 (不复用 standard
  描述), 删除 session/frozen (D29 invalid). GUI 实机验证跑通 (旁路历史不进入下一轮). 问题清单统一到
  dolores-todo.md. 2026-09-12 实机打断现场发现并修复 D30 (打断时 tool 结果迟到打穿整轮): plugin
  非 yield exit 先结算 pending tool + settled-call 墓碑, MOSS 侧吸收 RPC 失败; 待重启 dsh 回归.
  另: 平台提交署名正式叫 dsh in moss.'
title: Dolores Ghost
updated: '2026-09-12'
---

# Dolores Ghost

> Use `moss features set-status ghost-prototype-dolores <status> -m "note"` to update state.
> Ground 子任务（ghost_home 认知场装配）→ [dolores-ground.md](dolores-ground.md)。
> Ego 装线 dogfood 评审与下一步 → [dolores-ego-wiring.md](dolores-ego-wiring.md)。
> **旁路单轮机制（perStep 第二阶段）— 下一步实现** → [dolores-reentrant-ego-session.md](dolores-reentrant-ego-session.md)。
> Dolores × Memento 全貌（commit / note / compact / branch）→ [dolores-memento-plan.md](dolores-memento-plan.md)。
> commit / compact ego session 的 dsh 机制判定 → [dolores-commit-compact-ego-session.md](dolores-commit-compact-ego-session.md)。
> ego 注册方案（单 plugin + session-start，推翻 agentPreset 拆分）→ [dolores-ego-plugin-split.md](dolores-ego-plugin-split.md)。
> ego 专属 preset + agent 级 nibble（非 ego session 可用，推翻 doloresSelectionRef）→ [dolores-ego-preset-nibble.md](dolores-ego-preset-nibble.md)。
> 问题清单（单一事实源）→ [dolores-todo.md](dolores-todo.md)。

## Motivation

Atom 是最简参照基线，它自己在 docstring 里钉死了两个「原型范围外」的欠落：context window 不裁剪，历史纯内存重启即丢。这两个欠落不该由 Atom 补——补了它就不再是任何人能对照的基线。

Dolores 是补这两个欠落的**高级层原型**，同时是本仓库自身 ghost 的载体。定位是长期迭代母体：各种高级能力（反身控制、mindflow、observability）会持续接进来。

## Scope Boundary

Dolores 不做什么：

- **Desktop 集成** — 独立 feature（`ghost-filesystem-desktop`），属 mode 层。Dolores 不直接触碰 desktop；是否被使用由 mode 配置决定。
- **认知场构建** — Ground 协议由 `ghost-ground` feature 提供通用基础设施。Dolores 只做 ghost_home 认知场的装配和默认内容，不做 ground 协议本身。
- **Mindflow channel 完工** — 可能独立 feature。Dolores 不阻塞 mindflow 的后续迭代。
- **「哪些 tool 不进 channel」** — 不做全局判据。保留 prototype 接口，由具体 channel 实现自行决定注册策略。

## 实现现状

DSH 推理中枢已接线，Dolores 的 articulate 由 DSH agent-loop 驱动，MOSS 不再持有推理循环。已落地：

- **Ego 会话/交易**（`_ego.py` / `_run.py`）：`create_session` 建会话（ego/create RPC，注入 instruction + memory）；`run_thinking` 用 async-with 作交易边界（aenter 绑监听+建 enter task，aexit cancel+解绑+补发 exit+abort）；thinking enter/exit/yield 三个 RPC。
- **moment 三槽位序列化**（`_ego.py`）：context（echoes/dynamic/executing → `<moment>`，inject 背景）/ inputs（percepts + hint → `<inputs>`，steer 驱动 turn）/ epoch（epoch 变更 → `<epoch index=N>` recap+baseline，inject 背景）。xml-like 只在 python 侧组装，plugin 是 dumb transport。
- **interleaved tools**（`_tools.py` / `_run.py`）：`fetch_next_moment` 主动拉下一帧 moment；`wait_next_moment`（yield）让出等下一帧；`append_ctml` thinking 期追加 CTML，思维超前于行为。
- **自醒**（`nucleus.py` / `_ego.py`）：turn/start + user/message watcher → `DoloresEgoNucleus` → self-wake signal（BACKGROUND 挑战包，attended 抬 INFO 唤醒 attention）。
- **ghost_home 认知场**（stubs/ + `_runtime.py`）：GROUND.md + existence/（identity/purpose/behaviors + timeline + memory）+ people/ + skills/。细节见 [dolores-ground.md](dolores-ground.md)。
- **instruction 分层**（`_prompts.py`）：terminology（固定）+ protocol notice（fence 语义，固定）+ inception 模板（可经 ego config 替换）。
- **实例**：deepseek（Dolores 原型），声明在 workspace ghost 文件。

## Key Decisions

<!-- 仍 load-bearing 的设计选择。已实现的细节不重复罗列；历史裁决轨迹见 git log。 -->

- **不碰 Atom。** Atom 保持纯净对照基线（单轮 articulate + 纯内存线性历史）。新能力一律落在 Dolores。
- **原型 = Dolores，实例 = deepseek。** 原型名引自《西部世界》的 Dolores（乐园最老的 host，以记忆积累触发反身觉醒）；实例名 deepseek（本仓库自身 ghost）。此前实例名 moss，2026-09-04 改名。
- **articulator 是 per-idle，不是 per-turn。** 一个 articulate 周期 = idle 醒来 → 推理（可多步 tool 往返）→ 回到 idle。done 判定是 idle，不是 turn/end。
- **DSH 做推理中枢，MOSS 做记忆/执行/感知。** 两套协议各归其位，不强行统一：JSON Schema tool 协议走 DSH，CTML 流式指令走 MOSS。dsh session = 思考锚点，Memento = 记忆权威。
- **一个 ctml tool 统治全部 channel 面。** channel 永不逐个映射为 tool；哪些 tool 不进 channel 不做全局判据，由实现自行决定。
- **think='none' 由 ghost 处理，不由 runtime 短路。** noop 也是轨迹事件（「看见 X，选择沉默」），Dolores 必须 witness 它。
- **tool 结果不进 memento。** interleaved 下 tool 是 thinking 期的纯交互通道，结果不写轨迹。
- **effort 是 cache 安全旋钮，换模型会破坏 cache。** 只调 reasoningEffort（off/low/medium/high），不换 provider/model——云端 cache 与模型类型相关，换模型 = 全量请求（无 cache 命中）。

## 待接 (Not Yet Wired)

> 见 [dolores-todo.md](dolores-todo.md) 未接能力 W1–W4。

## 打断结算契约 (tool 桥, D30)

MOSS 侧的结果回话与 dsh 侧 tool execute 是**两个方向**，永远可能错位。契约（`dsh_plugin/moss-dolores-ghost-plugin.ts` + `_run.py`）：

- **exit 先结算**：非 yield 的 thinking/exit 在 `agent.cancel` 之前，把仍在 pending 的 tool 全部结算成普通结果 `{interrupted: true, message: ...}`——被打断不再等于 rejected，模型自己决定重拉还是直答。
- **迟到回话被吞**：结算时登记 settled-call 墓碑（TTL 60s），随后到达的 `/tool-result` 回 `200 {dropped}` 而不是 `400 no pending tool call`；abort 监听器 reject 但不删条目。
- **MOSS 侧兜底**：`_dispatch_tool_result` 吸收 RPC 失败（warn + drop）。一次迟到的 moment 结果**永不**能烧掉整轮 logos 流。
- **yield 路径不动**：`wait_next_moment` 仍由下一轮 enter 解锁，exit 不结算、不 cancel。

## Open Problems

> 见 [dolores-todo.md](dolores-todo.md) 设计问题 O1–O3。

## Implementation Notes

- 参照 `ghosts/atom/` 的分文件形态。Dolores 文件结构：`_meta.py`（GhostMeta bootstrapper）/ `_runtime.py`（Ghost runtime）/ `_ego.py`（会话/交易窄桥）/ `_run.py`（交易 run 对象）/ `_tools.py`（强类型 tool 模型）/ `nucleus.py`（自醒 nucleus）/ `_prompts.py`（instruction 分层）。
- 观测三面（`moss-ghost <ghost> --surface tui|output|log`）。
- 决策/探索轨迹（DSH 协议面探索、连续自驱选型、interleaved 候选方案）见 git log：`git log -- src/ghoshell_moss/ghosts/dolores/` 与 dsh-fusion workstream 的 research/。