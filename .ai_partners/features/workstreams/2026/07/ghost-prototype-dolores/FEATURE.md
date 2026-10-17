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
status_note: '2026-10-17 修 compact 头部重建: memory 改由 epoch recap 槽交付 + compact 后开新 epoch.
  _epoch_payload 送达后打戳 (index 单调), memory 先取值缓存, 取消跨 RPC 用 take_compact_memory 补开 epoch,
  plugin 侧 epochIndex 去重 + 新 session 重置 lastEpochIndex/pendingMoments. moss the ghost 实机三轮
  review 硬化 (P0/P1 全收), test 226 passed.'
title: Dolores Ghost
updated: '2026-10-17'
---

# Dolores Ghost

> Use `moss features set-status ghost-prototype-dolores <status> -m "note"` to update state.
> Ground 子任务（ghost_home 认知场装配）→ [dolores-ground.md](dolores-ground.md)。
> ghost_home 治理（场拓扑 + 两档可见性 + journal 时间线）→ [dolores-ghost-home-governance.md](dolores-ghost-home-governance.md)。
> Ego 装线 dogfood 评审与下一步 → [dolores-ego-wiring.md](dolores-ego-wiring.md)。
> **旁路单轮机制（perStep 第二阶段）— 下一步实现** → [dolores-reentrant-ego-session.md](dolores-reentrant-ego-session.md)。
> Dolores × Memento 全貌（commit / note / compact / branch）→ [dolores-memento-plan.md](dolores-memento-plan.md)。
> commit / compact ego session 的 dsh 机制判定 → [dolores-commit-compact-ego-session.md](dolores-commit-compact-ego-session.md)。
> ego 注册方案（单 plugin + session-start，推翻 agentPreset 拆分）→ [dolores-ego-plugin-split.md](dolores-ego-plugin-split.md)。
> ego 专属 preset + agent 级 nibble（非 ego session 可用，推翻 doloresSelectionRef）→ [dolores-ego-preset-nibble.md](dolores-ego-preset-nibble.md)。
> 自迭代能力地图（下阶段 ground + 认知工具）→ [dolores-self-iteration-map.md](dolores-self-iteration-map.md)。
> 全链路实机运行复盘（prompt 反转释放焦虑 + milestone 幻觉纠正）→ [dolores-full-chain-live-run-retro.md](dolores-full-chain-live-run-retro.md)。
> 工具面设计（双重降级获得升级 + 精确工具面 + 决策依据）→ [dolores-tool-surface.md](dolores-tool-surface.md)。
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

## 人喊停的可达性契约 (D04)

**命题**：人在 dsh 界面点停时，物理身体必须在**点击那一刻**停，而不是等这一轮自己走完。

- **修前实测（2026-10-04 00:29）**：`moss_react` 已被 dsh 正确中断，语音却 **41.3 秒播满**。三处根因：
  ① 豁免按 **turn 号**记 —— `moss_react` / `moss_wait_next` 的结果**常态**带 `cancel=True`，等于把每一轮都
  记成"自己人"，人点的 `user` abort 被一起吞掉；② `logos()` **单线程顺序**消费，`_handle_react` 的
  `wait_actions_done()` 正占着循环等身体跑完，turn/end 排队 41 秒才被消费 —— 信号到了没人开门；
  ③ `on_session_event("*")` 把 host 内环工具（bash / read / write / edit / subagent）也当"等我们回答的
  tool"，warning 刷屏，把真正的 D39 信号埋掉。
- **修法（P1–P4）**：中断类 abort 挪到**入队前**（`_predispatch_abort`，同步幂等，不等消费）；
  判据挂 **cause**（`hook` = 我们自己的 hook cancel 才豁免，`user` / 无 cause / 其它 = 中断，`error` 永远中断）；
  事件按 `moss*` 命名空间过滤；action 退出改 `interpreter.close(cancel_managing=True)`（只清本动作范围持有的
  命令），`shell.clear()` 收在 thinking 退出那一处 —— 那是"整具身体"的粒度，多路 interpreter 时不会误伤别的轨道。
- **配套内核修正**：`abort_reason: str | None`（对齐 `Future.cancel(msg)`，reason 不当词汇表用）；
  `cancel_executing` → `cancel_managing`（这个参数管的是 parsed tasks，不是"执行"本身）。
- **修后实测（2026-10-04 01:06）**：语音在 3.1 秒处被掐断；探针早于消费者 20ms，整树 clear 在 22ms 内完成。
- **未闭环**：`moss_react` tool 入口**本身**仍未实测（复现走的是 `moss_interpret` + `moss_wait_next`，
  死锁形状相同）；`_abort_clear` 的 interpreter 级 close 还没有 mindflow 集成用例。
  回归入口：`.ai_partners/features/regressions/dolores-ghost-prototype/REGRESSION.md`（TC-000..005）。

## observe 续帧契约 (D25)

`need_observe` 的生命周期：命令执行完 → `InterpreterStoppedEvent(need_observe=True)` → `shell_trajectory.when_need_observe` → `moments.add_echoes([], need_observe=True)` → 帧循环 `while need_observe()` 生成**回声续帧**。续帧是 ghost 的**自我延续**（回看自己刚做过什么），往往没有 percepts —— 这正是它与「外部输入」的分界，也是曾经被误判的地方：

- **`needsObserve` 标记**：`_ego.needs_observe()` 从 `moment.previous.need_observe` 判定这一帧是不是续帧，随 `thinking/enter` payload 传入 plugin。
- **续帧必须自己开一轮**：plugin 在 `inputs` 为空且 `needsObserve` 时用 **steer** 开 turn（不能用 inject —— inject 只投递不唤醒，此刻没有在跑的 step，帧会一直躺着）。早先「inputs 为空不起 turn」的规则本意是不为「无 percepts」造占位，对自我延续不成立。
- **同一条根因的另外两个现象**：续帧滞留 `pendingMoments` → 下一轮开头帧顺序错乱；fetch 的 moment 同样落进这批缓冲 → 迟到一轮。修 D25 一并解掉。

## Open Problems

> 见 [dolores-todo.md](dolores-todo.md) 设计问题 O1–O3。

## Implementation Notes

- 参照 `ghosts/atom/` 的分文件形态。Dolores 文件结构：`_meta.py`（GhostMeta bootstrapper）/ `_runtime.py`（Ghost runtime）/ `_ego.py`（会话/交易窄桥）/ `_run.py`（交易 run 对象）/ `_tools.py`（强类型 tool 模型）/ `nucleus.py`（自醒 nucleus）/ `_prompts.py`（instruction 分层）。
- 观测三面（`moss-ghost <ghost> --surface tui|output|log`）。
- 决策/探索轨迹（DSH 协议面探索、连续自驱选型、interleaved 候选方案）见 git log：`git log -- src/ghoshell_moss/ghosts/dolores/` 与 dsh-fusion workstream 的 research/。