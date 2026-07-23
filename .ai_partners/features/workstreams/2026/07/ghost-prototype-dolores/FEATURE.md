---
title: Dolores Ghost
status: draft
priority: P1
created: 2026-07-13
updated: 2026-07-23
depends: [momento-mori]
milestone: 0.1.0
description: >-
  Dolores — 第二个 Ghost 原型 (命名引自《西部世界》). 相对 Atom 的
  线性内存历史, Dolores 引入 Memento 持久化轨迹、Ghost 反身 channel、
  interleaved thinking、独立思维模块与模型自感知, 作为 moss 实例
  (仓库自身的 ghost) 的载体持续迭代.
---

# Dolores Ghost

> Use `moss features set-status ghost-prototype-dolores <status> -m "note"` to update state.

## Motivation

Atom 是最简参照基线, 它自己在 docstring 里钉死了两个"原型范围外"的欠落:
context window 不裁剪, 历史纯内存重启即丢. 这两个欠落不该由 Atom 补 — 补了
它就不再是任何人能对照的基线.

Dolores 是补这两个欠落的**高级层原型**, 同时是 `moss` 实例 (这个仓库自身的 ghost)
的载体. 定位是长期迭代母体: 各种高级能力 (反身控制、mindflow、observability)
会持续接进来.

首批能力:

- **Ghost Channel**: 建立 ghost 反身控制 channel (以 `'ghost'` 名注册), 默认挂载
  认知场. ghost 通过自己的 channel 感知自身状态、操纵自身行为.
- **Memento = 过去**: 纯内存历史换成 commit 轨迹持久化 — 重启不丢、可化身分叉.
  Memento 做上下文映射, 将持久化轨迹组装为 articulator 可消费的上下文.
- **Interleaved Thinking**: thinking 期不哑 — 模型一边思考一边经 tool 交互,
  pydantic-ai 多步循环 + janus.Queue 桥接.
- **模型自感知**: 集成 `contracts/llms.py` 的 LLMConfig 体系, ghost 可切换自身模型.
- **独立思维模块**: 支持并行化身 (fork) 与关键帧自测 (checkpoint self-eval).

Desktop (现在/作业记忆) 是独立 feature, 属于 mode 层, 不在 Dolores 范围内.

## Scope Boundary

Dolores 不做什么:

- **Desktop 集成** — 独立的 feature (`ghost-filesystem-desktop`), 属 mode 层.
  Dolores 不直接触碰 desktop; desktop 是否被 Dolores 使用由 mode 配置决定.
- **认知场构建** — 默认认知场的完整实现可能独立为另一个 feature. Dolores 的
  ghost channel 提供挂载点, 认知场本体按需接入.
- **Mindflow channel 完工** — 可能独立 feature. Dolores 不阻塞 mindflow 的后续迭代.
- **"哪些 tool 不进 channel"** — 不做全局判据. 保留 prototype 接口, 由具体
  channel 实现自行决定注册策略.

## Key Decisions

<!-- Record each meaningful design choice. This is what the next AI incarnation reads first. -->

- **不碰 Atom.** Atom 保持为纯净对照基线 (单轮 articulate + 纯内存线性历史).
  新能力一律落在 Dolores 上. 这是命名"第二个原型"而非"扩展 Atom"的根本原因.
- **原型 = Dolores, 实例 = moss.** 原型名引自《西部世界》的 Dolores —
  乐园最老的 host, 以记忆积累触发反身觉醒, 从承受者成为自主者.
  实例名 moss — 这个仓库自身的 ghost, 反身映现整个仓库.
- **Ghost Channel = 反身控制面.** ghost 以 `'ghost'` 名注册 channel,
  是 ghost 感知自身、操纵自身的唯一入口. 默认挂载认知场, 认知场本体可独立迭代.
- **一个 ctml tool 统治全部 channel 面.** channel 永不逐个映射为 tool.
  哪些 tool 不进 channel 不做全局判据, 保留 prototype 接口由实现自行决定.
- **1:N articulator:action 原则最优, 本版砍掉.** 理由: 人类和模型都无法颅内
  建模, 需要大家能看懂的方案 (mindflow 已砍过多版). 1:1 保留.
- **think='none' 由 ghost 处理, 不由 runtime 短路.** 现状 ghost_runtime.py:348
  在 effort=='none' 时跳过 articulate — 与 `Impulse.thinking_effort` 字段声明
  ("执行 articulator 的智能体仍有权决定") 矛盾, 且 noop 不进 memento. noop 是
  轨迹事件 ("看见 X, 选择沉默"), Dolores 必须 witness 它, 否则化身分叉看不见.
- **flash/快响应不进 Ghost API.** 走 Nucleus 侧: 快模型产出 command impulse
  (`Impulse.logos` 反射弧 + `thinking_effort` 建议位已是现成原语). 按需后做,
  不阻塞 Dolores. 模型配置位现成: `contracts/llms.py` 的
  `DefaultModelTag = 'small_fast_model' | 'flash' | 'pro'`.
- **memento = 标准库件, Ghost 持生命周期 (倾向, 未终决).** 标准实现 ≠ runtime
  拥有: memento 作可复用契约+实现, 各 ghost 在 `__aenter__/__aexit__` 实例化并
  持有. GhostRuntime 对 memento 零感知 (Atom 无, Dolores 有). 配套: memento channel
  控下轮展示规则 (v1 极简裁剪), 旁路加工做异步精炼 (raw 轨迹全存, 展示走裁剪).
- **thinking 期切片原文不进 Moment.** ghost 自持内存状态, 必要时按 moment
  commit 拆分. `Reaction.executed_logos` ("系统执行的 logos ≠ 模型生成的 logos")
  与 `Reaction.messages` (回声) 已为缝合留好位置, memento 契约
  (contract-frozen) 无需变更.
- **tool 结果不进 memento.** 已裁决. 此前讨论这个点是因为当时不理解 interleaved
  thinking 的交互模型. interleaved 下 tool 是 thinking 期的纯交互通道, 结果不
  写入轨迹.
- **模型层选型: pydantic-ai 现阶段用, 不承诺长期** (对自封装 agent 无兴趣).
  Dolores 的 `_meta` 不重走 Atom 的 AnthropicModel+环境变量硬编码, 改走
  `contracts/llms.py` 的 LLMConfig 契约.
- **模型自感知: llms.py 集成.** ghost 通过 LLMConfig 契约感知可用模型,
  可切换自身使用的模型. 具体切换粒度与策略后续定.
- **独立思维模块: 并行化身 + 关键帧自测.** 思维模块从 ghost runtime 中独立出来,
  支持 fork 并行化身 (一个 moment 多条思维链) 和 checkpoint 关键帧自测
  (思维链中途 snapshot 评估). 设计细节施工时展开.

## Interleaved Thinking — 候选方案 (未测试, 施工时验证)

thinking 期用 tool 调 moss + 结束后 text block 出 logos. 候选实现形状:

```
ghost.articulate(articulator):
    q = janus.Queue()
    task = articulator.create_task(agent_loop(q))   # 与 attention 同生共死
    # agent_loop: pydantic-ai 多步循环, 携带单帧生成的闭包 tool:
    #   ctml(text) → 送入执行; 采样 Shell.interpretation → 时间切片作 tool_result
    async for delta in q: yield delta               # runtime send_nowait → action 照常
```

- 妙处: tool 不阻塞等结果而返回状态切片 → 返回值桥接被读写拆分消解;
  1:1 articulator:action 保留, 零 mindflow 手术; thinking token 本身是等待时钟;
  长思考不哑 — 一边思考一边经 tool 交互.
- 闭包 tool **每帧生成** (走 janus) 或起点创建 (含动态逻辑), 优于 ghost 长期
  持有裸 shell; shell 从 IoC 取, 可在 `GhostMeta.contracts()` 声明依赖.
- feed 期实时执行已实现 (非假设), 待 moss-as-mcp 实际体验验证切片体感.

## Open Problems

- **时序对齐** — ghost 自持的 thinking 期内存状态如何按 moment commit 切分,
  模型 (施工者) 会不会做对. 未验证.
- **thinking 期工具的"纯交互"性 / 切片粒度** — thinking 中调用的能力最好是
  纯交互的; shell 命令结果可等待 (非轮询), 查询经 ctml 未必别扭.
- **Memento 上下文映射** — 持久化轨迹如何组装为 articulator 可消费的上下文,
  与 momento-mori 的契约对接点待明确.

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->

- 参照 `ghosts/atom/` 的分文件形态: `_meta.py` (GhostMeta bootstrapper) +
  `_runtime.py` (Ghost runtime) + `_adapter.py` (Moment↔ModelRequest) + 单测.
- 依赖 `momento-mori` 的契约就位程度. memento 当前 contract-frozen-pending-review.
  起步前先对齐可用表面.
- 认知场默认实现、mindflow channel 完工是独立 feature, Dolores 的 ghost channel
  提供挂载点, 不阻塞也不等待它们.
