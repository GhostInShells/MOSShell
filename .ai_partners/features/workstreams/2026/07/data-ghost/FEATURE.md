---
title: Data Ghost
status: draft
priority: P1
created: 2026-07-13
updated: 2026-07-13
depends: [ghost-filesystem-desktop, momento-mori]
milestone: 0.1.0
description: >-
  Data — 第二个 Ghost 原型 (命名取自《星际迷航》的 android Data). 相对 Atom 的
  线性内存历史, Data 把"上下文"拆为 Desktop (现在/作业记忆) + Memento (过去/轨迹)
  两层, 作为 moss 实例 (仓库自身的 ghost) 的载体, 并持续承载高级能力迭代.
---

# Data Ghost

> Use `moss features set-status data-ghost <status> -m "note"` to update state.

## Motivation

Atom 是最简参照基线, 它自己在 docstring 里钉死了两个"原型范围外"的欠落:
context window 不裁剪, 历史纯内存重启即丢. 这两个欠落不该由 Atom 补 — 补了
它就不再是任何人能对照的基线.

Data 是补这两个欠落的**高级层原型**, 同时是 `moss` 实例 (这个仓库自身的 ghost)
的载体. 定位是长期迭代母体: 各种高级能力 (反身控制、mindflow、observability)
会持续接进来. 首批集成的两个 P0 能力:

- **Desktop = 现在**: `articulate()` 不再垂直堆 `message_history`, 而是从机面
  (pin 的表面 + open 的场) 组装 context. 接合点是 Ghost ABC 的 `channel()`
  hook — ghost 反身控制的 channel 以 `'ghost'` 名注册, desktop 的
  open/pin/update 动词挂在这里. Desktop 是**作业记忆**, 不与世界自动同步.
  (对 moss 实例而言, 它的 desktop 大概率就是仓库本身.)
- **Memento = 过去**: 纯内存历史换成 commit 轨迹持久化 — 重启不丢、可化身分叉.
  (对 moss 实例而言, memento 甚至可能直接入库.)

## Key Decisions

<!-- Record each meaningful design choice. This is what the next AI incarnation reads first. -->

- **不碰 Atom.** Atom 保持为纯净对照基线 (单轮 articulate + 纯内存线性历史).
  新能力一律落在 Data 上. 这是命名"第二个原型"而非"扩展 Atom"的根本原因.
- **原型 = Data, 实例 = moss.** 原型型号取自星际迷航的 android Data (求生成人、
  反思型人格; "data" 亦是信息最小单元, 恰配一个代码仓库的 ghost). 实例名 moss —
  这个仓库自身的 ghost, 反身映现整个仓库.
- **上下文双层化 = 本原型的立命之处.** 相对 Atom 的线性 append 历史, Data 的
  context 由 Desktop (现在) + Memento (过去) 组装. 这是 Data 区别于 Atom 的唯一
  硬结构决策, 其余 (mindflow / observability hooks) 都是后续可选迭代.
- **上下文组装不出 runtime.** 信息链路: ghost 的 channel (`Ghost.channel()`) →
  GhostRuntime → shell → 静态面经 MossSystemPrompter 回流 → articulator 带
  moment 回到 ghost. shell 是唯一世界面, ghost 不绕开它另组世界. (否掉过一个
  `assemble_context` hook 提案 — 破坏 shell/ghost 拆分.)
- **一个 ctml tool 统治全部 channel 面.** channel 永不逐个映射为 tool. 真议题
  是反方向: 哪些 tool **不进** channel (判据未决, 见 open problems).
  desktop 修改、bash/mcp/skills 全被此覆盖 — 它们是/将是 channel.
- **1:N articulator:action 原则最优, 本版砍掉.** 理由: 人类和模型都无法颅内
  建模, 需要大家能看懂的方案 (mindflow 已砍过多版). 1:1 保留.
- **think='none' 由 ghost 处理, 不由 runtime 短路.** 现状 ghost_runtime.py:348
  在 effort=='none' 时跳过 articulate — 与 `Impulse.thinking_effort` 字段声明
  ("执行 articulator 的智能体仍有权决定") 矛盾, 且 noop 不进 memento. noop 是
  轨迹事件 ("看见 X, 选择沉默"), Data 必须 witness 它, 否则化身分叉看不见.
- **flash/快响应不进 Ghost API.** 走 Nucleus 侧: 快模型产出 command impulse
  (`Impulse.logos` 反射弧 + `thinking_effort` 建议位已是现成原语). 按需后做,
  不阻塞 Data. 模型配置位现成: `contracts/llms.py` 的
  `DefaultModelTag = 'small_fast_model' | 'flash' | 'pro'`.
- **memento = 标准库件, Ghost 持生命周期 (倾向, 未终决).** 标准实现 ≠ runtime
  拥有: memento 作可复用契约+实现, 各 ghost 在 `__aenter__/__aexit__` 实例化并
  持有. GhostRuntime 对 memento 零感知 (Atom 无, Data 有). 配套: memento channel
  控下轮展示规则 (v1 极简裁剪), 旁路加工做异步精炼 (raw 轨迹全存, 展示走裁剪).
- **thinking 期切片原文不进 Moment.** ghost 自持内存状态, 必要时按 moment
  commit 拆分. `Reaction.executed_logos` ("系统执行的 logos ≠ 模型生成的 logos")
  与 `Reaction.messages` (回声) 已为缝合留好位置, memento 契约
  (contract-frozen) 无需变更.
- **模型层选型: pydantic-ai 现阶段用, 不承诺长期** (对自封装 agent 无兴趣).
  Data 的 `_meta` 不重走 Atom 的 AnthropicModel+环境变量硬编码, 改走
  `contracts/llms.py` 的 LLMConfig 契约.

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

## 真问题与可选方案 (open)

- **tool 结果进不进 moment / articulator 是否支持 outcome** — 全场关键点.
  "写路径汇入同一条 logos 流" 是可选方案 (若采纳, executed_logos 缝合白拿),
  **不是决定**. 不假设调研比施工时看到的准.
- **时序对齐** — ghost 自持的 thinking 期内存状态如何按 moment commit 切分,
  模型 (施工者) 会不会做对. 未验证.
- **thinking 期工具的"纯交互"性 / 切片粒度** — thinking 中调用的能力最好是
  纯交互的; shell 命令结果可等待 (非轮询), 查询经 ctml 未必别扭, "哪些 tool
  不进 channel" 的判据悬而未决.
- **desktop 修改动作的落点** — 若 interleaved 方案成立, 经 ctml tool 顺解;
  若不成立, 首版 desktop read-mostly.

## Implementation Notes

<!-- Gotchas, non-obvious behaviors, reasons for rejecting simpler alternatives. -->

- 参照 `ghosts/atom/` 的分文件形态: `_meta.py` (GhostMeta bootstrapper) +
  `_runtime.py` (Ghost runtime) + `_adapter.py` (Moment↔ModelRequest) + 单测.
- 依赖 desktop (`ghost-filesystem-desktop`) 与 memento (`momento-mori`) 的契约
  就位程度. memento 当前 contract-frozen-pending-review, desktop in-progress
  (channel 落点已定, K14~K18). 起步前先对齐这两条的可用表面.
