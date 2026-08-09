---
title: Dolores Ghost
status: draft
priority: P1
created: 2026-07-13
updated: 2026-08-09
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
- **认知场构建** — Ground 协议由 `ghost-ground` feature 提供通用基础设施.
  Dolores 只做 ghost_home 认知场的装配和默认内容, 不做 ground 协议本身.
- **Mindflow channel 完工** — 可能独立 feature. Dolores 不阻塞 mindflow 的后续迭代.
- **"哪些 tool 不进 channel"** — 不做全局判据. 保留 prototype 接口, 由具体
  channel 实现自行决定注册策略.

## Ghost Home Ground

Dolores 的认知场 = ghost_home 目录 + Ground 协议. ghost_home 是 ghost 自身的
认知基建根目录, 挂载在 ghost channel 的 `ground` 子路径上.

### 双 GroundSet 架构

Dolores 持有两个 GroundSet:

| GroundSet | 根 | 何时使用 | 场是 |
|---|---|---|---|
| **ghost_home** | ghost 自身认知目录 | 始终存在, 默认 | skills / memory / experience |
| **project_root** | 被操作项目的根 | `--mode` 决定 | features / .design / .discuss |

两个 GroundSet 平级不嵌套. project_root 由 mode 提供 (如 `--mode meta` 时是
MOSS 仓库本身), ghost_home 始终是 ghost 的默认面. ghost channel 负责管理
当前注意力落在哪个 GroundSet 的哪个场上.

### ghost_home 目录结构

```
ghost_home/
  GROUND.md                    # ghost 自身认知入口 (L0)
  skills/                      # Claude-compatible skills 范式
  memory/                      # 大记忆体系
    existential/               # 存在主义总结 — 我是谁, 我的价值观
    temporal/                   # 时态摘选 — 年/月/周/日 分层
  experience/                  # 经验机制 (project-level 场景经验)
    L1/                        # 两层渐进式披露 — 索引层
      ...                      # 详情层
  .grounds/                    # ghost 自身模板
```

### 场上挂载

Ghost channel 的 `ground` 子路径提供:

```
ghost.ground
  ├── open / close / reopen    # 场开合 (两个 GroundSet 间切换)
  ├── pin / unpin / update     # 注视操作
  ├── frame / observe          # 诊断
  └── <label>                  # 每个 opened ground = command-less virtual channel
        instruction = 法链
        context_messages = 帧
```

### 化身与认知自迭代

活数据 (perspectives + memento + inputs) 每次运行产生. 上下文组装是独立化身
接口 — 不同 mode/user/task 从同一份活数据组装不同上下文. Worktree fork 构建
新化身, 复用活数据, 验证不同行为模式:

```
fork → worktree 隔离化身 → 并行运行 → snapshot → compare → 学习
```

这是 Dolores 独立思维模块 (并行化身 + 关键帧自测) 的地基.

### 与 MOSS Project Ground 的关系

`moss-project-ground` feature 定义 MOSS 项目自身的 GROUND.md (项目根,
features/designs/specs 寻路). Dolores 在 `--mode meta` 下实例化两个
GroundSet: ghost_home (自身认知) + project_root (MOSS 项目认知).
在其他 mode 下 project_root 指向被操作的项目.

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
- **模型自感知: _llms 模块 + ghost.model channel.** Dolores 内建 `_llms` 模块,
  封装 LLMConfig 的查询与切换. 通过 ghost channel 以 `ghost.model` 路径挂载,
  暴露以下能力:

  - **current-model**: 返回当前 `ResolvedModel` (provider, model name, context_window)
  - **list-models**: 返回 `LLMConfig.list_models()`, 列出所有可切换模型
  - **switch-model**: `ghost._meta` 更新持有的 `ResolvedModel`, 下一帧 `articulate()` 生效.
    走 `LLMConfig.get_model()` 默认 fallback 到 default, 不怕误配
  - **window-status**: 自省窗口压力 — context_window 上限 + 上一帧实际 token 用量 +
    剩余预算. 运行时数据来自 adapter (输入侧 moment 组装的 token 数) 与 articulator
    (API response 的 usage 字段)

  切换本身是简单赋值, 回归周期预算小. 窗口自省让 ghost 知道自己"还剩多少",
  自主决定裁剪或切换.
- **独立思维模块: 并行化身 + 关键帧自测.** 思维模块从 ghost runtime 中独立出来,
  支持 fork 并行化身 (一个 moment 多条思维链) 和 checkpoint 关键帧自测
  (思维链中途 snapshot 评估). 设计细节施工时展开.

## 连续自驱 — 醒×续 选型与故障恢复 (2026-08-09 讨论收敛)

> 人类架构师 + 模型 (deepseek-v4-flash) 的连续自驱选型讨论. 自驱是一期核心命题,
> 本节记录收敛轨迹, 施工时以此为准, 细节在开发中展开.

### 选型: 醒 × 续 两维度

自驱不是单一命题, 是 **醒 (wake) × 续 (continue) 两维度的乘积**:

| 维度 | 问题 | 一期方案 |
|---|---|---|
| 醒 | 下一轮思考从哪来 | task nucleus 低优 signal 驱动 + 状态保留 |
| 续 | 同轮 articulate 能否不中断 | articulator 锁死权限命令 |

- **醒 = task nucleus (方案4).** 自驱的本质是"有任务在推进", 状态保留在 nucleus
  里 = 自驱有内存. IdleNucleus (方案1) 是无状态退化形, 被 task nucleus 超集覆盖
  (空闲 = 一个低优 idle task). 最优雅的形式是 topic/signal + 自定义 loop 逻辑,
  但一期关心的不是默认实现, 是 ghost 可修改的语法本身.
- **续 = articulator 锁死权限命令 (方案3).** agent 用 command 控制自注意力提权、
  故障验证状态、或抛超异常让机体 pause (pause 模块已做). **不动 articulator 生命周期,
  自驱命令是 channel 表面的一份子** — 这正是 "ghost 可以修改的语法本身". 一期可
  只提供语法, 不开放 ghost 自改自己.
- **砍掉 next() flag (方案2).** 与 raise_observe 语义重叠, 是 attention 内推进的
  细粒度操控, 不是自驱时钟. 一期不做 attention 接口手术.
- **反身语法必须机械到不需要预训练.** 模型预训练分布里没有"操纵自己注意力仲裁器"
  的样本, 反身操作认知负担极高. 语法必须极简、语义直接 (屏蔽/降权/关掉), 模型只
  调用, 不理解内部. 一期只做 pull 容错 (低反身), 屏蔽语法留二期.

### 故障恢复与快速 compact

- **快速 compact 前提 = commit 持续提交 × compact agent 分段输入.** memento §12
  里 compact 本质是移动 detail cursor; commit 持续提交让 compact agent 每轮只处理
  最近一小段 staging. 两个前提互相咬合, 只提 commit 侧不完整.
- **致命故障 (弱网/tokens 超标) 需要 ghost runtime 级保护**, 优先级层级:
  `estop > retry (免疫普通 impulse, 不免疫 estop) > normal impulse preemption`.
  estop 连 retry 都能停, 否则 estop 失效.
- **上下文崩溃是最致命故障** — 协议脏数据导致无法重新进入 agent 调用. 办法:
  压缩所有 commit, 故障轮 commit 以故障方式压缩掉, 不允许重载展开.

### task 状态机

标准系统收敛到同一核心: `pending → running → {done | failed | cancelled}`.
现成证据: CommandTaskState (command.py:87) + MCP tasks (SEP-2663).

MOSS 两条独有轴 (别家没有):

1. **waiting (input_required)** — 双工态. task 执行中需要等外部输入 (端侧数据 /
   ghost 决策). task 在等"下一轮思考的输入"时就是 waiting.
2. **interrupted (preempted)** — 抢占暂停态, ≠ cancelled, 可恢复. mindflow 特有,
   别家无抢占故无此态. **interrupted 永远不是终态.**

- done 是唯一干净终态; failed/cancelled 是 dirty 终态.
- retryable 是 task 的**字段 (policy 不是 state)**, 不是新状态 — 弱网重试与普通
  失败靠它区分.
- **task ack 走 matrix 协议** (matrix-operator 的 service kind). task 状态活在
  mesh 上 (zenoh query/pub-sub), ghost 是投影消费者; ack 投递/确认/持久化是矩阵
  职责, ghost 自身状态异常时 ack 不丢. nucleus 与端侧共享同一协议.

### 坏 impulse 落点: perspectives, 不是 percepts

坏 impulse 是 **mindflow 内部状态异常**, 不是外部世界事件. 落点必须区分:

- **percepts** = 外部输入, source-keyed. 放这里会诱导 ghost "收到外部信号, 该
  响应" — 但 ghost 无运行时修改 mindflow 的能力 (重启不了), 会被诱导去做做不到
  的事.
- **perspectives** = 系统层内观快照 (moss_dynamic / safemode / Mindflow 自解释).
  放这里正确传达: "感知系统里有异常, 认知到它存在, 但这不是要响应的外部事件,
  你改变不了它." 与 mindflow-channel "自解释走 instruction 不走 context_messages"
  同构.

**两层设计** (不冲突):
- 认知层: 坏 impulse 帧内可见 (perspective), 不落盘 (`Moment.for_saving` 清空
  perspectives).
- 轨迹层: 坏 impulse 升级为致命故障 (articulator 崩溃/超时/协议脏数据) 才落
  故障 commit (L3 + faulted, 可文本读不可上下文展开).

### 与既有决策的关系

- 互不矛盾: interleaved thinking (thinking 期不哑) 是"续"的候选实现形态;
  mindflow-channel 提供感知反身面; memento 提供快速 compact 前提.
- 本次讨论把自驱从"一个命题"拆成"醒 × 续"两个独立维度, 落点分别在 task nucleus
  和 articulator 命令面.

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

- 参照 `ghosts/atom/` 的分文件形态. Dolores 的文件结构预期:

  - `_meta.py` — GhostMeta bootstrapper (LLMConfig 集成)
  - `_runtime.py` — Ghost runtime (Memento + interleaved thinking)
  - `_adapter.py` — Moment↔ModelRequest
  - `_llms.py` — 模型自感知模块 (LLMConfig 查询/切换/窗口自省), 以 `ghost.model` 路径挂入 ghost channel
  - 单测
- 依赖 `momento-mori` 的契约就位程度. memento 当前 contract-frozen-pending-review.
  起步前先对齐可用表面.
- 认知场默认实现、mindflow channel 完工是独立 feature, Dolores 的 ghost channel
  提供挂载点, 不阻塞也不等待它们.
