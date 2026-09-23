---
date: 2026-09-23
feature: ghost-prototype-dolores
model: deepseek-flash
---

# Dolores 工具面设计 — 双重降级获得升级

> dev2 版本 dolores 的最后一个改造实验。本文记录精确工具面 + 决策依据。
> 最终效果待实机测试确定；做完一次验证后，按实际删减。

## 核心洞察：双重降级获得升级

dsh + deepseek 后训练让模型不能很好理解 output ctml（deepseek v4.1 过拟合 harness，永远输出
markdown），这是重大问题。现在的解法不是对抗，而是**双重降级**：

1. ctml 从「流式 output」降级到「tool 入参」；
2. 进而强制中断 harness 的生命周期（用 cancel 压掉 final answer），把 harness 也降级。

两边降级到同一层，**模型的心智与机制的形态重新平等**。ctml 作为第一公民以这种方式保留——
否则模型仍必须输出一段无用的 final answer。降级的目的就是压掉 final answer。

## final answer 幻觉（致命）

实体机器人交互里，人类不看 final answer 里的 markdown，但模型以为人类看到了，于是产生双方
上下文无法对齐的幻觉。这个幻觉是致命的——模型在对着一个没人看的虚空输出。所以 final answer
必须被 `moss_wait_next_moment` 取代：没有要输出的时候，直接结束回合等下一帧。

## 精确工具面

| tool | 参数（默认） | 语义 | 流式 |
|---|---|---|---|
| `moss_ctml_append` | `ctml` | 独立 articulator 追加，wait compiled 立即继续；interpret error 时 abort thinking | **是（唯一）** |
| `moss_wait_action_done` | `interrupt=false`, `timeout=-1` | wait all actions done + observe；`interrupt=true` 发 replan action | 否 |
| `moss_wait_next_moment` | — | 等所有 action done + cancel_turn，代替 final answer | 否 |
| `moss_react` | `char`, `kwargs=None`, `wait_next_moment=True` | 单字符执行预设 template，command 不 observe | 否 |
| `moss_shell_status` | — | 观测 shell 状态（改名自 moss_observe_status） | 否 |
| `moss_reasoning` | `effort` | 声明思考深度，下一轮生效 | 否 |
| `moss_channel_facade` | `channel_path`, `recursive=True` | 读 channel 操作面，recursive 取代 moss_channels | 否 |

## 决策依据（为什么）

### 1. 流式解析是内核逻辑 fix，不是性能优化

`moss_ctml_append` 是**唯一做流式**的 tool。四个理由，每一条都是「丢了就是丢了」：

- **多重通道语法极其关键**。ctml 的 `chunks__` 等流式通道语法，丢失了它，模型与 Shell 之间
  最细颗粒的实时交互能力就没了。
- **abort thinking**。流式解析下，模型输出到哪个字符有错，哪个字符就立刻中断——观测到
  interpret error 应 abort thinking，模型的输出要被终止。tool 解析则无论如何模型都一次输出完，
  无法中途掐断。这是内核逻辑 fix，不是提速。
- **100k ctml 阻塞**。没有流式解析，模型输出一段 100k 的 ctml 是完全阻塞的，很蠢。
- **partial 编译丢失**。tool use 丢掉编译阶段的 partial，所有可提速逻辑都变慢：特别小的交互
  （say）感知不到，特别大的交互（macro 10k ctml）纯负债。

### 2. moss_channel_facade 用 recursive 合并 moss_channels

ego 持有 shell facade，加 `recursive` 参数等于前缀匹配路径。一个参数合并一个 tool，做起来不难。
`moss_channels`（列全部 channel）被取代。

### 3. moss_react 的极速路径 + cancel flag

快速对话场景（`<say>xxx</say>`）里：模型输出 `moss_react('char')` → wait done → 发 tool 结果 →
cancel 会话。**建模方式：tool call 返回值协议里都带一个 cancel flag**，这样反应速度极快。
原本这个效果应由 final answer 里的 ctml 输出承担——现在 react 接替了它。

**cancel flag 必须带 turn**：界面上也可以 cancel（外部打断），不带 turn 号可能有「下一轮启动
时误 cancel」的错误。带 turn 才能区分「本轮的 react cancel」和「下一轮的正常启动」。

### 4. interrupt 原语与 wait_action_done 的心智成本

interrupt 本身有 ctml 原语。`moss_wait_action_done` 原生加 `interrupt` 参数的动机是**降低模型
心智成本**——模型不用记 ctml 的 interrupt 语法，直接 tool 参数 `interrupt=true`。

`timeout=-1` 是无限等。因为 mindflow 有注意力衰减机制，外部世界有输入时这里不会真的永久卡死。

## mindflow 侧的配套改动（已落地）

- `BaseAction` 新增 `_interpret_error` 字段 + `set_interpret_error()`（同时 set compiled）。
- `wait_compiled(raise_interpret_error=False)` 在 interpret error 时 raise `InterpretError`；
  默认 False 向前兼容。
- `MindflowInShell` 在 ctml 解析出错路径改调 `set_interpret_error` 而非 abort，错误进下一轮 moment。

## 待实机验证

- `moss_ctml_append` 的流式解析 + interpret error abort thinking 是否真实生效。
- `moss_react` 的 cancel flag 极速路径。
- 双重降级后，模型是否不再产生「对虚空输出」的幻觉。
