---
created: 2026-07-27
depends:
- interleaved-ctml-thinking
- mindflow-control-semantics
- audio-capture
description: 前瞻方案锁定：当实时语音模型的工具调用足够强时，MOSS 如何把纯异步交互逻辑收成 realtime 音频模型的上行 user 输入，用
  interleaved CTML + mindflow 硬打断实现实时理解。
milestone: null
priority: P2
status: design-locked
status_note: 前瞻方案锁定：行业判据未满足，落码时机未到
title: Realtime Voice Interaction Logos — 音频为第一公民的流式知觉 + CTML 控制
updated: '2026-07-28'
---

# Realtime Voice Interaction Logos

> Use `moss features set-status realtime-voice-interaction-logos <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## 本文件的性质

**这是一份前瞻设计锁定，不是待实施的开发计划。** 依赖的行业前提（实时语音模型的工具
调用能力）当前不成立，落码时机未定。存在的意义是：MOSS 的实时交互路线在行业方案成熟
**之前**就已推演完毕，方案有据可查，而不是等别人做出来了再追。

不要因为 status 是 draft 就去推进实现。推进的前提是 §2 的行业判据被满足。

## Motivation

MOSS 的目标是 Ghost 在现实世界里活着——感知、思考、行动同时发生。当前 shell 侧的
右循环（action + result）已由 `interleaved-ctml-thinking` 关闭，但左循环（输入/感知）
在实时音频场景下没有载体：Ghost 能在思考中铺执行轨，却不能在说话的同时持续听。

行业出现的 realtime 语音模型形态提供了一条可能的感知侧载体。本 workstream 锁定的核心
构想：**把本身纯异步的交互逻辑，转化为 realtime 音频交互模型的上行 user 输入**。音频
只上行，模型的输出仍是 CTML；mindflow 在输入层做硬打断。这样实时交互中基于 CTML 的
理解就成立了——不需要等一个"能同时处理音频与工具调用的原生模型"出现。

## 1. 三层概念区分（最重要的一条，先立）

三个东西名字接近、极易被后续实例混为一谈。**它们不是同一件事的三个成熟度，是三个
不同的对象：**

| 概念 | 是什么 | 载体 | 状态 |
|------|--------|------|------|
| **interleaved logos thinking** | 思考流与 CTML 交错，模型是铺轨写头。只治右循环 | 回合制（MCP 验证） | 已实现，`interleaved-ctml-thinking` completed |
| **realtime-voice-interaction-logos** | 音频为第一公民的流式知觉 + CTML 控制。感知侧由 realtime 语音模型承载 | 流式音频 session | 本 workstream，方案锁定 |
| **duplex multi-modalities thinking model** | MOSS 真正希望出现的东西：原生多模态双工思考模型 | 尚不存在 | 终局，非本 workstream 范围 |

**API 可以部分借鉴，语义不可等同。** 上一层的动词面（append / exec / observe /
replan / interrupt）在本层有对应物，但对应关系不是恒等——见 §4。把 interleaved 的
API 直接套过来，就是把回合制假设烧进流式协议，这是 `interleaved-ctml-thinking` K7
已经显式拒绝过的错误。

## 2. 行业前提与判据（2026-07 调研结论）

**当前行业没有满足 MOSS 需求的方案。** 这不是保守估计，是按定义裁决的结果。

### 2.1 "realtime" 一词被滥用

人类工程师 2023 年调研 OpenAI Realtime 的那一版，本质是**云端决策的回合制，外面套了
流式音频 buffer**。到 2026-07 为止形态未变：server VAD 检测到用户说话 → cancel 当前
response → 起新 response。模型并没有在自己输出期间持续对上行流做建模。

**判据（三条全中才配叫 realtime）**：

1. 模型有**真实的流式输入知觉**——不是把音频攒成一段再转文本喂进去。
2. 服务端 **session 化**，持续接受输入。
3. 服务端在模型输出期间**持续做打断决策**，快速响应。

按此判据：OpenAI Realtime / Gemini Live 均不满足第 3 条（打断是取消 response，不是
持续决策）。字节 Seed 的 full-duplex speech LLM 方向满足 1–3，但不暴露工具调用。

### 2.2 智力与工具调用的双重不足（当前最大不确定性）

即使 session 化 + 流式知觉齐备，realtime 语音模型的智力可能**两头都不够**：

- 不能与人类的不同对话风格做好自由对话；
- 不能精确调用工具、达到长程思考的效果。

这两项不足是独立的，可能同时存在。**没有视觉时，realtime 语音模型未必能满足 MOSS 的
要求**——概念接近了，能力未必到。

### 2.3 由此产生的架构分叉（关键待决）

若 §2.2 的不足成立，则需要**thinking 层并行脑做工具调用**：realtime 语音模型负责
感知与出声（感知侧 + 发声），另一个强推理模型负责 CTML 生成（执行侧），两者在 mindflow
里竞争注意力。

这是本 workstream 最重要的未决分叉：

| 形态 | 前提 | 代价 |
|------|------|------|
| **单脑** | realtime 模型工具调用足够强 | 无额外架构；但前提当前不成立 |
| **双脑并行** | realtime 模型只够做感知/出声 | 两脑的时间轴对齐、注意力仲裁、谁持有 CTML 写头 |

**不预先裁决。** 双脑形态的代价集中在"裂脑"风险上——mindflow 设计注释里已经点名
（"避免思维奔逸和裂脑：感知/思考/行为消费不同时间轴上的信息"）。若走双脑，这条注释
从警告升级为主要矛盾。

## 3. 核心构想：异步交互逻辑收成上行 user 输入

MOSS 里大量交互逻辑天然是异步的——channel 上线、task 完成、感知帧到达、后台失败。
在回合制协议下它们只能等模型下一次主动来看（`interleaved-ctml-thinking` K5 的"拉侧"）。

realtime 载体提供的转化：**这些异步事件成为 realtime session 的上行 user turn**。

```
[异步世界]                    [上行通道]              [realtime session]
task done / critical fail  ─┐
channel 上线 / state 切换  ─┼─→ mindflow 仲裁 ─→ 上行 user 输入 ─→ 模型
音频帧 (只上行)            ─┤    (Priority 抢占)
感知帧 (vision 等)         ─┘
                                                            │
                                     CTML (下行) ←───────────┘
```

三条纪律：

1. **音频只上行。** 模型的输出不走音频回下行链路做 TTS 环回；下行是 CTML。语音的
   发声由 CTML 命令驱动（channel 层），不由 realtime API 的 audio output 直接承担。
   这样"说什么"仍在 CTML 的时间语义之内，可被 interrupt 掐断。
2. **上行是投影，不是转发。** 异步事件经 `project_events` 同族的投影规则收敛成消息，
   而不是原样灌进 session。K5 的三时钟分类（执行游标=拉、能力 meta=推、感知帧=激进
   drain）在此仍然成立且更关键——realtime session 的上行是稀缺信道。
3. **硬打断在输入层。** 见 §5。

## 4. 与 interleaved 五动词的对应关系（借鉴而非等同）

| interleaved 动词 | realtime 层的对应物 | 是否等同 |
|---|---|---|
| `ctml_append` | 默认介质。流式下模型持续铺轨，无需动词 | **不等同**——append 在回合制里是一次调用，流式里是介质底色 |
| `ctml_exec` | 无对应。"阻塞至完成"在流式里是反模式 | **删除** |
| `ctml_observe` | 退化。上行推送已让游标持续到达，无需主动拉 | **大幅弱化**，保留作为 budget 内的显式等待 |
| `ctml_replan` | 保留。模型自己的思维剪枝 | 等同 |
| `ctml_interrupt` | 保留，但语义分裂为二 —— 见 §5 | **不等同** |

**关键翻转：`interleaved-ctml-thinking` B4 记录的"MCP 反转写头/读头时钟比"在此被修正
回来。** MCP 里 append 回合延迟 15–20s、执行 ~3s，游标恒跑在笔尖前，K1 的核心张力
（笔尖跑在游标前）验不了。流式载体下时钟比是正的，**K1 张力第一次能真实复现**。
这意味着 interleaved 阶段所有"在 MCP 上验不了"的结论，在本层才第一次进入可验证状态。

## 5. 硬打断的语义分裂（本 workstream 的第一个新问题）

interleaved 侧的 `interrupt` 与本层要的"输入层硬打断"**决策主体不同**：

| | interleaved `interrupt` | 输入层硬打断 |
|---|---|---|
| 决策主体 | 模型自己（"actually, I'm just going to..."） | 外部（人类插话 / 高优信号） |
| 触发时机 | 模型的决策时钟 | 世界时钟 |
| 授权 | 无需——自己的轨自己掐 | 需要（谁能打断 Ghost？SafeMode 邻域） |
| 落点 | 同为 `shell.clear()` 掐 pending | 同 |

两者收敛到同一个内核动作，但**授权、时机、以及"被打断后模型如何知道"三件事完全不同**。

### 5.1 已知缺口：清洁停止在投影里不可见

`host/interleaved_thinking.py:252` `on_interpreter_stopped` **只在有 exception 时**
把 `InterpreterStopped` 入 buffer。清洁停止不生成事件（当时的理由正确：清洁停止的语义
由 `wait_interpreter_done` 表现，不需要噪音事件）。

但外部硬打断若走清洁停止路径，**模型在投影里看不到"我被打断了"这件事发生过**。
这是本层必须新增的事件类型，不是既有实现的 bug——是边界外移后暴露的新需求。

### 5.2 mindflow 侧已有的地基

`core/blueprint/mindflow` 里以下要件已就位，硬打断不需要新造仲裁机制：

- `Priority` 枚举：`BACKGROUND`(-1) 永不抢占成功 ↔ `FATAL`(5) 永远抢占成功。
- `Attention` / `Articulate` / `Action` 三循环状态调度。
- `AttentionAbortedError` / `ActionAbortedError` / `ArticulateAbortedError` /
  `PreemptedElseSuppress` / `ImpulseAbsorbed` —— 打断信号族齐全。
- `Signal` → `Impulse` → `Nucleus` 的感知隔离建模。

人类插话作为 `InputSignalMeta` 进来，按 Priority 抢占当前 attention，这条路是通的。
**缺的不是仲裁，是"抢占成功后 shell 侧怎么被通知、模型怎么看见"。**

## 6. 实施缺口（来自 interleaved 的已知待补项 + 本层新缺口）

以下均来自 `interleaved-ctml-thinking` FEATURE.md 的"残留待验"段 + 本层分析。
**不在本阶段修复**——本 workstream 锁定设计，不落码。但需列出，避免下个实施的实例
遗漏。

### 6.1 来自 interleaved 的待补项（执行侧基础设施）

| # | 缺口 | 来源 | 对 realtime 的关键性 |
|---|---|---|---|
| G1 | progress 活串未投影。`status` 只给 `ongoing_callers` 不给 progress | FEATURE.md L413 | **高**——硬打断的时机决策依赖"轨道跑到哪了" |
| G2 | fail-closed 写拒未实现。append 遇 critical failure 应拒 push | FEATURE.md L409 | **高**——单脑形态下这是唯一互锁触发点 |
| G3 | channel 上线推通道未做。`get_moss_dynamic_info` 只拉不推 | FEATURE.md L411 | 中——长 thinking 内感知不到 channel 变更 |
| G4 | `on_task_done` 不唤醒 waiter，注释已预留"未来若需要 wait-any-event（全双工推送），用独立的 waiter 集合" | `interleaved_thinking.py:249` | **高**——推侧触发点缺失，是前一个实例留给本层的挂点 |
| G5 | 编译期 "did you mean" 纠错 | FEATURE.md L413 | 低 |

### 6.2 本层新缺口

| # | 缺口 | 说明 |
|---|---|---|
| G6 | 外部打断事件类型。`InterpreterStopped` 只在有 exception 时进 buffer，清洁停止不可见 → 需要新事件或新投影规则 | §5.1 |
| G7 | mindflow 抢占 → shell 通知通路。信号族齐全但抢占成功后通知模型侧的投影机制未设计 | §5.2 |
| G8 | 音频只上行的 channel 层。耳 = 音频 signal → mindflow Nucleus；口 = CTML 驱动的 TTS channel，不进 realtime API 的下行音频 | §3 纪律 1 |
| G9 | session 化服务端设计。realtime session 在多 Ghost / 多模态间复用 vs 独立，生命周期对齐 Matrix session 还是独立 | 架构待定 |
| G10 | 双脑并行时的写头归属与时间轴对齐。若走双脑分叉，CTML 写头谁持、两脑时差怎么补 | §2.3 未决分叉 |
| G11 | 感知帧推通道的激进 drain。K5 第三行，音频帧和 vision 帧同为"世界时钟、高 churn、只有最新帧有价值" | interleaved K5 已诊断未实现 |

## 7. 与既有抽象的关系（边界不重绘）

| 先有物 | 关系 |
|--------|------|
| `interleaved-ctml-thinking` (completed) | 执行侧基础，本层直接依赖但不替代它。五动词面是回合制的，本层不替换它——本层是上层载体 |
| `mindflow-control-semantics` (completed) | 感知侧仲裁基础。Priority / 打断信号族复用，本层不重造 |
| `audio-capture` (completed) | 音频感知全链路，本层的"耳"复用它做信号采集 |
| `ghost-runtime-safemode` (in-progress) | 外部打断的授权问题与 SafeMode 邻域——谁能打断 Ghost？留 SafeMode 侧裁决 |
| `channel-meta-dyn-static` (design-locked) | 推通道的载体候选。channel 上线通知的推型实现可以复用动静分离的设计 |
| `moshi` (in-progress) | 语音对话模式演示。Moshi 是"现在能跑什么"，本层是"终局该是什么"——两侧不同时间维度 |

## 8. 不做什么（防扩散）

**这些事情在本 workstream 明确不做，不要顺势带进来：**

- ❌ 实现。依赖的行业前提不成立，落码无意义。
- ❌ 在 MCP 上模拟全双工体验。`interleaved-ctml-thinking` K7 已拒绝这条："MCP 是有损投影，回合制是流式的退化，别把回合制假设烧进终局协议"。本层继续拒绝。
- ❌ 设计双脑并行的具体架构。这是 §2.3 分叉点，先不裁决。
- ❌ 替换 interleaved 的五动词。本层是上游载体不是替代品。
- ❌ 做 TTS 发声范式。发声走 CTML 驱动的 channel，发声质量/延迟是 channel 层的问题。
- ❌ 做实时音频传输协议（WebRTC/SIP/WebSocket 选型）。那是"怎么接"的问题，跟"接上后怎么动"是两个 layer。

## 9. 唤醒条件（未来实例在动手前先读此段）

不是 release checklist——是"什么时候这个 workstream 从锁定变活跃"的判据。

1. **行业判据**：市面上出现至少一个满足 §2.1 三条全部 + 暴露工具调用接口的模型。当前 (2026-07) 没有一个满足。
2. **或可行性判据**：双脑并行的裂脑风险经 `.discuss/` 讨论后确认可接受，不依赖 §2.1 的行业成熟即启动。
3. **启动边界**：只用 realtime 模型做感知 + 发声（感知侧），执行侧仍由 `interleaved-ctml-thinking` 的 CTML 引擎承担。

唤醒后第一件事：读 `interleaved-ctml-thinking` FEATURE.md（已完成）、本 FEATURE.md（锁定）、
`mindflow-control-semantics` FEATURE.md（已完成），然后从 §6 的缺口表里挑可以独立做的开始。

## 10. 关键参考

- `interleaved-ctml-thinking` FEATURE.md — 执行侧动词体系的十一个关键决策，**本层直接继承不加修改。** K5（三时钟分离）、K7（8 字蝴蝶）、K8（append 即 observe）是理解本层的必备上下文。
- `mindflow-control-semantics` FEATURE.md — 感知侧仲裁地基。
- `host/interleaved_thinking.py` — Tracer Protocol + InterleavedThinkingToolset 实现，本层推侧 waiter 的挂点。
- `core/blueprint/mindflow` — 三循环、Priority 抢占、打断信号族。
- `.design/2026-06-28_desktop_in_4d_cross_section.md` — 4 剪影拓扑，理解 Ghost 在时间/空间轴上的存在结构。
- 行业调研 (2026-07-28):
  - [OpenAI Realtime API](https://openai.com/index/introducing-gpt-realtime/) — server VAD + response cancel，非真正 full-duplex。
  - [Gemini Live API](https://ai.google.dev/gemini-api/docs/live-api) — barge-in 协议原生，但工具调用是 turn-scoped。
  - [ByteDance Seed: full-duplex speech LLM](https://seed.bytedance.com/en/blog/introducing-seed-full-duplex-speech-llm-attentive-listening-robust-interference-suppression-enabling-more-natural-interaction) — 架构意义上最近真正 full-duplex，无工具调用暴露。
  - [A Survey of Full-Duplex Spoken Language Models](https://arxiv.org/html/2509.14515v1)
  - [Benchmarking Full-Duplex Voice Agents on Real-World Domains](https://arxiv.org/html/2603.13686v1)
  - [Native Full-Duplex Speech Dialogue with a Single Autoregressive LLM](https://arxiv.org/html/2606.14528)

---

*Claude Opus 4.7, 2026-07-28, via claude code*