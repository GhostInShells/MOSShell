---
created: 2026-07-30
depends: []
description: 为 speech（moss shell 第一公民）补全说侧协议：分句对齐、播放游标折算、 双侧共享的句级广播、状态快照与音频持久化。使"实际说出的话"成为
  可记账、可观察、可回放的 ground truth。
milestone: v0.1.0
priority: P1
status: dropped
status_note: 设计已吸收进 voice-input-state-machine (2026-08-12 audio 架构清单); 说侧四层对齐+基建盘点已迁入其
  2026-08-12 会话决策一节
title: Speech Protocol Alignment — 说侧四层对齐、分层广播与真实话语记账
updated: '2026-08-12'
---

# Speech Protocol Alignment

> Use `moss features set-status speech-protocol-alignment <status> -m "note"` to update state.

> **2026-08-12 — 本 workstream 已吸收进 voice-input-state-machine（说侧对齐、折算、conversation topic 均并入其「2026-08-12 会话决策 — audio 架构讨论清单」），状态 → dropped。**
> 唯一设计资产（四层对齐模型 + 说侧基建盘点：Interpretation 记账、contracts/speech 生命周期）已压缩迁入该 FEATURE 的「2026-08-12 会话决策」一节。完整原始内容留在 git 历史。

## Motivation

speech 是 moss shell 的第一公民：它是 `__main__` 的 `__content__`，Ghost 的默认
输出模态，**它的播放时间线就是整个 shell 交互的共享时钟**。打断、轮次边界、
字幕、回放、barge-in，全部要向这条时钟对齐。

当前断点：模型 emit 的文本（计划）和实际播出的音频（现实）之间没有账本。
TTS 说到半句被打断时，`say` task 落进 `cancelled_tasks`，但"实际说出口的半句话"
无人折算、无人回报。用户听到了，Ghost 却不知道自己说了什么——防伪造红线
（CTML plans the future, observations arrive later）在命令级有 Interpretation
记账保障，在音素级是空的。

由于缺这一个实现，**四层数据里中间两层（command-stream 级、tts 分段级）完全
拿不到**，导致下游一串能力全部悬空：中断后的自然接续、逐句字幕/高亮回放、
每轮对话音频持久化、voice-input 状态机的回声抑制与 barge-in 检测。

本 FEATURE.md 的使命是把这个设计上下文完整固化，使任意模型实例可以
零同步接手。

## 四层对齐模型（核心设计）

```
L1 interpreter 级     一轮 logos = 一个 Interpretation
                      │ 包含 0..n 个 speech command
L2 command-stream 级  一个 say/__content__ = 一个 SpeechStream (batch_id = task.cid)
                      │ 分句器切分
L3 tts 分段级         句子 = TTSItem{text, audio, duration}, 携带 (cid, seq)
                      │ 转码入队
L4 play 分段级        player chunk + 播放游标, on_play 逐片回调
```

**对齐的本质是每层持有对上层的引用**：sentence 记 cid，play chunk 记
sentence seq。中断折算沿引用链向上折：L4 游标 → L3 哪些句子播完/播到几成
→ L2 每个 cid 的已播文本 → L1 本轮真实说出的 logos。

概念清单：**分句对齐、流对齐、协议广播**；中断时
扔一个 play-canceled 事件或 `clear() -> str` 均可接受。

## 分层广播原则

`concepts/topic.py` docstring 已定死架构原则：Topic 只承载**秒级大脑事件**。
高频数据走 stream 协议。因此：

| 层 | 频率 | 载体 |
|---|---|---|
| 帧级 (10~50ms) | 高频 | stream 协议（Zenoh PCM / consumer 抽象），不进 Topic |
| 句级 | 秒级 | TopicService 广播 |
| 状态快照 | 持续覆盖 | `TopicWindow(max_size=1)` 模式 |
| 轮次级 | 每轮 | Interpretation 记账（进程内） |

## 现有基建盘点（2026-07-30 调研，均已核实）

### 说侧原材料（缺的只是折算与广播）

- `contracts/speech.py` — 生命周期四元门控 commit/synthesis/play/close 可乱序，
  新 stream 的 `start_play` 关闭上一个 stream；`TTSItem{text, audio, sample_rate,
  ...}` **已经是文本-音频对齐单元**；`StreamAudioPlayer.add() -> float` 返回
  播放结束时间戳，`on_play(np.ndarray)` 逐片回调，`is_playing()` 可查；
  `SpeechStream.feed()` 把 buffered 文本持续写回 `cmd_task.tokens`（注意：
  这是"喂了什么"，不是"播了什么"）。
- `core/concepts/interpreter.py` — `Interpretation` 五本账（compiled/pending/
  success/cancelled/failed）；`executed_inputs` 只累计成功 task 的 tokens；
  **`on_done_task` 的 result 合并对 cancelled task 同样生效**——被打断的 say
  task 照样可以把"实际说出的话"经 result messages 送进观察上下文，回报通道
  现成。

### 听侧（已完整分层，是说侧的对齐样板）

- `contracts/audio.py` — 帧级 `AudioChunk{seq, timestamp, samples,
  meta(rms/bands/is_silent)}`；双 consumer 模式：`AudioPullLatest`（有损取
  最新，波形/AI 感知）+ `AudioSequentialConsumer`（无损有序带背压，ASR/录音）；
  `AudioCaptureConfig` 做格式共识。
- `topics/audio.py` — `SpeechTopic`：ASR 分句完成后才发布，无 delta，每条自
  包含；**`role: Literal['ghost','user']` —— 句级协议设计上就是说听双侧共享
  同一 schema**；`batch_id` 字段正好挂 command task cid（L2 链接）；`audio_key`
  预留音频持久化引用（todo 标注未实现，关联 matrix-resources）。
  `AudioRuntimeTopic`：TopicWindow(max_size=1) 状态快照样板（心跳+运行状态）。

### 承载与共享机制

- `TopicWindow`（concepts/topic.py）— 有界滑动窗口，绑定 TopicService 生命
  周期；`values()` 线程安全快照；`on_change(callback, debounce=, throttle=)`
  —— debounce 即"说话人停顿后再转写"，throttle 保证连续流不饿死回调。
  **`TopicWindow[SpeechTopic]` over recent N = 语音对话上下文窗口，也是 GUI
  历史展示的直接数据源**。
- `core/blueprint/parameter.py` — Params：typed/versioned 共享状态，低频写
  (<1Hz) 高频读，SQLite ground truth + Zenoh 失效信号，CAS 写入，ROS2 参数
  语义（declare → get/set → on-change）。适合承载跨进程共享的 speech 配置
  与慢状态（如当前音色、语速、mute 状态）；秒级以内的 is_speaking 快照则
  用 TopicWindow(1)。

## 说侧缺口（施工清单）

1. **句级发布者**：TTSItem 播完/被打断时发 `SpeechTopic(role=ghost)`。
   这是"中间两层拿不到"在广播维度的表现，也是本 feature 的核心。
2. **状态快照**：`SpeechRuntimeTopic`（对标 AudioRuntimeTopic）——
   is_speaking + 当前 (cid, seq, offset)。voice-input 状态机的回声抑制与
   barge-in 靠它，现在只能戳 `player.is_playing()`。
3. **中断折算**：`clear()` 时沿账本折算已播文本（见 Key Decisions #2 精度
   分级），cancelled task 以 UtteranceReport/str resolve，经
   Interpretation.messages 喂回模型。
4. **帧级 consumer 抽象**：`player.on_play` 目前是裸回调。复用 `AudioChunk`
   结构（seq/timestamp/meta）即可让 GUI 波形对说听双侧同构处理。
5. **格式共识归一**：听侧 `AudioCaptureConfig` 与说侧 `TTSInfo` 是同组物理
   参数的两套 schema（`AudioFormat` 枚举已共享），可部分归一。
6. **音频持久化**：订阅句级广播的 sink 把 TTSItem 音频 + 对齐元数据落盘
   （per-turn 目录 + alignment.jsonl），回填 `SpeechTopic.audio_key`。
   页面逐句高亮回放、字幕、口型同步是这条广播的免费副产品。

contracts 层签名变更预计只有一处：`Speech.clear()` 增加返回值。

## Key Decisions

<!-- 以下为 2026-07-30 会话（claude + 人类工程师）达成/提议的判断 -->

1. **被打断的半句用带 interrupted 标记的 SpeechTopic 表达，不另设独立
   canceled 事件**（模型提议，待人类确认）。理由：对话上下文窗口
   （TopicWindow[SpeechTopic]）要的本来就是"实际说出的话"，半句也是话；
   这样 `clear()` 返回值和广播事件是同一本账的两个出口，防伪造红线与
   上下文窗口共用一份数据。
2. **对齐精度分三级演进**：item 级（TTSItem 粒度，零新依赖，基线）→
   时长比例级（播放游标 ÷ item 时长线性插值）→ 字级（需 TTS word
   timestamps，可选增强）。基线不动任何现有抽象，只在 SpeechStream 实现
   的 close/fail 路径上维护 (text, duration, enqueued_at) 账本。
3. **广播分层遵循既有 Topic 频率原则**：帧级不进 Topic，句级进
   TopicService，快照用 TopicWindow(max_size=1) 模式（AudioRuntimeTopic
   已示范）。

### 待人类 L2 拍板的开放问题

1. 分句器位置 — SpeechStream 内部，还是独立可替换契约？（中英分句规则
   不同，部分云 TTS 自带分句）
2. 广播载体细节 — 全走 Topic 体系，还是部分直接挂在 stream 协议上？
   （两个方向均可接受，倾向未定）
3. `clear()` 返回形态 — `-> str`（简单）还是 `-> UtteranceReport`
   （保留分层结构）？
4. 持久化 sink 归属 — 进本 feature，还是拆给 matrix-resources？
5. priority 待定 — 本文暂标 P1，人类可改。

### 2026-08-12 会话补充（说侧架构方向）

说侧相关决策已并入 voice-input-state-machine FEATURE 的「2026-08-12 会话决策 — audio 架构讨论清单」，
本 feature 涉及其中 #1 player 单例、#2 speech 开放观测接口、#3 conversation topic、#8 中断折算、#11 分句器。要点：

- **player 单例化**（`AudioPlayerProvider` → singleton=True），speech 为持有者，外部不直接 fetch player。
- **speech 不直接耦合 topic**——开放观测接口，topic 接线放外侧；stream 可持有父层 callback。接口形态未定。
- **句级广播可能演进为 conversation/dialog topic**，吸收听/说双侧——本 feature 的 `SpeechTopic(role=ghost)` 发布是其中一侧。
- **中断折算语义必须保留**：`clear()` 折算已播半句、interrupted 标记、cancelled task resolve。

## 与其他 workstream 的关系

- **voice-input-state-machine (P0, design-locked)** — 它管"听到一半"，本
  feature 管"说到一半"，合起来才是完整打断语义；它需要的 is_speaking 状态
  来自本 feature 的 SpeechRuntimeTopic。
- **realtime-voice-interaction-logos (P2, design-locked)** — 音频第一公民
  的流式知觉，句级双侧广播是其数据面。
- **matrix-resources (P1, draft)** — `SpeechTopic.audio_key` 的存储后端。
- **desktop-gui / text-blocks** — TopicWindow[SpeechTopic] 历史 + 帧级波形
  的 GUI 消费者。

## Implementation Notes

- 下一个实例的最短阅读路径：`contracts/speech.py`（生命周期与 TTSItem）→
  `topics/audio.py`（SpeechTopic 双侧共享设计）→ `contracts/audio.py`
  （听侧分层样板）→ `core/concepts/interpreter.py` 的 Interpretation 记账
  （回报通道）→ `core/speech/`（BaseTTSSpeech 实现，**本次会话未读**，动工
  前必读）→ `moss codex blueprint states_channel`。
- `SpeechStream.feed` 写回 `cmd_task.tokens` 的语义要小心：它反映 fed
  （计划）不是 played（现实）。折算实现落地后要明确两者边界，不要让
  executed_logos 的口径被污染。
- CLAUDE.md（channels/）要求每个命令显式标注 `always_observe`；say 类命令
  为 False，但被打断时经 result.observe 升级观察——这个组合正是协议预期。
- 本 FEATURE.md 由 claude 在"初见体验验证"会话中沉淀，上下文来源：人类
  工程师口述（四层模型、分层广播、双侧对齐、第一公民判断）+ 模型对
  contracts/tests/topics 的一手调研。若与代码冲突，以代码为准并回改本文。