---
title: Interleaved Voice
status: draft
# priority: importance within the current stage (iteration cycle) — not development urgency
priority: P1
created: 2026-09-17
updated: 2026-09-19
depends: []
milestone:
description: >-
  全功能交错语音对话体系: 听说两轴的组件化与分布式部署 (听 / 说 / 听+说 节点),
  回声消除与单进程听说共存, moss runtime 自带听说拉起, 听说一体化的 channel 控制面.
  判据是开箱体验 —— macOS 及不自带回声消除的系统上的对话可用性。
---

# Interleaved Voice

> Use `moss features set-status interleaved-voice <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

语音输入 (listener) 与语音输出 (speech) 两侧的概念骨架已各自收口 (见
`voice-input-state-machine` / `speech-governance`, 均已 completed): 听侧有
stream/segment/result 三层身份、可配置礼仪 (first_packet / deliver / stop)、signal
协议化与 clause→ClauseTopic 装线; 说侧有 clause/segment 双回调、播放对齐记账、
say/mute 命令面与说侧 topic 旁路。

**缺的是两者之间的东西。** 今天没有任何代码读"幽灵正在说话"去闸麦克风, barge-in
只有全局粒度 (打断嘴必然连手一起砍), 跨界状态没有载体。两侧都是能工作的器官, 但
合不到一个能对话的身体里。

本 workstream 的目标是**开箱体验**: 一个装了 MOSS 的人, 在 macOS (以及任何不自带
回声消除的系统) 上跑起 ghost, 就能进行一轮自然的交错语音对话 —— 不需要自己装
AEC、不需要自己接分布式进程、不需要手写装配代码。

### 目标 (2026-09-17 人类架构师确立)

1. **考虑回声消除, 支持单进程 听+说** —— 同一个进程里既开麦又出声, 不能自激。
2. **支持 matrix 分布式 听 / 说 / 听+说 节点 (node)** —— 两轴独立组件化后才有的
   部署自由度: 只有听、只有说、或同节点听说。
3. **从 moss runtime 开始支持 host 节点启动自带 听-说** —— runtime 装配层自带这对
   能力, 不是每个 node 自己接。
4. **听说一体化, 支持 channel 控制** —— 模型侧看到的是**一个**语音面, 不是"耳朵
   一个 channel + 嘴挂在 main 上"。

核心判据是 1 和 2 的交集: **分布式部署强制状态必须跨进程** —— 这直接决定了半双工
门控的真值载体形态 (不能是进程内直读)。

## 第一波: 先整理 (已确认的存量问题)

按人类架构师裁定, 以下三项是第一波要清的 —— 都在"目标"之前。

| # | 问题 | 证据 | 性质 |
|---|------|------|------|
| C1 | **`Speech.clear()` 不停止播放** | `core/speech/stream_tts_speech.py:319` 只 copy+clear `_outputted` 账本, 不碰 player; `BaseTTSSpeech` 不持 stream 注册表 | **bug** —— 任何不经 cancel 的 clear 路径都漏嘴 —— **已修 2026-09-19** |
| C2 | **两份 audio topic 定义并存** | `topics/audio.py` (ClauseTopic/AudioSampleTopic, 活的) 与 `types/audio.py` (ConversationTopic/AudioPlaybackTopic, 死的) 并存; 而 `matrix/openbox/topics.py` 声明的恰是**死的那两个** | 同一概念两份定义, 活的那份不在 canonical manifest —— **已清 2026-09-19** |
| C3 | **live topic 不在 canonical 清单** | `matrix/openbox/topics.py` 只导出 ConversationTopic/AudioPlaybackTopic | C2 的连带面 —— **已清 2026-09-19** |

C2/C3 的裁定方向 (2026-09-19, 人类架构师): **topics 不是 types 的兄弟层, 是 types 的下属**
—— 全部 topic schema 收进 `types/topics/`, `types/audio.py` 删除。死的那一对
(ConversationTopic/AudioPlaybackTopic) 从来没有 pub/sub 消费者: 前者只出现在清单的
`__all__` 里, 后者只被 `cli/audio/render.py` 当局部 DTO 用 (已换成本地 `_SpectrumFrame`)。
canonical manifest 现在导出的就是 live 的那几个 —— `moss manifests topics` 可见
`audio/sample` / `clause` / `vision/face` 三个 schema 注册。

同一波还清了 canonical 清单里最后一个"活的假象": `ErrorTopic` (docstring 自陈
"A topic used for testing") 被当 shipped topic 声明了, 已从清单和
`types/topics/__init__.py` 的双重全局导出里摘掉 —— 测试继续直接从
`core.concepts.topic` 拿。判据写进了 `matrix/openbox/topics.py` 的头部注释:
**注册即承诺该名字可跨进程解析, 只声明真有生产者的 topic**。

C1 的机理值得记牢: 今天嘴能停, 纯粹是因为 `shell._clear()` (`ctml_shell.py:738`)
里的 `tree.clear()` cancel 了 say 任务, CancelledError 沿 `SpeechStream.__aexit__`
→ `stream.close()` → `player.clear()` 反卷。**停嘴的责任落在了"取消任务"上, 而不是
"停嘴"这个动作本身** —— 所以它经常看起来有问题。

## 目标形态: 交错礼仪 = (听, 说) 开关的事件面状态机

> 2026-09-17 人类架构师定框: **核心是交错对话, 不是 AEC。**
> AEC 是达成交错对话的技术辅助手段之一 —— 它成立时"不用手动切换"。
> 交错礼仪与"主动/自动"是两个正交概念。

### AEC 的定位: 可行性谓词, 不是礼仪维度

AEC 在产品里有很多做法, openbox 只做一种。它不是礼仪的一个取值, 而是决定**哪些礼仪
在物理上可达**的谓词。openbox 只需要两种礼仪:

| 礼仪 | AEC | 迁移触发源 | 人的打断权 |
|------|-----|-----------|-----------|
| **用户手动切换** | 不需要 | 命令面 (人/模型显式切) | 保有 —— 代价是一次按键 |
| **听说打断 (barge-in)** | 需要 | 事件面 (听起音) | 保有 —— 零代价 |

**被排除的一格是"自动半双工"** (无 AEC × 事件面: 说时自动闸麦)。它不需要 AEC 也能
做, 但它把人的打断权整个拿走 —— ghost 说话期间人无法插话。openbox 的取舍依据是:
**任何一种礼仪都必须保住人的打断权**, 人的打断权比自动化程度优先。

### 状态模型

状态 = **听 on/off × 说 on/off** 两个布尔。迁移由事件驱动, 动作是开关与回调注册的
装卸。礼仪表是数据 (状态 × 事件 → 新状态 × 动作), 运行时持当前状态 —— 沿用
`etiquette.py` 已经确立的"礼仪即配置"范式。

两条有向干扰边, 各自独立, 分开做:

| 边 | 语义 | 本轮 |
|----|------|------|
| **听 → 说** | 听起音 → attenuate 说 | **做** |
| **说 → 听** | 说时听关 (完整半双工闸门) | **暂不做**; 只做**几秒说话保护** —— 说止后短暂抑制听的 commit, 防尾音自触发 |

现有 `host/listener/etiquette.py` 的三层里, 已有这张表的半边:

- `first_packet` (`barge_in` / `interrupt` / `priority`) = 边「听起音 → 说」
- `deliver` = 边「听尾包 → 交付」
- `stop` = 判停 (何时产生尾包)

缺的两件:

1. **说 → 听 那一半** (本轮只到说话保护)。
2. **听 on/off 与说 on/off 显式化为状态**。今天听的开合是"激活哪个礼仪"的副作用
   (`activate` / `stop`), 说的开合是 `mute` 的**命令级硬闸** —— 两者都不是事件面的
   开关。交错礼仪要求它们是状态变量, 而不是别的东西的副产品。

### 主动 / 自动 —— 第二条正交轴

**迁移的触发源**: 事件面 (自动) 还是命令面 (主动)。

这条轴的雏形已经在代码里: `ListenerController.once()` (主动 —— 模型发起听一次) 与
`always()` (自动 —— 常开) 是两个方法, 却被塞进同一个 `ListenEtiquette` 枚举, 与判停
策略混在一起。`etiquette.py` 已经把判停拆成 `StopSpec`; 剩下的 `once` / `always`
这一维, 就是该拆出来的"主动/自动"。

> 待人类架构师确认: 上述读法是否即"主动-自动"的本意。若它指的是控制主权
> (ghost/human/auto, 见旧 10-开关 #2), 那这一轴需要改写。

### 分布式后果: 同一条边的两种实现

"无非就是注册回调关系的一个状态" —— 这是**进程内形态**。当听与说分处两个进程
(目标 2), 同一条边必须退化成协议面:

| 进程内 | 跨进程 |
|--------|--------|
| 直接注册/摘除 observer | topic 订阅 / signal **上行** |
| 直接调方法 | channel command **下行** |
| 直接读属性 | **Parameter** (状态面, 消费者依赖前值) |

所以"一个状态"在进程内是 disposer 集合, 跨进程是 **Parameter + signal + command**
的注册集合。**礼仪表本身不变, 变的是边的实现** —— 这是目标 1 与目标 2 共享同一套
模型的根据, 也是 KD1 (状态真值必须在 Parameter 面上) 的由来。

## Design Index

### 听侧 (已落地)

- 三层身份: `contracts/asr.py` — `RecognitionPhase` / `RecognitionEvent` / `RecognitionClause` / `RecognitionSegment` / `RecognitionStream`
- 耳朵器官与会话: `contracts/listener.py`, `host/listener/listener.py` (`HostListener` / `HostListenerState`)
- 判停与礼仪: `host/listener/stop_judge.py`, `host/listener/etiquette.py`
- 控制面与 topic 装线: `host/listener/controller.py`
- 协议映射: `core/mindflow/listener_nucleus.py`
- node 装配: `host/nodes/listener_node.py`

### 说侧 (已落地)

- `contracts/speech.py` (`Speech` / `SpeechStream` / `TTSSpeech` / `TTSBatch` / `SpeechClause` / `SpeechSegment` / `PlaybackSample` / `StreamAudioPlayer`)
- 命令面: `core/speech/speech_module.py` (`SpeechChannelModule`)
- 说侧 topic 旁路: `host/moss_runtime.py:481` (`_clause_topic_bridge`) / `:531` (`_audio_sample_topic_bridge`)

### 音频设备层

- capture: `host/listener/capture/miniaudio_capture.py` (裸 PCM, 无 AEC)
- player: `core/speech/player/miniaudio_player.py` (裸 PCM, 无 AEC)
- AEC 参考信号可获取: `StreamAudioPlayer.on_play(callback: np.ndarray)` (`contracts/speech.py:461`)、`observe(PlaybackSample)`

### 相关 workstream

- `voice-input-state-machine` (completed) — 听侧全部设计
- `speech-governance` (completed) — 说侧全部设计
- `openbox-nuclei` (in-progress) — 感知核的机制标注层
- `mindflow-interleaved-thinking` (in-progress) — 三循环解耦, barge-in 的仲裁基线

## Key Decisions

### KD1: 半双工门控的真值必须是跨进程状态, 不能是进程内直读 (2026-09-17)

**决策**: 门控真值 (说侧是否在出声 / 听侧是否在收音) 走 **Parameter** (状态面),
不走进程内直读 `player.is_playing()`。

**Why**: 目标 2 (分布式 听 / 说 节点) 强制了这一点 —— 说不一定与听同进程, 进程内
直读在分布式形态下直接失效。Parameter 的机制已具备 (`core/blueprint/parameter.py`
的 declare/subscribe/on_change + `matrix/parameters/zenoh_parameters.py` 的 zenoh 实现,
由 `parameter-host-truth` 收口), 缺的只是这个 parameter 本身。

**状态**: 方向已定 (人类架构师 2026-09-17), 载体命名与字段待定。

### KD2: 命名 —— 语音双工不得占用 "duplex" (2026-09-17)

**决策**: `duplex` 一词已被 `core/duplex/` 占用 (channel provider/proxy 的跨进程
传输层)。语音侧的交错语义另立词汇, 不复用 `duplex`。

**Why**: `core/duplex/provider.py` / `proxy.py` / `protocol.py` 是 channel 运行时
的跨进程传输, 与音频双工无关。同名会造成检索与沟通的系统性歧义。

## Implementation Notes

### 回声消除的路线盘点 (调研结论, 已选 pywebrtc-audio AEC3)

miniaudio **不提供 AEC**, capture 与 player 都是裸 PCM。macOS 上系统级 AEC 存在
(CoreAudio 的 VoiceProcessingIO AudioUnit), 但 miniaudio 不暴露它。三条路线:

| 路线 | 机制 | 代价 |
|------|------|------|
| OS 级 | macOS VoiceProcessingIO AudioUnit (pyobjc / 原生辅助) | 平台锁定, 破坏"miniaudio 零系统依赖"的现有默认 |
| pip 级软件 AEC | **WebRTC AEC3 (`pywebrtc-audio`)** — 2026-09-19 已 `uv add --optional host` | 自带 delay estimator, 对齐是机制而非 hack; 真机效果待 live 验证 |
| 门控级 | 半双工: 说时闸麦 (不需要 AEC) | **与 barge-in 冲突** —— 见下 |

**关键张力 (必须先解)**: barge-in 要求"边说边听" (人在 ghost 说话时插话)。纯半双工
把麦克风关掉, 等于"ghost 说完才能打断", 这与目标里的对话体验直接冲突。所以第一波
不能只做半双工闸门, 必须至少选一条:

- 真 AEC (OS 级或 pip 级), 或
- 半双工 + **起音阈值绕过**: 麦常开, ASR 送入门控, 但用能量阈值 (高于预期回声电平)
  检测起音, 命中即 attenuate 播放并开门。

参考信号在本进程内可得 (`player.on_play` / `observe` 给出真实写入设备的帧), 这为
软件 AEC 提供了前提 —— 缺的是对齐与算法, 不是数据。

**已选 pip 级 (WebRTC AEC3) 并开始验证**: offline 合成实测 AEC3 稳态抑制 ~16dB、收敛
~0.25s; `stream_delay_ms` 提示 0 与提示真延迟结果相同 —— AEC3 的 delay estimator 自行
对齐, 印证「对齐是机制, 不是事后 hack」。留档脚本 (调研 + 结论 + live 判据):
[aec_alignment_probe.py](aec_alignment_probe.py)。live (speak → 查 ASR 有无回声) 待外放实测。

## 迭代路径与装线机制 (2026-09-17 会话决策)

> 人类架构师定路径。三条, 走完即实现装线。

### 1. CTML 通道提权 + speech 脱离 shell

- CTML 需要**通道提权的 prompt + 简单样例**, 尤其是 `<all>...<_>...</_>...</all>`
  这类嵌套语法 —— 提权 = 把某通道的命令提到主轨执行。
- **降权 speech 模块与 shell 的耦合**: `SpeechChannelModule` 默认**不从 container
  取** Speech (现状 `on_startup` 里 `CommandUtil.get_contract(Speech)` 是旧约束的
  遗留 —— "shell 有 speech 就实例化主轨音频输入"是以前"音频必须上主轨"的产物)。

### 2. shell 树 —— 并行/路由机制 (语音脱离主轨), 不是打断机制

shell 核心的隐藏杀器: **解释器多通道化**。

- `ChannelRuntime` 是 task 真相入口 (`channel.py:783` push_task → `push_task_with_paths`
  按 channel path 入独立子树执行栈)。不同 channel 的 runtime **各自独立时序, 无时间耦合**。
- shell 退化为解释器装线 (`ctml_shell.py:389` `interpreter(kind, config=channel子集)`
  → CTMLInterpreter → callback → push_task)。
- 作用域语法 (`open_scope`/`commit_scope` + `<->_`) = 父子依赖栈, 父关子连关。
- **主 shell 可随时拆某子树进旁路通道, 用 CTML 嵌套自控; 甚至托管给 agent (分形 shell)。**

**对 interleaved 的直接含义**（当前理解, 非决议）:

- 语音脱离身体交互靠 shell 树的**路由/并行**: speech 不在主轨 → 声音与躯体各自独立时序。
- **打断的主机制是 command 级** (取消 `say` task), 不是 clear。command 打断本身就是强
  周期约束 —— 取消那个 task, 粒度天然只伤它, 别的 channel 任务不动; `STOPPED(301)` +
  `played_text` 已由 speech-governance 就绪。**clear 是兜底**, 不是打断日常路径。
- 前提 = speech 不能托管到主轨 (即点 1)。动机 = 剥并行主轨, 语音脱离身体交互。

### 3. 地板机参数化进 TUI, 四种 UI 形态

同一个高阶状态机, 四个部署投影:

| 形态 | 载体 | 控制者 |
|------|------|--------|
| a. 嵌入 | ghost/moss runtime, TUI 呈现独立 state | 单进程, 天然命中 AEC |
| b. CLI | `moss audio dialog` | 人 |
| c. node | 图形界面控制 | 人 + 模型 |
| d. channel | 完全模型控制, 人类不控制 | 模型 |

四种形态共享同一状态机与逻辑, 只换渲染/控制面。

## Stage2 雏形 (当前阶段)

> 2026-09-17 人类架构师定范围。核心 = **host 启动时跟随 runtime 的音频交互原型**。
> 开发顺序: CLI 独立验证 → 集成 ghost runtime → TUI repl state。打磨是下一阶段。

| 步 | 内容 |
|----|------|
| 1 | **CLI 独立验证** (先): 地板机原型经 CLI 独立跑通 (once/always 聆听 + say + 打断) |
| 2 | **集成 ghost runtime** (后): 挂进 moss_runtime, 随 host 启动, speech 脱离主轨 |
| 3 | **TUI repl state**: TUI 呈现地板机状态 |

打磨 (AEC / HEAD / PlaybackSample 真相打磨) = 下一阶段, 不在 stage2。

### 术语对齐 (2026-09-17)

**代码为准, 模型翻译。** 人类说「首包/尾包」, 代码写 `FIRST`/`TAIL` (contracts/asr.py
枚举)。模型在文档/代码里用代码词, 对话里翻译。不重命名。

### 两个媒体真相 (2026-09-17)

地板机只吃两个拓扑切面, 不新造事件:

- **听起音 = HEAD** (= 现在的 `FIRST` / ASR 首包, 廉价实现, 待打磨成真 onset)。
- **说起音 = `PlaybackSample`** (真实写入设备的样本, `player.observe`)。

两者都"要打磨对"。`on_speak_start` 若做, 也应是这两个真相的**别名/打磨产物**, 不是
新事件。(speech 在 Event 治理上落后 listener —— listener 有首包/尾包, speech 没有;
这是已知债务, 不是设计目标。)

## 2026-09-19 会话决策 — 完整工作项清单

> 人类架构师 dump 的完整工作项, 记录在此免于每天反刍。按依赖顺序分块。
> 本次会话实测: miniaudio DuplexStream 播侧在本机静默失效 (无双工设备); pywebrtc-audio
> 已 `uv add --optional host` (arm64 + py3.12 wheel 命中, `EchoCanceller.process(near,far)` 冒烟通过)。

### 门控 —— 关键概念 (本次对齐, 推翻早先"门控在 FIRST 上")

**门控在音频层、ASR 之前, 不在 FIRST/ASR 结果层。** 作用是**降低 ASR 提交音频数 (省计费)**
+ 拦静音/回声。返回值是音频帧 (或缓冲后放行), **不是 boolean、不是 FIRST**。FIRST 是
ASR 的语义输出, 在门控之后; 门控做语义判断必然过严/过松。

- 机制 (拦路状态机): 侦测人声 onset → 开始 buffer (首帧不丢) → 放行 ASR; ASR 长时间空 / commit → 重启门控。
- 门控活在两个 segment 之间。宽窄: 叫名字 (wake word) 太窄太蠢; 纯拦静音太宽 (放回声); 正确宽度 = 拦静音 + 拦回声。
- **AEC 是门控"拦回声"那一格的实现, 不是独立东西。**

> 2026-09-19 后续对齐推翻上一条: 门控只做**人声检测 (VAD)**, 回声是 AEC 的独立职责 (已落地),
> 不是门控的一格。机制也从「滑动窗口 + onset 侦测」简化为**静音阈值**: `meta.rms_db < k → None
> (不 init), 否则放行`。最终形状见上方工作项 #2。

### 前置修复 (存量 bug + 机制)

1. **`Speech.clear()` bug** (= interleaved-voice C1): `TTSSpeech.clear()` 只清 `_outputted`
   账本、不停止播放 (`stream_tts_speech.py:319`)。任何不经 cancel 的 clear 路径都漏嘴。
   **已修 2026-09-19**: `BaseTTSSpeech` 加 stream 注册表, `clear()` 现关所有 in-flight
   stream (停嘴) 并返回其 `buffered()`; 删死账本 `_outputted` + `outputted()` (已不在 ABC)。
   测试 `test_stream_tts_speech.py::test_clear_stops_playback` 复现旧 bug、锁定新行为。
2. **recognizer 注册门控 + 生命周期**: recognizer 支持注册拦路门控, 并给出正确生命周期。
   **已做 2026-09-19** — 形状收窄为 `AudioGate = Callable[[AudioChunk], AudioChunk | None]` +
   `AudioGateFactory = Callable[[], AudioGate]` (工厂每 segment 产新鲜门控), 默认
   `silence_gate_factory(threshold_db=-50.0)` 读 capture 预计算的 `meta.rms_db`, 不重算能量。
   门控拦在 `_run_session` 的 init 之前: None → 不 init 继续缓冲; 非 None → 放行 + init,
   本 segment 内不再拦。测试锚定两条契约: 纯静音流不 init; 静音丢弃后首个人声帧放行。
   前置两条 refactor (同 wave 独立 commit, 可 review):
   - consumer 声明消费格式: `new_sequential_consumer(target_sample_rate=...)`, resample 下沉进 consumer (生产侧 fan-out 可复用、消费侧重采样不再各写一遍)。
   - recognizer 吃 `AudioChunk` (非 `np.ndarray`), listener 的 ad-hoc 拆包/resample 桥删除。
3. **AEC 屏蔽细节**: AEC 在两个接口表面 (near/far) 屏蔽实现, 启动时注册;**对齐延迟不能是
   "事后 hack 对齐"** (脚本里互相关/能量起点那种), 要在抽象上有机制。
   留档脚本 (调研 + offline 实测结论 + live 判据): [aec_alignment_probe.py](aec_alignment_probe.py)。
   offline 已验证 AEC3 稳态抑制 ~16dB、收敛 ~0.25s, 且 hint=0 与 hint=真延迟结果相同
   (delay estimator 自带对齐, 不用 hack); live (speak → 查 ASR) 待外放实测。

### 配置与降级

4. **shell speech 显式注册**: speech 从默认注册改显式注册; 历史单测要优化一遍。
   **已做 2026-09-20 (shell/module/host 三层)** — speech 一等公民 threading, 兜底去掉:
   - `CTMLShell._speech_context_manager` 不再 `container.get(Speech)` 兜底, 也不 NullSpeech;
     构造/set_speech 显式传入才算数, None → 不 set/不启动/不挂 content command (`_clear` 守护 None).
   - `SpeechChannelModule(speech=None)` 加构造注入; `on_startup` 去掉 `or NullSpeech()`,
     递归取 + `is_running()` 判活, None/未 running → 不装线 (不挂 say/mute). 递归保留:
     非 shell 场景 (远程 node 独立做音频) 靠容器取.
   - `Host.run(speech: bool = True)` / `run_ghost` → `ShellRuntimeImpl(speech: bool)`;
     `__aenter__` 里 `_resolve_speech()` (matrix bootstrap 后) resolve Speech 实例注入 shell,
     失败降 None (降级细节留 #7); 旁路桥改用 `self._speech`.
   - 契约锚定: `test_module_without_speech_wires_nothing` (无 speech → 不挂 say/mute).
   **待做**: CLI `--speech`/env OPTION 开关 (留给 #5) + provider 降级细化 (#7).
5. **moss runtime 启动 flag** (可能进 host 表面): 默认 speech; 可选 speech + listener 的
   interleaved voice 状态机 (或改名叫 AEC, 对齐行业); 可选择空。
   **已做 2026-09-20 (listen flag + 治理骨架)**: `Host.run/run_ghost` + `ShellRuntimeImpl`
   加 `listen: bool = False`; `_resolve_listener()` 对称 `_resolve_speech()` resolve ASRListener
   并组装 ListenerController (失败降 None); `_listen_lifecycle` context manager enter controller
   + wire AEC far 桥 (speech 是 TTSSpeech 时); `ShellRuntimeImpl.pause(toggle)` 级联到
   `ListenLifecycle.pause` (急停停听 / 恢复默认礼仪).
   **待做**: CLI `--speech/--listen`/env OPTION 开关.
6. **config type 加 `validate` 函数**: per-config 自校验 (如环境变量实际为空时 raise)。
   **已做 2026-09-20 (approach 2, 不做 ConfigStore 机制)**: 仅这几个 config 加 `validate()`,
   provider 显式调用 (非 ConfigStore 读时自动):
   - `TTSManagerConfig.validate()` — volcengine 分支检查 `app_key`/`access_token` 非空非 `$`;
     `TTSServiceProvider.factory` `get_or_create` 后调用.
   - `VolcengineSaucConfig.validate()` — 检查 `api_key` (listener 侧后接).
   **待做**: ConfigType 基类 `validate()` + ConfigStore 读时调用一次的机制 (可选, 用户判可跳过).
7. **provider 降级**: speech / listener provider 据 config validate 降级 (null speech / null listener)。
   **已做 2026-09-20 (speech 侧)**: `TTSSpeechServiceProvider.factory` catch `force_fetch(TTS)`
   异常 → `logger.warning` + 返回 `NullSpeech()`. `NullSpeech` 现在有不可用信号:
   `_NullSpeechStream.played_text()` 返回 `"speech 注册不可用"`, `__content__` 返回
   `played_message(samples) or chunks__.played_text() or None` 让消息浮出.
   于是 `speech=True`+缺 env → say 挂载但返回"注册不可用"; `speech=False` → 不挂 say.
   **待做**: listener provider 降级 (listener 未装线).
8. **SystemError / SystemBootstrap 模块**: 注册为 Project 默认依赖, provider 可获取它记录
   启动异常; 封装成 channel (moss 运行后 ghost 可看系统级异常, 可 pull 最近 n 条); 甚至考虑作 logger handler。

### voice 综合状态机

9. **对齐 AEC**: ASR 不听输出 (回声不进 ASR)。
10. **首包发 signal 但不打断 speech**: 两边一起说 (双讲) 也许是好 feature。→ 推翻早先
    "听起音 → attenuate 说"的 barge-in 假设 (KD 中"听→说 做"那条)。
11. **半双工 (说时不听) 可能不必要**: 它依赖外部界面启动 (永不自起), 有 AEC 后优先级下降。
12. **封装物料**: listener controller 只实现单侧机制; 整体封装要在 `host/` 模块下有物料,
    方便迁移成 node, 甚至预写在 `host/nodes/`。
    **已做 2026-09-20 (生命周期表面)**: controller 不走 provider — 定义 `ListenLifecycle`
    (contracts/listener.py) 仅承诺 enter/exit + pause, `ListenerController` 反向继承它;
    moss runtime 只认这个表面治理听侧, concrete 构造在 `_resolve_listener`. 判停/信号/
    自解释那面还在演化, 不上 IoC. 顺带修了一个既有 bug: `self._etiquette_config` 属性与
    同名方法冲突 (属性遮蔽方法), 改缓存名为 `_etiquette_config_cache`.

### 最关键的改造

13. **单进程听/说分句交错进统一历史**: 统一 `on_clause` 回调 (供 GUI), 可直接用 topic (已对齐过)。

### recognition 交互

14. **recognition 返回可自增的未发送数据**: commit 默认发 signal 被拦截后, 界面 buffer
    未发送对话、点击提交; 尾句可触发 llm func 重写 (避免差 ASR 物料)。
15. **尾包未发送时进 channel notice**: 模型思考可看 last clause 等信息, 有拉接口;
    相当于模型可自己给自己 commit。

### 明确不做 / 现状

- **"大模型改写 segment"不做** (唯一明确排除项)。
- "ASR 完成 + 人类手动发送"机制: 有现成实现可参考, 无需从零设计。
