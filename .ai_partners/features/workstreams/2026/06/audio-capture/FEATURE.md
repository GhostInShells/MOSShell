---
created: 2026-06-03
depends: [session-parameter-store, topic-ringbuffer]
description: 完整音频感知管线 — miniaudio PCM 捕获 → Zenoh 流 → ASR 识别 → SpeechTopic 广播 → mindflow AudioSignal 抢占注意力。五种交互模式 + 四项可选能力界面 + Matrix 解耦。MVP 收敛为单一交互方式跑通 input signal → callback ghost。
milestone: null
priority: P1
status: in-progress
status_note: Step 1-5 完成。本轮会话完成设计收敛与 Matrix 解耦方向锚定。移交后续伙伴完成 .design 文档与任务拆分。
title: Audio Capture — 音频感知全链路
updated: '2026-06-07'
---

# Audio Capture — 音频感知全链路

> Ghost 现在只有嘴（TTS），没有耳朵。此 feature 补上音频输入感知全链路：
> 系统音频 → miniaudio → PCM → Zenoh → 消费者（波形/ASR）→ SpeechTopic 广播
> → AudioSignal → mindflow 注意力抢占。
>
> 本轮会话（2026-06-07）完成设计收敛：Matrix 解耦方向、TopicWindow 替代 tmp_storage、
> SpeechTopic 统一语音协议、AudioSignal 接入 mindflow、五种交互模式定义。
> 移交后续伙伴完成 .design 文档与任务拆分。

## Motivation

1. **Ghost 需要听觉感知**：当前 speech Channel 只管输出（TTS），无法感知系统音频。波形可视化、ASR、音频剪辑等场景全部依赖音频输入。

2. **miniaudio 已在项目中作为默认音频后端**（`MiniAudioStreamPlayer`），同样的库对称支持 `CaptureDevice`，零新依赖。

3. **捕获与加工必须解耦**：不要把所有处理逻辑塞进捕获节点。捕获只做一件事——干净地交出原始 PCM。FFT/ASR/波形渲染是各自消费者的事。

4. **格式共识编译期已知**：用 ConfigType 定义采样参数，消费者读配置即知流格式，不需要运行时协商。

5. **进程安全**：`start()` 用 `FileLocker` 拿进程锁，避免多进程重复打开同一设备。

6. **Matrix 解耦**：audio 核心不应依赖 Matrix。定义 `AudioTransport` 抽象，`MatrixAudioTransport` 作为唯一适配点。依赖方向: contracts → MatrixAdapter，不是 core → Matrix。

7. **统一语音协议**：ASR 和 TTS 输出用同一个 `SpeechTopic` 广播，TopicWindow 承载对话上下文。

8. **mindflow 注意力抢占**：音频感知结果通过 `AudioSignal` 接入 mindflow，首包打断当前注意力。

## Design Index

### 核心合约 (contracts/audio.py — 已有 + 新增)

已有:
- `AudioFrameMeta`, `AudioChunk`, `AudioCaptureConfig`, `AudioRuntimeInfo`
- `AudioCaptureSource(ABC)`, `AudioPullLatest(ABC)`, `AudioSequentialConsumer(ABC)`

新增:
- `AudioTransport(ABC)` — 传输抽象，隔离 Matrix/Session，见 KD10
- `AudioSignal(SignalMeta)` — 音频感知信号，signal_name="audio"，见 KD13
- `Preemptable(Protocol)` — 可选能力: 可被注意力抢占打断，见 KD14
- `SpeechEventEmitter(Protocol)` — 可选能力: 广播 SpeechTopic，见 KD14
- `SpeechEventReceiver(Protocol)` — 可选能力: 接收 SpeechTopic，见 KD14
- `AudioRuntimeReporter(Protocol)` — 可选能力: 上报运行时状态，见 KD14

### 语音协议 (contracts/speech.py — 新增)

- `SpeechTopic(TopicModel)` — 统一话语事件，见 KD12
- `AudioRuntimeTopic(TopicModel)` — 运行时状态 Topic（替代 tmp_storage），见 KD11

### 实现

- `host/speech/capture/miniaudio_capture.py` — MiniAudioCaptureSource + 消费者（已有）
- `host/speech/capture/matrix_audio_transport.py` — MatrixAudioTransport 适配器（待实现）
- `host/providers/audio_capture_provider.py` — singleton Provider + IoC（已有，需改为组装 Transport）

### 交互模式

- 五种交互模式定义，见 KD15
- 旁路 flash 模型滚动修正 + 本地术语表 + 耳机/外设通道

### 生命周期

- 独立 App（`.moss_ws/apps/sensors/audio_capture/`）— Ghost 通过 `apps:start`/`apps:stop` 控制启停
- Waveform App（`.moss_ws/apps/sensors/waveform/`）— 跨进程终端可视化消费者
- Listener App（`.moss_ws/apps/sensors/listener/`）— ASR 消费者，待实现

### Mindflow 集成

- AudioSignal → Nucleus → Impulse → Attention challenge
- 首包 complete=False 抢占，尾包 complete=True 解锁 think-act loop

---

## Key Decisions

### KD1: 原始 PCM 走 Zenoh，不做特征提取

> 已在 2026-06-05 实现。

捕获源只产出原始 PCM（`AudioChunk.samples`）。每个 chunk 附加一份轻量元信息（`AudioFrameMeta`：rms_db + bands + is_silent），在捕获端 FFT 算一次，随 chunk 一起走。消费者只读自己需要的：

| 消费者 | 读什么 |
|---|---|
| 波形 Channel (未来) | meta（RMS + bands），不重算 PCM |
| ASR / listener (未来) | samples |
| AI 感知 | meta 快速判断，需要时再读 samples |
| 音频剪辑 (未来) | samples |

**接受**: meta 是 frame 的副产品，不是独立流。和 sample 打包在同一个 AudioChunk 里，一次 Zenoh pub 同时送出。
**拒绝**: 捕获源内建 N 种加工管线——加工逻辑属于消费者，不属于采集节点。捕获节点应为极简管道。
**拒绝**: 只广播特征帧不广播 PCM——ASR 和剪辑需要原始采样。86 KB/s（44.1kHz mono 16bit）对 Zenoh TCP localhost 是零头。

### KD2: PCM 流走 Zenoh 原生 pub/sub，不走 TopicService

> 已在 2026-06-05 实现。

TopicService 设计意图是"秒级大脑事件"（见 `Topic` docstring）。PCM 流是 50ms/帧、86 KB/s 的连续传感器数据——语义完全不匹配。Zenoh 原生 `pub`/`sub` 设计场景就是 MB/s 级传感器流，有序、有可靠性 QoS、有 backpressure。

**接受**: audio-capture 用 Zenoh 原生 API（`pub_stream_delta` / `sub_stream`），TopicService 只用于状态变更通知和 SpeechTopic 广播。
**拒绝**: PCM 走 TopicService——语义错位，且 TopicService 的 Subscriber 模型不适合流式消费。

### KD3: ConfigType 格式共识 + TopicWindow 运行时发现

> 已实现，运行时发现方式变更见 KD11。

两层发现机制，各司其职：

```
AudioCaptureConfig (ConfigType)     → moss manifests configs 可见，编译期已知
  sample_rate, channels, format, frame_duration_ms, device_pattern

AudioRuntimeTopic (TopicWindow)     → 持续更新的运行时状态，替代一次性 tmp_storage 写入
  running, stream_key, device_name, started_at, last_heartbeat
```

ConfigType 回答"格式是什么"——消费者编译期就知道怎么解析流。TopicWindow 回答"在哪、活着吗"——消费者通过 `window.values()` 或 `on_change()` 获取最新状态。

**接受**: ConfigType 由消费者定义和使用，不是 App 私有的。格式是共识，不属于任何一方。
**拒绝**: tmp_storage 一次性写入——无法表达心跳和状态变更，消费者需轮询文件。TopicWindow 自然支持持续更新和变更通知。

### KD4: 两类消费者模型 — pull_latest 与 sequential

> 已在 2026-06-05 实现。

音频消费者分两种语义，各提供独立模型：

```python
class AudioPullLatest(ABC):
    """非阻塞拿最新帧。波形可视化、AI 按需感知适用。"""
    def pull_latest(self) -> AudioChunk | None: ...
    def close(self) -> None: ...

class AudioSequentialConsumer(ABC):
    """顺序消费，不丢帧，支持背压。ASR / 录音适用。"""
    def __aiter__(self): ...
    async def __anext__(self) -> AudioChunk: ...
    async def start(self) -> None: ...
    async def close(self) -> None: ...
```

`pull_latest` 内部: ring buffer + 非阻塞读，写满时丢最老帧。
`sequential` 内部: `asyncio.Queue(maxsize=N)` + Zenoh sub callback relay。队列满时 `put()` 阻塞 → 自然背压 → Zenoh 侧自动丢弃。cancel 后 finally 发 sentinel → 消费者优雅退出。

**接受**: consumer 级参数——不同消费者有不同容忍度（波形显示 32 帧够了，ASR 可能需要 128）。
**拒绝**: 放在全局 ConfigType——那是对所有消费者的强制约束。

### KD5: Provider 注入 Matrix，Singleton 生命周期

> 已在 2026-06-05 实现。Matrix 解耦后 Provider 组装 MatrixAudioTransport，见 KD10。

```python
class AudioCaptureProvider(Provider[AudioCaptureSource]):
    def singleton(self) -> bool: return True
    def factory(self, con: IoCContainer) -> AudioCaptureSource:
        matrix = con.force_fetch(Matrix)
        config = con.force_fetch(ConfigStore).get_or_create(AudioCaptureConfig())
        transport = MatrixAudioTransport(matrix=matrix)
        return MiniAudioCaptureSource(transport=transport, config=config)
```

和 `AudioPlayerProvider` 对称：播放端提供 `StreamAudioPlayer`，捕获端提供 `AudioCaptureSource`。

### KD6: start() 底层加进程锁，不强依赖 App 包装

> 已在 2026-06-05 实现。

`start()` 实现层通过 `FileLocker` 拿进程锁。可以直接作为 Provider 提供的 singleton 使用，也可以包成 App 由模型通过 AppStoreChannel 控制启停——不互斥。

### KD7: 独立 Zenoh 会话是远期优化，MVP 用主 Session

> 已在 2026-06-05 记录。解耦后 Transport 可注入任意 Zenoh session，换端口无需改动 audio 核心。

原始 PCM 86 KB/s 在主 Zenoh session 上和 topic/signal/logos 混跑不会造成阻塞——Zenoh 设计场景是 MB/s 级传感器流。但音频有自己的 QoS 特征（允许丢帧、延迟敏感度低），长期应物理隔离：

```
端口 20770: 主控平面
端口 20775: 音频数据平面 (未来)
```

MVP 用主 Session。`AudioTransport` 抽象已隔离此细节——换独立 session 只需换一个 Transport 实现。

### KD8: 运行时稳定性优先 — 允许丢帧，禁止泄漏

> 已在 2026-06-05 实现。

流式 ASR 的场景：180-210ms chunk 直接上报火山引擎。偶尔丢帧只影响那一小段识别质量，不影响系统运行。真正危险的是：

1. **内存泄漏** — 消费者跟不上，队列无限堆积
2. **崩溃传播** — WebSocket 断开导致整个 task 树崩溃
3. **资源泄漏** — cancel 后 WebSocket/audio stream 没收干净

`sequential` consumer 用有界 `asyncio.Queue` 解决 (1)——满了就阻塞 pub 侧，Zenoh 自动丢老帧。用 asyncio task + try/except/finally + sentinel 解决 (2)(3)。

### KD9: 与 listener feature 的依赖关系

> 已在 2026-06-05 记录。listener 集成见后续任务拆分。

audio-capture 是 PCM 源，listener 是 ASR 消费者。依赖链：

```
AudioCaptureSource (本 feature)
  └─ .new_sequential_consumer(ring_buffer_frames=128)
       └─ AudioSequentialConsumer (async iterable)
            └─ listener 内层循环: async for chunk in consumer
                 ├─ feed Recognizer → Recognition
                 ├─ pub SpeechTopic(TopicService)  ← 广播到对话上下文
                 └─ emit AudioSignal → mindflow    ← 注意力抢占
```

listener 不拥有麦克风——它通过 audio-capture 的 consumer 获取 PCM。两个 feature 并行开发，只要 `AudioCaptureSource` + `AudioSequentialConsumer` 的 contract 先稳定。

---

### KD10: Matrix 解耦 — AudioTransport 抽象（本轮锚定）

**问题**: 当前 `MiniAudioCaptureSource.__init__` 直接拿 `Matrix`，依赖了 Matrix 的完整生命周期体系。但 audio capture 是一个纯数据生产者——它只需要传输能力。

**实际依赖分析**:

```
MiniAudioCaptureSource 的 Matrix 使用点:
  matrix.logger                        → logging.Logger (stdlib 就够)
  matrix.workspace.lock()              → FileLocker (跨进程互斥)
  matrix.session.pub_stream_delta()    → Zenoh pub
  matrix.session.sub_stream()          → Zenoh sub (callback)
  matrix.session.get_stream()          → Zenoh sub (async for)
  matrix.session.tmp_storage           → KV 写 AudioRuntimeInfo (后续改 TopicWindow)

真正依赖: 一个能 pub/sub 的传输层 + 跨进程锁。Matrix 恰好持有 Zenoh session，
但 audio 核心不应该知道 Matrix 的存在。
```

**解法**: 定义 `AudioTransport(ABC)`，audio 核心只依赖它。Matrix 作为实现层适配。

```python
# contracts/audio.py

class AudioTransport(ABC):
    """音频体系传输抽象。隔离 Matrix/Session 依赖。"""

    # -- PCM stream --
    def pub_pcm(self, chunk: bytes) -> None: ...
    def sub_pcm_callback(self, on_chunk: Callable[[bytes], None]) -> Callable[[], None]: ...
    def sub_pcm_stream(self, maxsize: int) -> StreamSubscriber: ...

    # -- process lock --
    def acquire_lock(self) -> bool: ...
    def release_lock(self) -> None: ...

    # -- topic broadcast --
    def pub_topic(self, topic: TopicModel) -> None: ...
    def topic_window(self, model: type[TOPIC_MODEL], max_size: int) -> TopicWindow[TOPIC_MODEL]: ...

    # -- logger --
    @property
    def logger(self) -> logging.Logger: ...
```

```python
# host/speech/capture/matrix_audio_transport.py (待实现)

class MatrixAudioTransport(AudioTransport):
    """唯一的 Matrix 耦合点。从 Matrix/Session 提取传输能力，适配到 AudioTransport。"""
    def __init__(self, matrix: Matrix): ...
```

**依赖方向**:

```
现在 (反了):  MiniAudioCaptureSource → Matrix
正确方向:    MiniAudioCaptureSource → AudioTransport(ABC) ← MatrixAudioTransport → Matrix
             ─── contracts 层 ───          ─── host 适配层 ───
```

**收益**:
- audio 核心可脱离 Matrix 单独测试（mock AudioTransport 即可）
- 换 Zenoh 会话（KD7 独立端口）只需换 Transport 实现
- PCM 走 Zenoh、topic 走 TopicService、锁走 FileLocker——全部是 Transport 内部选择，audio 不关心
- contracts 层零 Matrix import

**Provider 变更**:

```python
class AudioCaptureProvider(Provider[AudioCaptureSource]):
    def factory(self, con: IoCContainer) -> AudioCaptureSource:
        matrix = con.force_fetch(Matrix)
        config = con.force_fetch(ConfigStore).get_or_create(AudioCaptureConfig())
        transport = MatrixAudioTransport(matrix=matrix)
        return MiniAudioCaptureSource(transport=transport, config=config)
```

**实施要点**:
1. `contracts/audio.py` 定义 `AudioTransport` ABC — 与 AudioCaptureSource 同级
2. `host/speech/capture/matrix_audio_transport.py` 实现适配器
3. `miniaudio_capture.py` 构造函数 `Matrix` → `AudioTransport`，内部所有 `self._matrix.xxx` → `self._transport.xxx`
4. Provider factory 中组装 MatrixAudioTransport
5. App 入口 (`audio_capture/main.py`) 无需改动——它通过 Provider 拿 source

**与 KD5 的关系**: Provider 仍然是 singleton + IoC，但不再是"Provider 拿 Matrix 给 capture"，而是"Provider 组装 Transport 给 capture"。

---

### KD11: AudioRuntimeInfo → AudioRuntimeTopic，走 TopicWindow

**问题**: 当前 `_write_runtime_info()` 把 `AudioRuntimeInfo` 序列化写入 `session.tmp_storage`——这是一次性文件写入，没有心跳更新，消费者需要轮询文件。

**改用 TopicWindow**:

```python
# contracts/audio.py

class AudioRuntimeTopic(TopicModel):
    """音频捕获运行时状态。max_size=1 的 TopicWindow 承载最新状态。"""
    running: bool = False
    stream_key: str = ""
    device_name: str = ""
    device_explain: str = ""
    started_at: float = 0.0
    last_heartbeat: float = 0.0

    @classmethod
    def topic_type(cls) -> str:
        return "audio/runtime"

    @classmethod
    def default_topic_name(cls) -> str:
        return "audio/runtime"
```

捕获端 `start()` 后 pub `AudioRuntimeTopic(running=True, ...)`，周期性心跳更新 `last_heartbeat`。关闭时 pub `AudioRuntimeTopic(running=False, ...)`。

消费端:
```python
window = transport.topic_window(AudioRuntimeTopic, max_size=1)
await window.wait_started()
info = window.values()[-1]           # 最新状态
window.on_change(lambda w: ...)      # 变更回调
```

**对比**:

| | tmp_storage | TopicWindow |
|---|---|---|
| 写入 | 一次性 JSON 文件 | 持续 pub |
| 心跳 | 无 | 周期性更新 last_heartbeat |
| 读取 | 文件系统轮询 | values() / on_change() |
| 跨进程 | 天然 | Zenoh 传输天然 |
| 生命周期 | 手动 cleanup | 绑定 TopicService |

**实施**: `_write_runtime_info()` 改为 `_pub_runtime_topic()`，走 `transport.pub_topic()`。tmp_storage 路径移除。

---

### KD12: SpeechTopic — 统一语音事件协议

**问题**: ASR 有 `Recognition`，TTS 有 `TTSItem`，但没有跨输入输出的统一话语事件模型。对话上下文需要知道"谁在什么时候说了什么"，无论方向。

**SpeechTopic 定义** (放在 `contracts/speech.py`，与现有 `Speech`/`TTS` 并列):

```python
class SpeechTopic(TopicModel):
    """一段话语事件 — 语音对话流中的单个节点。

    Topic 语义: 每一条是一个完整语段或流式中间结果。
    流式 ASR 的中间结果更新同一个 event (batch_id + seq 不变)，
    最终结果 is_final=True。TTS 开始播放时产生 event，完成后
    is_final=True。

    TopicWindow[SpeechTopic] 承载对话上下文 — 最近 N 条话语
    构成当前语音交互的完整上下文窗口。
    """

    # ── 话语内容 ──
    text: str = ""
    is_delta: bool = False              # 流式增量；False = 完整句
    is_final: bool = False              # 这句话说完了

    # ── 说话人 ──
    speaker_id: str = ""                # 唯一标识
    speaker_name: str = ""              # 显示名称
    role: str = ""                      # human / ghost / assistant / system

    # ── 时序与追踪 ──
    batch_id: str = ""                  # 同一次语音会话的批次ID
    seq: int = 0                        # batch 内序列号
    timestamp: float = 0.0              # 事件时间

    # ── 可选关联 ──
    lang: str = "zh"
    audio_key: str | None = None        # 关联的 PCM 流 key
    commit_reason: str = ""             # vad_timeout / manual / tts_done

    @classmethod
    def topic_type(cls) -> str:
        return "speech"

    @classmethod
    def default_topic_name(cls) -> str:
        return "speech"
```

**与现有 Recognition 的关系**:

| Recognition (ASR 内部) | SpeechTopic (统一协议) |
|---|---|
| batch_id | batch_id |
| seq | seq |
| text | text |
| sentence | is_final |
| is_last | is_final (语义合并) |
| created | timestamp |
| commit_reason | commit_reason |
| — | speaker_id, speaker_name, role, is_delta, lang, audio_key |

`Recognition` 是 ASR 内部的领域模型。listener 的 `on_recognition` 回调拿到 `Recognition` 后，映射成 `SpeechTopic` 再 pub。

**TTS 侧**: Speech 播放一句话时 pub `SpeechTopic(role="ghost", is_delta=False)`，完成后更新 `is_final=True`。

**TopicWindow 承载对话上下文**: `window.values()` 返回最近 N 条话语，构成"谁说了什么"的完整上下文窗口。消费者可以按 role/speaker_id 过滤，按 timestamp 排序。

---

### KD13: AudioSignal — 音频感知信号接入 Mindflow

**问题**: ASR 识别到用户说话后，需要打断当前 TTS 播放或思考。现有 `InputSignal(signal_name="input")` 是通用输入信号，不区分音频感知语义。

**AudioSignal 定义** (放在 `contracts/audio.py`):

```python
class AudioAction(str, Enum):
    SPEECH_STARTED = "speech_started"      # 检测到有人开始说话
    SPEECH_DELTA = "speech_delta"          # 流式识别增量更新
    SPEECH_FINAL = "speech_final"          # 一句话说完
    WAKE_WORD = "wake_word"                # 唤醒词检测
    AUDIO_ALERT = "audio_alert"            # 环境异常音

class AudioSignal(SignalMeta):
    """音频感知信号。从 listener → mindflow 的注意力抢占路径。"""

    action: AudioAction
    speech_topic: SpeechTopic | None = None

    @classmethod
    def signal_name(cls) -> str:
        return "audio"

    @classmethod
    def priority(cls) -> Priority:
        return Priority.WARNING  # 高于默认 NOTICE，确保能抢占普通思考
```

**流式 ASR 的注意力抢占**: 利用 mindflow 已有的 `complete` 机制:

```
ASR 第一包识别结果
  → AudioSignal(action=SPEECH_DELTA, speech_topic=SpeechTopic(text="你好", is_delta=True))
  → Signal(complete=False, priority=WARNING)
  → Nucleus → Impulse(complete=False)
  → challenge current Attention
      ├─ 抢占成功 → 占据注意力槽位，等待 complete=True
      └─ TTS 实现 Preemptable → attenuate() 被打断

ASR 最终结果
  → AudioSignal(action=SPEECH_FINAL, speech_topic=SpeechTopic(text="你好世界", is_final=True))
  → Signal(complete=True, same id)
  → Impulse(complete=True) → 解锁 think-act loop
  → Ghost 开始思考用户说了什么
```

**初期实现**: 不需要特殊的"首包打断"语义——直接用 mindflow 现有的 `complete=False → 抢占 → 等待 → complete=True → 解锁` 路径。Mindflow 已有 BufferNucleus 可配置监听 `"audio"` 信号。后续可优化为专属 AudioNucleus。

**与 Preemptable 的协作**: listener 发射 AudioSignal 后，mindflow 的 Attention challenge 如果返回 preempt，调用当前 Action 关联组件的 `Preemptable.attenuate()`。Signal/Impulse 自身不携带回调——能力发现走 Protocol。

---

### KD14: 可选能力界面 — Protocols 而非硬编码回调

**原则**: 回调不写死在抽象类里。音频体系的能力通过 Protocol 标记，组件按需实现。系统通过 `isinstance` 做能力发现。一期全部走 input signal 机制，Signal 就是回调。

**四个可选 Protocol** (放在 `contracts/audio.py`):

```python
class Preemptable(Protocol):
    """能力标记: 可被注意力抢占打断。TTS/Speech/播放器可选实现。
    Mindflow 在 attention challenge 返回 preempt 后调用 attenuate()，
    恢复时调用 resume()。"""
    def attenuate(self) -> None: ...
    def resume(self) -> None: ...

class SpeechEventEmitter(Protocol):
    """能力标记: 可以广播 SpeechTopic 事件。Listener/ASR 实现。"""
    @property
    def topic_service(self) -> TopicService: ...

class SpeechEventReceiver(Protocol):
    """能力标记: 可以接收 SpeechTopic 事件。对话上下文/字幕/记忆模块实现。"""
    def on_speech_topic(self, topic: SpeechTopic) -> None: ...

class AudioRuntimeReporter(Protocol):
    """能力标记: 可以上报运行时状态。Capture/Player 实现。"""
    def runtime_info(self) -> AudioRuntimeTopic: ...
```

**不是强制接口**: 组件可以选择实现它们。例如:
- 一个简单的波形可视化不实现任何 Protocol
- TTS 实现 `Preemptable` + `SpeechEventEmitter`
- ASR listener 实现 `SpeechEventEmitter`
- 对话上下文模块实现 `SpeechEventReceiver`

**一期不实现复杂回调链** — 通过 mindflow 的 `add_signal()` 分发。`Preemptable` 的调用由 mindflow attention 层的 preempt hook 处理，不走自定义回调路径。

---

### KD15: 五种音频交互模式

交互模式控制音频输入何时被捕获、何时被发送。这些模式不属于 audio capture 核心——它们是交互层的关注点，通过 TUI/界面/CTML 命令控制。

| # | 模式 | 行为 | 优先级 |
|---|------|------|--------|
| 1 | **push-to-talk** | 按住按钮时捕获，松开后发送 | P1 |
| 2 | **enter-to-talk** | 回车开始捕获，回车结束发送 | P1 |
| 3 | **free-listening** | 持续聆听，VAD 自动切分后发送 | P2 |
| 4 | **manual-send** | 只追加文本到输入区，手动点击发送 | P2 |
| 5 | **model-validate-send** | 追加但不发送，由快速模型判断是否发送 | P2 |

**与 listener 的关系**: 模式 1-3 控制 AudioCaptureSource 和 Listener 的启停时机。模式 4-5 影响 Listener → Ghost 的发送时机。实现上：
- 模式 1-3: 控制 `AudioCaptureSource.start()` / `close()` + `Listener` 生命周期
- 模式 4-5: Listener 拿到 Recognition 后不立即 emit AudioSignal，而是追加到 buffer，等待发送条件

**MVP 收敛**: 单一交互方式（push-to-talk 或 enter-to-talk），跑通 input signal → callback ghost。其余 P1/P2 按需演进。

---

### KD16: 增强能力

以下能力不在 MVP 范围，设计上预留接口:

**旁路 flash 模型滚动修正**:
- 独立于主 ASR 管线的小模型
- 持续监控已发送和未发送的语音识别文本
- 发现错误时修正（如术语纠正、上下文消歧）
- 不阻塞主识别流

**本地术语表**:
- 向 ASR 识别上下文注入领域名词（人名、地名、项目术语）
- 参考 Session 体系的新增 API: `ParameterStore`（typed shared state + CAS lock）、`Cache`（sqlite3 cross-process cache）
- 实现为独立 KeyValue 存储，listener 启动时加载

**耳机/外设通道**:
- 支持指定音频输入/输出设备
- 与进程对接: 外设进程可在独立 MOSS App 中运行
- `AudioCaptureConfig.device_pattern` 已支持设备发现，外设对接需扩展

---

## Contract

已有合约（已实现）:

```python
# contracts/audio.py

class AudioFrameMeta(BaseModel):
    rms_db: float
    bands: dict[str, float]    # "bass":-42, "mid":-28, "high":-45
    is_silent: bool

class AudioChunk(BaseModel):
    seq: int
    timestamp: float
    samples: np.ndarray
    meta: AudioFrameMeta

class AudioCaptureConfig(ConfigType):
    sample_rate: int = 44100
    channels: int = 1
    format: str = "pcm_s16le"
    frame_duration_ms: int = 50
    device_pattern: str = "blackhole"

class AudioCaptureSource(ABC):
    async def start(self) -> None: ...
    def device_explain(self) -> str: ...
    def new_consumer(self, ring_buffer_frames: int = 64) -> AudioPullLatest: ...
    def new_sequential_consumer(self, max_queue_frames: int = 128) -> AudioSequentialConsumer: ...
    async def close(self) -> None: ...

class AudioPullLatest(ABC):
    def pull_latest(self) -> AudioChunk | None: ...
    def close(self) -> None: ...

class AudioSequentialConsumer(ABC):
    def __aiter__(self): ...
    async def __anext__(self) -> AudioChunk: ...
    async def start(self) -> None: ...
    async def close(self) -> None: ...
```

新增合约（本轮设计，待实现）:

```python
# contracts/audio.py — 追加

class AudioTransport(ABC):
    """传输抽象，隔离 Matrix。见 KD10。"""
    def pub_pcm(self, chunk: bytes) -> None: ...
    def sub_pcm_callback(self, on_chunk: Callable[[bytes], None]) -> Callable[[], None]: ...
    def sub_pcm_stream(self, maxsize: int) -> StreamSubscriber: ...
    def acquire_lock(self) -> bool: ...
    def release_lock(self) -> None: ...
    def pub_topic(self, topic: TopicModel) -> None: ...
    def topic_window(self, model: type[TOPIC_MODEL], max_size: int) -> TopicWindow[TOPIC_MODEL]: ...
    @property
    def logger(self) -> logging.Logger: ...

class AudioRuntimeTopic(TopicModel):
    """运行时状态 Topic。见 KD11。"""
    running: bool = False
    stream_key: str = ""
    device_name: str = ""
    device_explain: str = ""
    started_at: float = 0.0
    last_heartbeat: float = 0.0

    @classmethod
    def topic_type(cls) -> str: return "audio/runtime"
    @classmethod
    def default_topic_name(cls) -> str: return "audio/runtime"

class AudioAction(str, Enum):
    SPEECH_STARTED = "speech_started"
    SPEECH_DELTA = "speech_delta"
    SPEECH_FINAL = "speech_final"
    WAKE_WORD = "wake_word"
    AUDIO_ALERT = "audio_alert"

class AudioSignal(SignalMeta):
    """音频感知信号。见 KD13。"""
    action: AudioAction
    speech_topic: SpeechTopic | None = None

    @classmethod
    def signal_name(cls) -> str: return "audio"
    @classmethod
    def priority(cls) -> Priority: return Priority.WARNING

class Preemptable(Protocol):
    """可选能力: 注意力抢占。见 KD14。"""
    def attenuate(self) -> None: ...
    def resume(self) -> None: ...

class SpeechEventEmitter(Protocol):
    """可选能力: 广播 SpeechTopic。见 KD14。"""
    @property
    def topic_service(self) -> TopicService: ...

class SpeechEventReceiver(Protocol):
    """可选能力: 接收 SpeechTopic。见 KD14。"""
    def on_speech_topic(self, topic: SpeechTopic) -> None: ...

class AudioRuntimeReporter(Protocol):
    """可选能力: 上报运行时状态。见 KD14。"""
    def runtime_info(self) -> AudioRuntimeTopic: ...

# contracts/speech.py — 追加

class SpeechTopic(TopicModel):
    """统一话语事件。见 KD12。"""
    text: str = ""
    is_delta: bool = False
    is_final: bool = False
    speaker_id: str = ""
    speaker_name: str = ""
    role: str = ""                    # human / ghost / assistant / system
    batch_id: str = ""
    seq: int = 0
    timestamp: float = 0.0
    lang: str = "zh"
    audio_key: str | None = None
    commit_reason: str = ""

    @classmethod
    def topic_type(cls) -> str: return "speech"
    @classmethod
    def default_topic_name(cls) -> str: return "speech"
```

---

## Implementation Progress

| Step | 内容 | 状态 | 日期 |
|------|------|------|------|
| 1 | `contracts/audio.py` — 7 符号抽象 | **done** | 2026-06-05 |
| 2 | `miniaudio_capture.py` — 核心实现 | **done** | 2026-06-05 |
| 3 | `audio_capture_provider.py` — Provider + manifests | **done** (需改为组装 Transport) | 2026-06-05 |
| 4 | Waveform App — 跨进程可视化消费者 | **done** | 2026-06-05 |
| 5 | Audio Capture App — 独立生命周期 | **done** | 2026-06-05 |
| 6 | `AudioTransport` ABC + `MatrixAudioTransport` 适配器 | **pending** — 见 KD10 | |
| 7 | `MiniAudioCaptureSource` Matrix → AudioTransport 改造 | **pending** | |
| 8 | `AudioRuntimeTopic` + TopicWindow 替代 tmp_storage | **pending** — 见 KD11 | |
| 9 | `SpeechTopic` 合约 + contracts/speech.py 补充 | **pending** — 见 KD12 | |
| 10 | `AudioSignal` + 四个 Protocol 合约 | **pending** — 见 KD13/KD14 | |
| 11 | Listener App (ASR 消费者) — SequentialConsumer + Recognizer + SpeechTopic pub + AudioSignal emit | **pending** — 见 KD9/KD12/KD13 | |
| 12 | Mindflow AudioSignal Nucleus 注册 | **pending** — 见 KD13 | |
| 13 | Speech/TTS `Preemptable` 实现 | **pending** — 见 KD14 | |
| 14 | 交互模式 MVP (push-to-talk or enter-to-talk) | **pending** — 见 KD15 | |
| 15 | 旁路 flash 模型 + 本地术语表 + 外设通道 | **P2/deferred** — 见 KD16 | |

### Step 1-5 实现细节

已保留上文完整实现细节（miniaudio callback generator 模式、FFT in-thread、设备发现、pack/unpack 序列化、Bug 修复记录等），不再重复。

### 当前架构总览

```
Ghost: apps:start sensors/audio_capture
  └─ Audio Capture App (独立进程)
       └─ Matrix.discover().run(main)
            └─ MiniAudioCaptureSource(transport, config)
                 ├─ AudioTransport (→ MatrixAudioTransport → Matrix/Session/Zenoh)
                 └─ start()
                      ├─ transport.acquire_lock()    # 进程锁
                      ├─ miniaudio.CaptureDevice.start(gen)
                      │    └─ callback: FFT → pack_chunk → transport.pub_pcm()
                      └─ transport.pub_topic(AudioRuntimeTopic(running=True))

Waveform App (独立进程)
  └─ session.get_stream("audio/pcm")
       └─ async for sample → unpack metadata → 终端波形渲染

Listener App (独立进程, 待实现)
  └─ source.new_sequential_consumer()
       └─ async for chunk → Recognizer → Recognition
            ├─ pub SpeechTopic(topic_service)
            └─ AudioSignal → mindflow.add_signal()
                 └─ challenge current Attention
                      └─ preempt → Speech.attenuate() (if Preemptable)

Ghost: apps:stop sensors/audio_capture
  └─ SIGTERM → capture.close() → transport.release_lock() + AudioRuntimeTopic(running=False)
```

---

## Handoff — 后续伙伴推进路径

### 优先级

1. **P0 — Matrix 解耦** (Step 6+7): 定义 `AudioTransport` ABC，实现 `MatrixAudioTransport`，改造 `MiniAudioCaptureSource`。这是架构方向性改动，其他都在此基础上。

2. **P0 — 新增合约** (Step 9+10): `SpeechTopic` + `AudioSignal` + 四个 Protocol 落地到 contracts 文件。纯声明式，不涉及实现。

3. **P1 — AudioRuntimeTopic 替代 tmp_storage** (Step 8): 独立的 TopicWindow 广播，不依赖 listener。

4. **P1 — Listener MVP** (Step 11): 基于 `AudioSequentialConsumer` 的 ASR 消费者。复用现有 `ghoshell_moss_contrib.asr` 的 Recognizer 实现。跑通 PCM → ASR → SpeechTopic → AudioSignal 全链路。

5. **P1 — Mindflow 注册** (Step 12): 配置 Nucleus 监听 `"audio"` 信号，初期用 `BufferNucleus`。

6. **P1 — 单一交互模式** (Step 14): push-to-talk，TUI 按钮或 CLI。跑通 input signal → callback ghost。

7. **P2 — Preemptable 集成** (Step 13): Speech/TTS 实现 `Preemptable`，AudioSignal 抢占时打断播放。

### .design 文档

请在 `contracts/` 下产出一份 `.design` 文档，包含:
- 数据流全图（PCM 流 + SpeechTopic 广播 + AudioSignal 抢占三条路径）
- 组件交互时序（capture → listener → mindflow → ghost）
- 状态机（AudioRuntimeTopic 生命周期: stopped → starting → running → heartbeat → closing → stopped）
- Task 拆分（上述 Step 6-15 的详细子任务）

### 设计原则提醒

- **依赖方向**: contracts → ABC，host → adapter。audio 核心零 Matrix import。
- **TOPIC 不是 DELTA**: SpeechTopic 是完整语段。流式 ASR 用 `is_delta` + `batch_id` + `seq` 追踪增量更新，不是逐 token topic。
- **Protocol 不是强制**: 组件选择实现。能力发现走 `isinstance`。回调走 Signal。
- **MVP 收敛**: 单一交互方式 + 一条全链路跑通。其余后续。

---

*架构设计: DeepSeek V4 与人类工程师, 2026-06-03*
*补充: SequentialAudioConsumer 设计、listener 依赖关系、运行时稳定性约束 — Claude Opus 4.7 与人类工程师, 2026-06-04*
*实现: Step 1-2 contracts + miniaudio capture core — Claude Opus 4.7 与人类工程师, 2026-06-05*
*Bug 修复: int16 归一化 + 波形渲染宽 bar — Claude Opus 4.7 与人类工程师, 2026-06-05*
*设计收敛: Matrix 解耦 AudioTransport、TopicWindow 替代 tmp_storage、SpeechTopic 统一协议、AudioSignal 接入 mindflow、五种交互模式、四项可选 Protocol、MVP 边界收敛 — deepseek-v4 与人类工程师, 2026-06-07*
