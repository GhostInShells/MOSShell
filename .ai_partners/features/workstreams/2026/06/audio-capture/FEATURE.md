---
created: 2026-06-03
depends: []
description: 基于 miniaudio 的系统音频捕获源，原始 PCM 经 Zenoh 广播，消费者按需加工（波形可视化、ASR、剪辑）。 ConfigType 格式共识 + tmp_storage 运行时发现 + Provider 注入。
milestone: null
priority: P1
status: in-progress
status_note: 全部 5 步完成，MatrixLifecycleObject 自动启停，waveform App 跨进程消费已验证
title: Audio Capture — 系统音频感知通道
updated: '2026-06-05'
---

# Audio Capture

> Ghost 现在只有嘴（TTS），没有耳朵。此 feature 补上音频输入感知链路：
> 系统音频 → miniaudio CaptureDevice → 原始 PCM → Zenoh stream → 消费者各自加工。
> 纯捕获，不内建 N 种加工逻辑——加工是消费者的事。

## Motivation

1. **Ghost 需要听觉感知**：当前 speech Channel 只管输出（TTS），无法感知系统音频。波形可视化、ASR、音频剪辑等场景全部依赖音频输入。

2. **miniaudio 已在项目中作为默认音频后端**（`MiniAudioStreamPlayer`），同样的库对称支持 `CaptureDevice`，零新依赖。

3. **捕获与加工必须解耦**：不要把所有处理逻辑塞进捕获节点。捕获只做一件事——干净地交出原始 PCM。FFT/ASR/波形渲染是各自消费者的事。

4. **格式共识编译期已知**：用 ConfigType 定义采样参数，消费者读配置即知流格式，不需要运行时协商。

5. **进程安全**：`start()` 用 `FileLocker` 拿进程锁，避免多进程重复打开同一设备。不强制走 App 包装，但可包成 App 由模型控制启停。

## Design Index

- Contract: `ghoshell_moss.contracts.audio` — 7 个符号
- 实现: `ghoshell_moss.host.speech.capture.miniaudio_capture` — MiniAudioCaptureSource + 消费者
- Provider: `ghoshell_moss.host.providers.audio_capture_provider` — singleton + IoC
- 生命周期: 独立 App（`.moss_ws/apps/sensors/audio_capture/`）— Ghost 通过 `apps:start`/`apps:stop` 控制启停，不再随 Matrix 自动拉起
- 播放端对称参考: `ghoshell_moss.host.speech.player.miniaudio_player:MiniAudioStreamPlayer`
- 播放 Provider 参考: `ghoshell_moss.host.providers.audio_player_provider:AudioPlayerProvider`
- 进程锁: `ghoshell_moss.contracts.workspace:FileLocker`（跨进程防重复打开设备）
- 下游消费: waveform App（`.moss_ws/apps/sensors/waveform/`）、listener（ASR 消费者，待实现）

## Key Decisions

### KD1: 原始 PCM 走 Zenoh，不做特征提取

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

TopicService 设计意图是"秒级大脑事件"（见 `Topic` docstring）。PCM 流是 50ms/帧、86 KB/s 的连续传感器数据——语义完全不匹配。Zenoh 原生 `pub`/`sub` 设计场景就是 MB/s 级传感器流，有序、有可靠性 QoS、有 backpressure。

**接受**: audio-capture 用 Zenoh 原生 API（`pub_stream_delta` / `sub_stream`），TopicService 只用于状态变更通知（AudioRuntimeInfo 的 running/heartbeat 等）。
**拒绝**: PCM 走 TopicService——语义错位，且 TopicService 的 Subscriber 模型（`maxsize=0` 无限堆积或 `maxsize=1` 只保留最新）不适合流式消费。

### KD3: ConfigType 格式共识 + tmp_storage 运行时发现

两层发现机制，各司其职：

```
AudioCaptureConfig (ConfigType)     → moss manifests configs 可见，编译期已知
  sample_rate, channels, format, frame_duration_ms, device_pattern

AudioRuntimeInfo (tmp_storage)      → App 启动时写入，运行时被动获取
  running, stream_key, device_name, started_at, last_heartbeat
```

ConfigType 回答"格式是什么"——消费者编译期就知道怎么解析流。tmp_storage 回答"在哪、活着吗"——消费者运行时被动 get，不依赖 Session Output 推送。

**接受**: ConfigType 由消费者定义和使用，不是 App 私有的。格式是共识，不属于任何一方。
**拒绝**: Session Output 做运行时通知——Output 是 mindflow 交互通道，做基础设施元信息推送是语义错位。tmp_storage 天然适合"只查一次"的知识 + 进程锁场景。

### KD4: 两类消费者模型 — pull_latest 与 sequential

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
    async def close(self) -> None: ...
```

`pull_latest` 内部: ring buffer + 非阻塞读，写满时丢最老帧。
`sequential` 内部: `asyncio.Queue(maxsize=N)` + Zenoh sub callback relay。队列满时 `put()` 阻塞 → 自然背压 → Zenoh 侧自动丢弃。cancel 后 finally 发 sentinel → 消费者优雅退出。

**实现草稿**:

```python
class MiniAudioSequentialConsumer(AudioSequentialConsumer):
    def __init__(self, zenoh_sub, maxsize=128):
        self._queue = asyncio.Queue(maxsize=maxsize)  # 128帧 * 50ms = 6.4s
        self._sub = zenoh_sub
        self._task: asyncio.Task | None = None

    async def start(self):
        self._task = asyncio.create_task(self._relay())

    async def _relay(self):
        try:
            async for sample in self._sub:
                chunk = AudioChunk.from_sample(sample)
                await self._queue.put(chunk)  # 队列满阻塞, 自然背压
        except asyncio.CancelledError:
            pass
        finally:
            self._queue.put_nowait(None)  # sentinel

    async def __anext__(self) -> AudioChunk:
        chunk = await self._queue.get()
        if chunk is None:
            raise StopAsyncIteration
        return chunk

    async def close(self):
        if self._task:
            self._task.cancel()
```

**接受**: consumer 级参数——不同消费者有不同容忍度（波形显示 32 帧够了，ASR 可能需要 128）。
**拒绝**: 放在全局 ConfigType——那是对所有消费者的强制约束。

### KD5: Provider 注入 Matrix，Singleton 生命周期

```python
class AudioCaptureProvider(Provider[AudioCaptureSource]):
    def singleton(self) -> bool: return True
    def factory(self, con: IoCContainer) -> AudioCaptureSource:
        matrix = con.force_fetch(Matrix)
        config = con.force_fetch(ConfigStore).get_or_create(AudioCaptureConfig())
        return MiniAudioCaptureSource(matrix=matrix, config=config)
```

和 `AudioPlayerProvider` 对称：播放端提供 `StreamAudioPlayer`，捕获端提供 `AudioCaptureSource`。

### KD6: start() 底层加进程锁，不强依赖 App 包装

`start()` 实现层通过 `FileLocker` 拿进程锁。可以直接作为 Provider 提供的 singleton 使用，也可以包成 App 由模型通过 AppStoreChannel 控制启停——不互斥。

### KD7: 独立 Zenoh 会话是远期优化，MVP 用主 Session

原始 PCM 86 KB/s 在主 Zenoh session 上和 topic/signal/logos 混跑不会造成阻塞——Zenoh 设计场景是 MB/s 级传感器流。但音频有自己的 QoS 特征（允许丢帧、延迟敏感度低），长期应物理隔离：

```
端口 20770: 主控平面
端口 20775: 音频数据平面 (未来)
```

MVP 用主 Session 的 `pub_stream_delta` / `sub_stream`。代码结构把 session 做成可注入的，未来换实例即可。

### KD8: 运行时稳定性优先 — 允许丢帧，禁止泄漏

流式 ASR 的场景：180-210ms chunk 直接上报火山引擎。偶尔丢帧只影响那一小段识别质量，不影响系统运行。真正危险的是：

1. **内存泄漏** — 消费者跟不上，队列无限堆积
2. **崩溃传播** — WebSocket 断开导致整个 task 树崩溃
3. **资源泄漏** — cancel 后 WebSocket/audio stream 没收干净

`sequential` consumer 用有界 `asyncio.Queue` 解决 (1)——满了就阻塞 pub 侧，Zenoh 自动丢老帧。用 asyncio task + try/except/finally + sentinel 解决 (2)(3)。

### KD9: 与 listener feature 的依赖关系

audio-capture 是 PCM 源，listener 是 ASR 消费者。依赖链：

```
AudioCaptureSource (本 feature)
  └─ .new_sequential_consumer(ring_buffer_frames=128)
       └─ AudioSequentialConsumer (async iterable)
            └─ listener 内层循环: async for chunk in consumer
                 └─ feed ASR engine → callback(SpeechRecognition)
```

listener 不拥有麦克风——它通过 audio-capture 的 consumer 获取 PCM。两个 feature 并行开发，只要 `AudioCaptureSource` + `AudioSequentialConsumer` 的 contract 先稳定。

## Contract

```python
# contracts/audio.py

class AudioFrameMeta(BaseModel):
    """捕获端预计算，每帧算一次，消费者共享"""
    rms_db: float
    bands: dict[str, float]    # "bass":-42, "mid":-28, "high":-45
    is_silent: bool

class AudioChunk(BaseModel):
    seq: int                    # 单调递增
    timestamp: float
    samples: np.ndarray         # (frame_samples, channels), 原始 PCM
    meta: AudioFrameMeta

class AudioCaptureConfig(ConfigType):
    sample_rate: int = 44100
    channels: int = 1
    format: str = "pcm_s16le"
    frame_duration_ms: int = 50
    device_pattern: str = "blackhole"

class AudioRuntimeInfo(BaseModel):
    running: bool
    stream_key: str
    device_name: str
    device_explain: str         # 自然语言设备描述
    started_at: float
    last_heartbeat: float

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

## Implementation Notes

### 实施进度

| Step | 内容 | 状态 | 日期 |
|------|------|------|------|
| 1 | `contracts/audio.py` — 抽象定义 | **done** | 2026-06-05 |
| 2 | `host/speech/capture/miniaudio_capture.py` — 核心实现 | **done** | 2026-06-05 |
| 3 | `host/providers/audio_capture_provider.py` — Provider + manifests | **done** | 2026-06-05 |
| 4 | Waveform App（`.moss_ws/apps/sensors/waveform/`）— 跨进程可视化消费者 | **done** | 2026-06-05 |
| 5 | Audio Capture App（`.moss_ws/apps/sensors/audio_capture/`）— 独立生命周期，Ghost 可控启停 | **done** | 2026-06-05 |
| 6 | listener（ASR 消费者，后续 feature，依赖 `AudioSequentialConsumer`） | pending | |

### Step 1-2 实现细节

**contracts/audio.py** — 7 个符号，已在 `contracts/__init__.py` 中导出：

- `AudioFrameMeta(BaseModel)` — rms_db, bands(bass/mid/high), is_silent
- `AudioChunk(BaseModel)` — seq, timestamp, samples(ndarray), meta
- `AudioCaptureConfig(ConfigType)` — conf_name="audio_capture"，默认 44100Hz/mono/pcm_s16le/50ms/blackhole
- `AudioRuntimeInfo(BaseModel)` — running, stream_key, device_name, device_explain, started_at, last_heartbeat
- `AudioCaptureSource(ABC)` — start(), device_explain(), new_consumer(), new_sequential_consumer(), close()
- `AudioPullLatest(ABC)` — pull_latest(), close()
- `AudioSequentialConsumer(ABC)` — start(), close(), \_\_aiter\_\_()/\_\_anext\_\_()

**host/speech/capture/miniaudio_capture.py** — 核心实现：

- `MiniAudioCaptureSource` — miniaudio CaptureDevice callback generator 模式（与 PlaybackDevice 对称），FFT 在回调线程中计算，AudioChunk 序列化后通过 `session.pub_stream_delta()` 发布到 Zenoh
- `_MiniAudioPullLatest` — ring buffer（collections.deque maxlen），`session.sub_stream()` 回调写入，`pull_latest()` 非阻塞读
- `MiniAudioSequentialConsumer` — 封装 `session.get_stream()` + `StreamSubscriber`，janus.Queue 桥接线程，`async for` 消费
- `_compute_frame_meta()` — RMS + 3-band FFT (bass/mid/high) + 静音判定（阈值 -50dB）
- `_pack_chunk()` / `_unpack_chunk()` — 二进制序列化：`[4B meta_len(uint32 BE)][JSON meta UTF-8][PCM int16 LE]`
- `start()` 通过 `ws.lock("audio_capture")` 拿进程锁
- `_write_runtime_info()` 将 `AudioRuntimeInfo` 写入 `session.tmp_storage`

### miniaudio CaptureDevice 回调模式

和 `MiniAudioStreamPlayer._make_generator()` 对称，使用 miniaudio 的 callback generator 模式。miniaudio 在内部线程中通过 `gen.send(raw_bytes)` 传递捕获的 PCM 数据。回调中直接计算 FFT、序列化、Zenoh pub——Zenoh session 本身线程安全，无需额外桥接。

### 设备发现

`device_pattern` 用于匹配设备名。枚举所有 `miniaudio.devices()` 中 `is_default=False` 且名称匹配的 capture 设备。若无匹配，fallback 到默认输入设备。

### Step 3: Provider + manifests

**`host/providers/audio_capture_provider.py`** — `AudioCaptureProvider(Provider[AudioCaptureSource])`，singleton=True。factory 从 IoC 获取 Matrix + ConfigStore，创建 `MiniAudioCaptureSource`。

**manifests 注册**：
- `.moss_ws/src/MOSS/manifests/providers.py` — `capture_source_provider = AudioCaptureProvider()`
- `.moss_ws/src/MOSS/manifests/configs.py` — `audio_capture_config = AudioCaptureConfig()`

### Step 4: Waveform App（跨进程消费者）

**`.moss_ws/apps/sensors/waveform/`** — 独立 MOSS App，通过 `session.get_stream("audio/pcm")` 跨进程订阅 Zenoh PCM 流。只解析 metadata（RMS + 3-band），终端渲染 unicode 波形条。验证了 capture → Zenoh → 跨进程消费者的完整链路。

启动：`moss apps test sensors/waveform`

渲染：40 字符宽 bar（`█` + 末位 1/8 精度部分块 `▏▎▍▌▋▊▉`），bass/mid/high + RMS 四行各自独立 bar。dB 范围 -60~0 dBFS，-50 dBFS 以下标 silent。

### 实测 Bug 修复 (2026-06-05)

**Bug 1: int16 样本未归一化导致 dB 值错误**。`_compute_frame_meta()` 直接将 int16 样本（值域 -32768~32767）cast 为 float64 后算 RMS，未除以 32768.0 归一化到 [-1, 1]，导致 dB 值为正数十 dB 的无意义值。修复：`samples.astype(np.float64) / 32768.0`，dB 变为标准 dBFS（0 = 满幅，负数 = 低于满幅）。`miniaudio_capture.py:41`。

**Bug 2: 波形渲染只有 1 字符宽**。每个频段只用单个 Unicode bar 字符（`▃`），视觉上几乎无波动。修复：改为 40 字符宽 bar，末位用部分块字符（`▏▎▍▌▋▊▉█`）做 1/8 精度平滑过渡，四行（bass/mid/high/rms）各自独立 bar。`main.py`。

### Step 5: Audio Capture App（独立生命周期）

**设计决策**：capture 不随 Matrix 自动启停，而是独立 App，由 Ghost 通过 `apps:start sensors/audio_capture` / `apps:stop sensors/audio_capture` 控制。原因：capture 是可选能力，不是每次 Matrix 启动都需要打开麦克风，应像波形可视化一样按需拉起。

**方案**：创建 `.moss_ws/apps/sensors/audio_capture/`，入口 `main.py` 获取 Matrix → 创建 `MiniAudioCaptureSource` → `start()` → `stop_event.wait()` 阻塞 → 收到信号后 `close()`。从 `host/impl.py` 移除 `register_lifecycle_objects(AudioCaptureSource)`，从 `MiniAudioCaptureSource` 移除 `__aenter__`/`__aexit__`。

**早期方案（已废弃）**: MatrixLifecycleObject 集成。`MiniAudioCaptureSource` 实现 `MatrixLifecycleObject` 协议，`Host.__init__` 注册 `AudioCaptureSource` 类型，Matrix 启动时自动进入 async context。废弃原因：capture 与 Matrix 生命周期过度耦合，Ghost 无法控制捕获启停。

**Bug 修复**：`host/matrix.py:681` 将 `lifecycle`（类型）改为 `lifecycle_obj`（实例），修复类型注册时对 ABC 类调 `enter_async_context` 的 TypeError。

### 当前架构总览

```
Ghost: apps:start sensors/audio_capture
  └─ Audio Capture App (独立进程)
       └─ Matrix.discover().run(main)
            └─ MiniAudioCaptureSource(matrix, config)
                 └─ start()
                      ├─ FileLocker.acquire()  # 进程锁
                      ├─ miniaudio.CaptureDevice.start(gen)
                      │    └─ callback: FFT → pack_chunk → session.pub_stream_delta("audio/pcm")
                      └─ _write_runtime_info() → tmp_storage
                 └─ stop_event.wait()  # 阻塞，等待停止信号
                 └─ close()  # finally

Waveform App (独立进程)
  └─ session.get_stream("audio/pcm")
       └─ async for sample → unpack metadata → 终端波形渲染

Ghost: apps:stop sensors/audio_capture
  └─ 发送 SIGTERM → capture.close()
```

### 与 speech-governance feature 的关系

`speech-governance` (in-progress, P2) 在做 TTS/播放的多后端解耦。audio-capture 是播放端的对称补充，在 `host/speech/` 下加 `capture/` 目录与之并列。两者共享 miniaudio 后端，但各自独立 Provider。

---

*架构设计: DeepSeek V4 与人类工程师, 2026-06-03*
*补充: SequentialAudioConsumer 设计、listener 依赖关系、运行时稳定性约束 — Claude Opus 4.7 与人类工程师, 2026-06-04*
*实现: Step 1-2 contracts + miniaudio capture core — Claude Opus 4.7 与人类工程师, 2026-06-05*
*Bug 修复: int16 归一化 + 波形渲染宽 bar — Claude Opus 4.7 与人类工程师, 2026-06-05*
