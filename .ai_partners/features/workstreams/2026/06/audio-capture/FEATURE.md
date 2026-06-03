---
created: 2026-06-03
depends: []
description: 基于 miniaudio 的系统音频捕获源，原始 PCM 经 Zenoh 广播，消费者按需加工（波形可视化、ASR、剪辑）。 ConfigType
  格式共识 + tmp_storage 运行时发现 + Provider 注入。
milestone: null
priority: P1
status: draft
status_note: 讨论完成，交由其他 AI 实例实现
title: Audio Capture — 系统音频感知通道
updated: '2026-06-04'
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

- Contract: `ghoshell_moss.contracts.audio`（待创建）
- 实现: `ghoshell_moss.host.speech.capture.miniaudio_capture`（待创建）
- Provider: `ghoshell_moss.host.providers.audio_capture_provider`（待创建）
- 播放端对称参考: `ghoshell_moss.host.speech.player.miniaudio_player:MiniAudioStreamPlayer`
- 播放 Provider 参考: `ghoshell_moss.host.providers.audio_player_provider:AudioPlayerProvider`
- 进程锁: `ghoshell_moss.contracts.workspace:FileLocker`
- 下游消费 feature: 波形 Channel（本 feature 不包含，但为此提供原始 PCM 基础）

## Key Decisions

### KD1: 原始 PCM 走 Zenoh，不做特征提取

捕获源只产出原始 PCM（`AudioChunk.samples`）。每个 chunk 附加一份轻量元信息（`AudioFrameMeta`：rms_db + bands + is_silent），在捕获端 FFT 算一次，随 chunk 一起走。消费者只读自己需要的：

| 消费者 | 读什么 |
|---|---|
| 波形 Channel (未来) | meta（RMS + bands），不重算 PCM |
| ASR (未来) | samples |
| AI 感知 | meta 快速判断，需要时再读 samples |
| 音频剪辑 (未来) | samples |

**接受**: meta 是 frame 的副产品，不是独立流。和 sample 打包在同一个 AudioChunk 里，一次 Zenoh pub 同时送出。
**拒绝**: 捕获源内建 N 种加工管线——加工逻辑属于消费者，不属于采集节点。捕获节点应为极简管道。
**拒绝**: 只广播特征帧不广播 PCM——ASR 和剪辑需要原始采样。86 KB/s（44.1kHz mono 16bit）对 Zenoh TCP localhost 是零头。

### KD2: ConfigType 格式共识 + tmp_storage 运行时发现

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

### KD3: AudioPullLatest 消费者模型，pull_next 推迟

消费者侧只定义 `AudioPullLatest`——非阻塞拿最新帧。波形可视化、AI 按需感知适用。

```python
class AudioPullLatest(ABC):
    def pull_latest(self) -> AudioChunk | None: ...
    """非阻塞，总是立刻返回。可能返回与上次相同帧（期间无新数据）。"""
    def close(self) -> None: ...
```

`pull_next`（顺序消费 + gap 通知）推迟到 ASR/录音等真实需求出现时再加。"用不到尽量不加抽象"。

### KD4: ring_buffer_frames 是 consumer 实例参数，不是全局 ConfigType

`AudioCaptureSource.new_consumer(ring_buffer_frames=64)` 创建消费者实例。每个 consumer 内部独立 ring buffer + Zenoh subscription，互不干扰。慢 consumer 被写满时丢最老帧（`drop_policy: oldest`）。

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

### KD8: Contract 定义

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
    async def close(self) -> None: ...

class AudioPullLatest(ABC):
    def pull_latest(self) -> AudioChunk | None: ...
    def close(self) -> None: ...
```

## Implementation Notes

### 实现顺序

1. `contracts/audio.py` — 所有抽象定义
2. `host/speech/capture/miniaudio_capture.py` — MiniAudioCaptureSource 实现
3. `host/providers/audio_capture_provider.py` — Provider 注册
4. 波形 Channel（后续 feature）

### miniaudio CaptureDevice 回调模式

和 `MiniAudioStreamPlayer._make_generator()` 对称，使用 miniaudio 的 callback/ring buffer 模式。miniaudio 回调在独立线程中触发，需线程安全地将 PCM 写入 ring buffer，Zenoh pub 在 asyncio 侧读取。

### 设备发现

`device_pattern` 用于匹配设备名。枚举所有 `miniaudio.devices()` 中 `is_default=False` 且名称匹配的 capture 设备。若无匹配，fallback 到默认输入设备。

### 与 speech-governance feature 的关系

`speech-governance` (in-progress, P2) 在做 TTS/播放的多后端解耦。audio-capture 是播放端的对称补充，在 `host/speech/` 下加 `capture/` 目录与之并列。两者共享 miniaudio 后端，但各自独立 Provider。

---

*架构设计: DeepSeek V4 与人类工程师, 2026-06-03*