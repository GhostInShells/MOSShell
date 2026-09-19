"""
Audio capture contracts — shared abstractions for system audio input.

Capture source → raw PCM → transport → consumers (ASR, waveform, AI perception).
"""
from abc import ABC, abstractmethod

import numpy as np
import threading
from pydantic import BaseModel, Field
from typing import AsyncIterator, NamedTuple
from typing_extensions import Self

from ghoshell_moss.contracts.configs import ConfigType

__all__ = [
    "AudioFrameMeta",
    "AudioChunk",
    "AudioCaptureConfig",
    "AudioCaptureSource",
    "AudioPullLatest",
    "AudioSequentialConsumer",
    "resample",
    "AudioSpectrum",
    "compute_spectrum",
    "LatestAudioWindow",
    "AUDIO_SAMPLE_INTERVAL",
]


def resample(audio: np.ndarray, *, origin_rate: int, target_rate: int) -> np.ndarray:
    """线性插值采样率转换. 同率时原样返回.

    输入/输出双侧共用 (capture→ASR 与 TTS→player), 是 host 层的音频桥接工具.
    """
    if origin_rate == target_rate:
        return audio
    if not isinstance(audio, np.ndarray):
        raise TypeError("audio must be numpy ndarray")
    if origin_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rate must be greater than 0")
    target_len = int(len(audio) * target_rate / origin_rate)
    x_orig = np.arange(len(audio))
    x_target = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(x_target, x_orig, audio).astype(np.int16)


# AudioSampleTopic 广播 cadence (秒) — 5Hz. 生产侧 task 按此 tick.
AUDIO_SAMPLE_INTERVAL = 0.2


class AudioSpectrum(NamedTuple):
    """一帧频谱采样摘要 — compute_spectrum 的返回."""

    rms_db: float
    peak: float
    spectrum_bins: list[float]
    waveform: list[float]


def compute_spectrum(
        pcm: np.ndarray,
        *,
        n_bins: int = 16,
        n_wave: int = 128,
) -> AudioSpectrum:
    """从 int16 PCM 计算频谱采样摘要 (听侧/说侧共用).

    - ``rms_db`` / ``peak`` 从原始信号算 (真实响度, 不因加窗衰减).
    - ``spectrum_bins`` 从去直流 + Hann 窗后的信号 FFT 算, 避免直流分量污染最低频 bin.
    - ``waveform`` 从去直流信号峰值保持下采样到 ``n_wave`` 点, 供 ECG/心跳线绘制.
    """
    f32 = pcm.astype(np.float64) / 32768.0
    if f32.size == 0:
        return AudioSpectrum(0.0, 0.0, [-96.0] * n_bins, [0.0] * n_wave)

    rms = float(np.sqrt(np.mean(f32 ** 2)))
    rms_db = 20.0 * np.log10(max(rms, 1e-10))
    peak = float(np.max(np.abs(f32)))

    centered = f32 - f32.mean()
    windowed = centered * np.hanning(centered.size)
    fft = np.abs(np.fft.rfft(windowed))
    n_fft = len(fft)

    bins: list[float] = []
    for i in range(n_bins):
        lo = int(i * n_fft / n_bins)
        hi = int((i + 1) * n_fft / n_bins)
        db = 20.0 * np.log10(max(float(fft[lo:hi].mean()), 1e-10))
        bins.append(round(db, 1))

    waveform = _downsample_waveform(centered, n_wave)
    return AudioSpectrum(round(rms_db, 1), round(peak, 3), bins, waveform)


def _downsample_waveform(x: np.ndarray, n: int) -> list[float]:
    """峰值保持下采样到 ``n`` 点: 每桶取 |幅值| 最大的元素, 保留符号."""
    if x.size < n:
        return x.tolist()
    bucket = x.size // n
    trimmed = x[: bucket * n].reshape(n, bucket)
    idx = np.argmax(np.abs(trimmed), axis=1)
    picked = np.take_along_axis(trimmed, idx[:, None], axis=1).ravel()
    return [round(float(v), 3) for v in picked]


class LatestAudioWindow:
    """线程安全的"最新窗口"累积器 — 累积采样, ``take()`` 时一次性取出并复位.

    供 AudioSampleTopic 生产侧用 (latest-value-wins, 无队列): producer 回调跨线程
    ``append``, event loop task 每 ``AUDIO_SAMPLE_INTERVAL`` ``take`` 一次.
    ``stale`` 标记自上次 take 后是否有新数据, 防止无新数据时重发同一窗口.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chunks: list[np.ndarray] = []
        self._stale = False

    def append(self, samples: np.ndarray) -> None:
        with self._lock:
            self._chunks.append(samples)
            self._stale = True

    def take(self) -> np.ndarray | None:
        """若自上次 take 后有新数据, 返回拼接后的窗口并复位; 否则返回 None."""
        with self._lock:
            if not self._stale or not self._chunks:
                return None
            pcm = np.concatenate(self._chunks)
            self._chunks.clear()
            self._stale = False
            return pcm


class AudioFrameMeta(BaseModel):
    """Per-frame metadata computed once at capture, shared by all consumers."""

    rms_db: float = 0.0
    bands: dict[str, float] = Field(default_factory=lambda: {"bass": -96, "mid": -96, "high": -96})
    is_silent: bool = True


class AudioChunk(BaseModel):
    """One frame of captured audio — raw PCM plus precomputed metadata."""

    model_config = {"arbitrary_types_allowed": True}

    seq: int = 0
    timestamp: float = 0.0
    samples: np.ndarray = Field(default_factory=lambda: np.array([], dtype=np.int16))
    meta: AudioFrameMeta = Field(default_factory=AudioFrameMeta)


class AudioCaptureConfig(ConfigType):
    """Format consensus — consumers read this to know stream parameters."""

    sample_rate: int = 44100
    channels: int = 1
    format: str = "pcm_s16le"
    frame_duration_ms: int = 50
    device_pattern: str = "blackhole"

    @classmethod
    def conf_name(cls) -> str:
        return "audio_capture"


class AudioCaptureSource(ABC):
    """Singleton capture source. Owns the microphone, fans PCM out to in-process consumers."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """产出音频的采样率 — 消费者据此对齐/重采样."""

    @property
    @abstractmethod
    def channels(self) -> int:
        """产出音频的通道数."""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    def device_explain(self) -> str: ...

    @abstractmethod
    def new_consumer(self, ring_buffer_frames: int = 64) -> "AudioPullLatest":
        """pull 最近的音频数据, 主动拉. """
        ...

    @abstractmethod
    def new_sequential_consumer(self, max_queue_frames: int = 128) -> "AudioSequentialConsumer":
        """"""
        ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def is_running(self) -> bool:
        """Whether the capture device is currently running (start() called and not yet closed)."""

    async def __aenter__(self):
        # 启动广播逻辑, 所有生产出来的消费者都会拿到音频, 直到其运行结束.
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class AudioPullLatest(ABC):
    """Non-blocking latest-frame consumer. For waveform display, AI perception."""

    @abstractmethod
    def pull_latest(self) -> AudioChunk | None: ...

    @abstractmethod
    def close(self) -> None: ...


class AudioSequentialConsumer(ABC):
    """Ordered consumer over a bounded queue. For ASR, audio recording.

    The producer (capture thread) never blocks: when the queue is full the
    newest frame is dropped. Real-time audio cannot backpressure the device.
    """

    @abstractmethod
    def shutdown(self, immediately: bool = False) -> None:
        """shutdown consumer, aiter 会主动结束.  """
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        """正式启动. 不启动时, 无法拉到数据."""
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """正式退出, 不再能拉到数据. """
        ...

    def __aiter__(self) -> "AsyncIterator[AudioChunk]":
        """循环拉取监听的音频片段. 不可重入."""
        return self

    @abstractmethod
    async def __anext__(self) -> AudioChunk: ...
