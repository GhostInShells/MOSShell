"""
Audio capture contracts — shared abstractions for system audio input.

Capture source → raw PCM → transport → consumers (ASR, waveform, AI perception).
"""
from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, Field
from typing import AsyncIterator
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
    """Singleton capture source. Owns the microphone, publishes PCM to Zenoh."""

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
    """Ordered, lossless consumer with backpressure. For ASR, audio recording."""

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
