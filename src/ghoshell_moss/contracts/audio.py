"""
Audio capture contracts — shared abstractions for system audio input.

Capture source → raw PCM → transport → consumers (ASR, waveform, AI perception).
"""
from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel, Field

from ghoshell_moss.contracts.configs import ConfigType

__all__ = [
    "AudioFrameMeta",
    "AudioChunk",
    "AudioCaptureConfig",
    "AudioCaptureSource",
    "AudioPullLatest",
    "AudioSequentialConsumer",
]


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

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    def device_explain(self) -> str: ...

    @abstractmethod
    def new_consumer(self, ring_buffer_frames: int = 64) -> "AudioPullLatest": ...

    @abstractmethod
    def new_sequential_consumer(self, max_queue_frames: int = 128) -> "AudioSequentialConsumer": ...

    @abstractmethod
    async def close(self) -> None: ...


class AudioPullLatest(ABC):
    """Non-blocking latest-frame consumer. For waveform display, AI perception."""

    @abstractmethod
    def pull_latest(self) -> AudioChunk | None: ...

    @abstractmethod
    def close(self) -> None: ...


class AudioSequentialConsumer(ABC):
    """Ordered, lossless consumer with backpressure. For ASR, audio recording."""

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    def __aiter__(self) -> "AudioSequentialConsumer": ...

    @abstractmethod
    async def __anext__(self) -> AudioChunk: ...
