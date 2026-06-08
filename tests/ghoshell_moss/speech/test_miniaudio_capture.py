"""Contract tests for AudioCaptureSource and its consumers.

Tests are written against the abstract interfaces and run against
MiniAudioCaptureSource with a mocked AudioTransport — no Zenoh or hardware needed.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from ghoshell_moss.contracts.audio import (
    AudioCaptureConfig,
    AudioCaptureSource,
    AudioChunk,
    AudioFrameMeta,
    AudioPullLatest,
    AudioSequentialConsumer,
    AudioTransport,
)
from ghoshell_moss.host.speech.capture.miniaudio_capture import (
    MiniAudioCaptureSource,
    pack_chunk,
    unpack_chunk,
)


# -- helpers --

def _make_transport() -> AudioTransport:
    t = MagicMock(spec=AudioTransport)
    t.logger = MagicMock()
    t.acquire_lock.return_value = True
    t.sub_pcm_callback.return_value = lambda: None
    return t


def _make_source(**config_kwargs) -> MiniAudioCaptureSource:
    return MiniAudioCaptureSource(
        transport=_make_transport(),
        config=AudioCaptureConfig(**config_kwargs),
    )


# ── AudioCaptureSource contract ──────────────────────────────────────


class TestAudioCaptureSource:
    """Contract: any AudioCaptureSource."""

    def test_device_explain_before_start(self):
        """start 前 device_explain 返回有意义字符串，不含 running 语义."""
        source: AudioCaptureSource = _make_source()
        explain = source.device_explain()
        assert isinstance(explain, str)
        assert len(explain) > 0

    @pytest.mark.asyncio
    async def test_double_close_is_idempotent(self):
        """重复 close 不抛异常."""
        source: AudioCaptureSource = _make_source()
        await source.close()
        await source.close()

    def test_new_consumer_returns_pull_latest(self):
        """waveform / AI 感知场景：获取非阻塞消费者."""
        source: AudioCaptureSource = _make_source()
        consumer = source.new_consumer(ring_buffer_frames=64)
        assert isinstance(consumer, AudioPullLatest)
        consumer.close()

    def test_new_sequential_consumer_returns_correct_type(self):
        """ASR / 录音场景：获取顺序消费者."""
        source: AudioCaptureSource = _make_source()
        consumer = source.new_sequential_consumer(max_queue_frames=128)
        assert isinstance(consumer, AudioSequentialConsumer)


# ── AudioPullLatest contract ─────────────────────────────────────────


class TestAudioPullLatest:
    """Contract: any AudioPullLatest."""

    def test_pull_latest_non_blocking_returns_none_or_chunk(self):
        """无数据时 pull_latest 非阻塞返回 None."""
        source = _make_source()
        consumer: AudioPullLatest = source.new_consumer(ring_buffer_frames=32)
        result = consumer.pull_latest()
        assert result is None or isinstance(result, AudioChunk)
        consumer.close()

    def test_close_idempotent(self):
        """close 可重复调用."""
        source = _make_source()
        consumer: AudioPullLatest = source.new_consumer(ring_buffer_frames=32)
        consumer.close()
        consumer.close()


# ── AudioSequentialConsumer contract ─────────────────────────────────


class TestAudioSequentialConsumer:
    """Contract: any AudioSequentialConsumer."""

    def test_iteration_without_start_raises(self):
        """未 start 就迭代应抛出 RuntimeError."""
        source = _make_source()
        consumer: AudioSequentialConsumer = source.new_sequential_consumer(max_queue_frames=32)
        with pytest.raises(RuntimeError):
            consumer.__aiter__()

    @pytest.mark.asyncio
    async def test_close_before_start_is_safe(self):
        """未 start 就 close 不抛异常."""
        source = _make_source()
        consumer: AudioSequentialConsumer = source.new_sequential_consumer(max_queue_frames=32)
        await consumer.close()


# ── Serialization contract ───────────────────────────────────────────


class TestAudioChunkSerialization:
    """AudioChunk 跨进程链路：capture 端 pack，consumer 端 unpack。"""

    def test_roundtrip_preserves_all_fields(self):
        """pack → unpack 后所有字段不变。这是跨进程传输的契约."""
        rng = np.random.RandomState(42)
        samples = (rng.randn(2205) * 8000).astype(np.int16)

        chunk = AudioChunk(
            seq=42,
            timestamp=1234567890.123,
            samples=samples.copy(),
            meta=AudioFrameMeta(
                rms_db=-12.3,
                bands={"bass": -20.1, "mid": -12.3, "high": -8.7},
                is_silent=False,
            ),
        )

        packed = pack_chunk(chunk)
        unpacked = unpack_chunk(packed)

        assert unpacked.seq == 42
        assert unpacked.timestamp == pytest.approx(1234567890.123)
        assert unpacked.meta.rms_db == -12.3
        assert unpacked.meta.bands == {"bass": -20.1, "mid": -12.3, "high": -8.7}
        assert unpacked.meta.is_silent is False
        assert np.array_equal(unpacked.samples, samples)

    def test_silent_frame_roundtrip(self):
        """静音帧的 is_silent 标志在往返中正确保持."""
        samples = np.zeros(100, dtype=np.int16)
        chunk = AudioChunk(
            seq=0, timestamp=0.0, samples=samples,
            meta=AudioFrameMeta(rms_db=-96, bands={"bass": -96, "mid": -96, "high": -96}, is_silent=True),
        )
        unpacked = unpack_chunk(pack_chunk(chunk))
        assert unpacked.meta.is_silent is True
        assert unpacked.meta.rms_db == -96
        assert np.array_equal(unpacked.samples, samples)
