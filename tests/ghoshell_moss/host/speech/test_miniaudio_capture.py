"""Contract tests for AudioCaptureSource and its consumers.

Tests run against MiniAudioCaptureSource with a mocked Workspace — no hardware.
Capture fans PCM out locally (no Zenoh), so the contract under test is:
register an observer via ``on_audio_chunk``, dispose it, and consumers read
chunks from an in-process queue.
"""
from unittest.mock import MagicMock

import numpy as np
import pytest

from ghoshell_moss.contracts.audio import (
    AudioCaptureConfig,
    AudioCaptureSource,
    AudioChunk,
    AudioPullLatest,
    AudioSequentialConsumer,
)
from ghoshell_moss.contracts.workspace import Workspace
from ghoshell_moss.host.listener.capture.miniaudio_capture import MiniAudioCaptureSource


# -- helpers --

def _make_workspace() -> Workspace:
    ws = MagicMock(spec=Workspace)
    return ws


def _make_source(**config_kwargs) -> MiniAudioCaptureSource:
    return MiniAudioCaptureSource(
        workspace=_make_workspace(),
        config=AudioCaptureConfig(**config_kwargs),
    )


def _make_chunk(seq: int = 1) -> AudioChunk:
    return AudioChunk(
        seq=seq,
        timestamp=float(seq),
        samples=np.zeros(16, dtype=np.int16),
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


# ── local fan-out contract ───────────────────────────────────────────


class TestLocalFanOut:
    """capture 本地扇出: 注册 observer → 分发 → disposer 摘除."""

    def test_observer_receives_fanned_out_chunk(self):
        source = _make_source()
        got: list[AudioChunk] = []
        source.on_audio_chunk(got.append)

        chunk = _make_chunk(seq=7)
        source._fan_out(chunk)

        assert len(got) == 1
        assert got[0].seq == 7

    def test_dispose_stops_future_dispatch(self):
        source = _make_source()
        got: list[AudioChunk] = []
        dispose = source.on_audio_chunk(got.append)

        source._fan_out(_make_chunk(1))
        dispose()
        source._fan_out(_make_chunk(2))

        assert [c.seq for c in got] == [1]

    def test_dispose_is_idempotent(self):
        source = _make_source()
        dispose = source.on_audio_chunk(lambda c: None)
        dispose()
        dispose()  # 不抛


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

    def test_pull_latest_sees_fanned_out_chunk(self):
        """扇出的帧能被 ring-buffer 消费者读到."""
        source = _make_source()
        consumer: AudioPullLatest = source.new_consumer(ring_buffer_frames=32)
        source._fan_out(_make_chunk(seq=3))
        latest = consumer.pull_latest()
        assert latest is not None and latest.seq == 3
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
    async def test_aexit_before_aenter_is_safe(self):
        """未 enter 就 exit 不抛异常."""
        source = _make_source()
        consumer: AudioSequentialConsumer = source.new_sequential_consumer(max_queue_frames=32)
        await consumer.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_receives_fanned_out_chunk_in_order(self):
        """扇出的帧按序被顺序消费者读到."""
        source = _make_source()
        consumer: AudioSequentialConsumer = source.new_sequential_consumer(max_queue_frames=32)
        await consumer.__aenter__()

        source._fan_out(_make_chunk(seq=1))
        source._fan_out(_make_chunk(seq=2))

        first = await consumer.__anext__()
        second = await consumer.__anext__()
        assert first.seq == 1
        assert second.seq == 2

        await consumer.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_shutdown_stops_iteration(self):
        """shutdown 后迭代立即结束."""
        source = _make_source()
        consumer: AudioSequentialConsumer = source.new_sequential_consumer(max_queue_frames=32)
        await consumer.__aenter__()

        consumer.shutdown()
        with pytest.raises(StopAsyncIteration):
            await consumer.__anext__()

        await consumer.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_resamples_when_target_rate_differs(self):
        """声明 target_sample_rate != capture 率时, 消费侧重采样."""
        source = _make_source()  # capture sample_rate = 16000
        consumer = source.new_sequential_consumer(max_queue_frames=32, target_sample_rate=8000)
        await consumer.__aenter__()

        source._fan_out(AudioChunk(seq=1, timestamp=1.0, samples=np.zeros(160, dtype=np.int16)))

        chunk = await consumer.__anext__()
        assert chunk.samples.size == 80  # 160 * 8000/16000

        await consumer.__aexit__(None, None, None)

    @pytest.mark.asyncio
    async def test_passthrough_when_target_rate_matches(self):
        """target_sample_rate == capture 率时不重采样 (原样透传)."""
        source = _make_source()  # capture sample_rate = 16000
        consumer = source.new_sequential_consumer(max_queue_frames=32, target_sample_rate=16000)
        await consumer.__aenter__()

        source._fan_out(AudioChunk(seq=1, timestamp=1.0, samples=np.zeros(160, dtype=np.int16)))

        chunk = await consumer.__anext__()
        assert chunk.samples.size == 160

        await consumer.__aexit__(None, None, None)
