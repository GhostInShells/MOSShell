"""HostListener 故障上报契约 — 听觉循环异常退出 / ASR 上报.

锚定: ①pump 崩溃不让异常逃逸, 转成一条可被 ``on_error`` 观察的故障, 会话判死
(``is_running()`` 说真话); ②ASR 自己的 ``on_error`` 扇到同一观察面.
"""
import asyncio
from typing import AsyncIterable

import numpy as np
import pytest

from ghoshell_moss.contracts.asr import ASR, ASRInfo, RecognitionEvent, RecognitionStream
from ghoshell_moss.contracts.audio import (
    AudioCaptureSource,
    AudioChunk,
    AudioPullLatest,
    AudioSequentialConsumer,
)
from ghoshell_moss.host.listener.listener import HostListener


class _FakeConsumer(AudioSequentialConsumer):
    def shutdown(self, immediately: bool = False) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def __anext__(self) -> AudioChunk:
        raise StopAsyncIteration


class _FakeCapture(AudioCaptureSource):
    def __init__(self):
        self._running = False

    @property
    def sample_rate(self) -> int:
        return 16000

    @property
    def channels(self) -> int:
        return 1

    async def start(self) -> None:
        self._running = True

    def device_explain(self) -> str:
        return "fake capture"

    def new_consumer(self, ring_buffer_frames: int = 64) -> AudioPullLatest:
        raise NotImplementedError

    def new_sequential_consumer(
            self, max_queue_frames: int = 128, target_sample_rate: int | None = None,
    ) -> AudioSequentialConsumer:
        return _FakeConsumer()

    def set_aec(self, aec) -> None:
        pass

    async def close(self) -> None:
        self._running = False

    def is_running(self) -> bool:
        return self._running


class _RaisingStream(RecognitionStream):
    """识别流: 首个 __anext__ 即抛 (模拟 ASR 致命终止)."""

    def __init__(self, exc: Exception):
        self._exc = exc

    @property
    def stream_id(self) -> str:
        return "fake"

    def on_segment(self, callback) -> None:
        pass

    def on_event_creating(self, callback) -> None:
        pass

    def commit(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def is_input_done(self) -> bool:
        return False

    def __aiter__(self):
        return self

    async def __anext__(self) -> RecognitionEvent:
        raise self._exc


class _FaultASR(ASR):
    """ASR fake: recognize 产一条立即抛错的流; 可控地触发 on_error."""

    def __init__(self, exc: Exception):
        self._exc = exc
        self._error_cb = None

    def get_info(self) -> ASRInfo:
        return ASRInfo(sample_rate=16000)

    def configure(self, params: dict) -> None:
        pass

    def on_error(self, callback) -> None:
        self._error_cb = callback

    def report(self, error: Exception) -> None:
        """测试钩子: 模拟 ASR 上报一条长程故障."""
        if self._error_cb is not None:
            self._error_cb(error)

    def recognize(
            self, audio_chunks: AsyncIterable[AudioChunk], *,
            stream_id: str | None = None, gate_factory=None,
    ) -> RecognitionStream:
        return _RaisingStream(self._exc)

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_pump_fault_reports_and_marks_state_dead():
    """听觉循环崩溃: on_error 观察到故障 + is_running() 转 False + 退出不重抛."""
    faults: list[Exception] = []
    listener = HostListener(capture=_FakeCapture(), asr=_FaultASR(RuntimeError("asr fatal")))
    listener.on_error(faults.append)

    await listener.__aenter__()
    state = await listener.listen()
    async with state:
        await asyncio.sleep(0.05)  # 让 pump 跑到崩溃

    assert len(faults) == 1
    assert isinstance(faults[0], RuntimeError)
    assert state.is_running() is False  # 会话判死, 不再撒谎 "还在听"
    await listener.close()


@pytest.mark.asyncio
async def test_asr_error_fans_out_to_listener_observers():
    """ASR 自报故障 (on_error) 扇到同一耳朵级观察面 (此前零消费者)."""
    faults: list[Exception] = []
    asr = _FaultASR(RuntimeError("unused"))
    listener = HostListener(capture=_FakeCapture(), asr=asr)
    listener.on_error(faults.append)

    asr.report(RuntimeError("connect drop"))

    assert len(faults) == 1
    assert "connect drop" in str(faults[0])
