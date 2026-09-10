"""Host Listener — 缝合 audio capture + ASR 的"耳朵"器官实现.

对称 core/speech/stream_tts_speech.py 的 BaseTTSSpeech:
- ``HostListener`` 持有 capture (设备) + asr (注入), 拥有两者的生命周期.
- ``HostListenerState`` 是一条 listening session, ``async with state`` 启动后台
  pump (consumer → resample → ASR recognition → 三个观察面 fan-out), ``__aexit__``
  优雅收尾 (shutdown consumer → 等最后尾包 → 退订).
"""
import asyncio
import contextlib
import logging
from typing import AsyncIterable, Callable, Optional

import numpy as np
from ghoshell_common.contracts import LoggerItf
from typing_extensions import Self

from ghoshell_moss.contracts.asr import (
    ASR,
    RecognitionResult,
    RecognitionSegment,
    RecognitionStream,
)
from ghoshell_moss.contracts.audio import (
    AudioCaptureSource,
    AudioChunk,
    AudioSequentialConsumer,
    resample,
)
from ghoshell_moss.contracts.listener import Discard, Listener, ListenerState

__all__ = ["HostListener", "HostListenerState"]


def _make_discard(observers: list, callback) -> Discard:
    def _discard() -> None:
        with contextlib.suppress(ValueError):
            observers.remove(callback)
    return _discard


class HostListener(Listener):
    """缝合 capture + asr 的耳朵器官. 同一时刻至多一条 session (再次 listen 取消前一条)."""

    def __init__(
        self,
        *,
        capture: AudioCaptureSource,
        asr: ASR,
        logger: Optional[LoggerItf] = None,
    ):
        self._capture = capture
        self._asr = asr
        self._logger = logger or logging.getLogger("moss")
        self._log_prefix = "[HostListener]"
        self._state: Optional[HostListenerState] = None
        self._started = False
        self._closed = False
        # Listener 级观察者: 订阅即挂当前 session, 并留档给未来 session (listen 时自动装线).
        self._audio_observers: list[Callable[[AudioChunk], None]] = []
        self._result_observers: list[Callable[[RecognitionResult], None]] = []
        self._segment_observers: list[Callable[[RecognitionSegment], None]] = []

    @property
    def state(self) -> ListenerState | None:
        return self._state

    async def listen(self) -> ListenerState:
        if self._closed:
            raise RuntimeError("listener is closed")
        # 取消前一条 session (优雅: shutdown → 尾包 → 退出).
        if self._state is not None:
            await self._state.__aexit__(None, None, None)

        state = HostListenerState(
            capture=self._capture,
            asr=self._asr,
            logger=self._logger,
        )
        for cb in self._audio_observers:
            state.on_audio_chunk(cb)
        for cb in self._result_observers:
            state.on_recognition_result(cb)
        for cb in self._segment_observers:
            state.on_recognition_segment(cb)
        self._state = state
        return state

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._state is not None:
            await self._state.__aexit__(None, None, None)
            self._state = None
        await self._asr.close()
        await self._capture.close()
        self._logger.info("%s closed", self._log_prefix)

    def is_listening(self) -> bool:
        return self._state is not None and self._state.is_running()

    # ── Listener 级观察者 (自动装线到当前 session) ──

    def on_audio_chunk(self, callback: Callable[[AudioChunk], None]) -> Discard:
        self._audio_observers.append(callback)
        if self._state is not None:
            self._state.on_audio_chunk(callback)
        return _make_discard(self._audio_observers, callback)

    def on_recognition_result(self, callback: Callable[[RecognitionResult], None]) -> Discard:
        self._result_observers.append(callback)
        if self._state is not None:
            self._state.on_recognition_result(callback)
        return _make_discard(self._result_observers, callback)

    def on_recognition_segment(self, callback: Callable[[RecognitionSegment], None]) -> Discard:
        self._segment_observers.append(callback)
        if self._state is not None:
            self._state.on_recognition_segment(callback)
        return _make_discard(self._segment_observers, callback)

    async def __aenter__(self) -> Self:
        if not self._started:
            self._started = True
            await self._capture.start()
            self._logger.info("%s capture started", self._log_prefix)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class HostListenerState(ListenerState):
    """一条 listening session 的独立生命周期. 可重入: __aenter__/__aexit__ 幂等."""

    def __init__(
        self,
        *,
        capture: AudioCaptureSource,
        asr: ASR,
        logger: LoggerItf,
    ):
        self._capture = capture
        self._asr = asr
        self._logger = logger
        self._log_prefix = "[HostListenerState]"

        # 采样率桥接: capture 产出率 vs asr 期望率.
        self._capture_rate = capture.sample_rate
        self._asr_rate = asr.get_info().sample_rate

        self._consumer: Optional[AudioSequentialConsumer] = None
        self._recognition: Optional[RecognitionStream] = None
        self._pump_task: Optional[asyncio.Task] = None
        self._started = False
        self._closed = False
        self._running = False

        self._audio_observers: list[Callable[[AudioChunk], None]] = []
        self._result_observers: list[Callable[[RecognitionResult], None]] = []
        self._segment_observers: list[Callable[[RecognitionSegment], None]] = []

    # ── ListenerState contract ──

    async def __aenter__(self) -> Self:
        if self._started:
            return self
        self._started = True
        self._running = True

        self._consumer = self._capture.new_sequential_consumer()
        await self._consumer.__aenter__()
        self._recognition = self._asr.recognize(self._audio_gen())
        self._recognition.on_segment(self._dispatch_segment)
        self._pump_task = asyncio.create_task(self._pump())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._closed:
            return
        self._closed = True

        # 停喂音频 → audio_gen 结束 → ASR 发最后一次负序号 → 尾包 → pump 自然结束.
        if self._consumer is not None:
            self._consumer.shutdown()
        if self._pump_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._pump_task
            self._pump_task = None
        if self._consumer is not None:
            await self._consumer.__aexit__(None, None, None)
            self._consumer = None

        self._recognition = None
        self._running = False

    def commit(self) -> None:
        if self._recognition is not None:
            self._recognition.commit()

    def is_running(self) -> bool:
        return self._running

    # ── per-session 观察面 ──

    def on_audio_chunk(self, callback: Callable[[AudioChunk], None]) -> Discard:
        self._audio_observers.append(callback)
        return _make_discard(self._audio_observers, callback)

    def on_recognition_result(self, callback: Callable[[RecognitionResult], None]) -> Discard:
        self._result_observers.append(callback)
        return _make_discard(self._result_observers, callback)

    def on_recognition_segment(self, callback: Callable[[RecognitionSegment], None]) -> Discard:
        self._segment_observers.append(callback)
        return _make_discard(self._segment_observers, callback)

    # ── internals ──

    async def _audio_gen(self) -> AsyncIterable[np.ndarray]:
        """consumer (AudioChunk) → resample → np.ndarray 的桥."""
        async for chunk in self._consumer:
            self._dispatch_audio(chunk)
            samples = np.asarray(chunk.samples).ravel().astype(np.int16)
            if samples.size == 0:
                continue
            if self._capture_rate != self._asr_rate:
                samples = resample(samples, origin_rate=self._capture_rate, target_rate=self._asr_rate)
            yield samples

    async def _pump(self) -> None:
        async for result in self._recognition:
            self._dispatch_result(result)

    def _dispatch_audio(self, chunk: AudioChunk) -> None:
        for cb in list(self._audio_observers):
            try:
                cb(chunk)
            except Exception:
                self._logger.exception("%s on_audio_chunk callback failed", self._log_prefix)

    def _dispatch_result(self, result: RecognitionResult) -> None:
        for cb in list(self._result_observers):
            try:
                cb(result)
            except Exception:
                self._logger.exception("%s on_recognition_result callback failed", self._log_prefix)

    def _dispatch_segment(self, segment: RecognitionSegment) -> None:
        for cb in list(self._segment_observers):
            try:
                cb(segment)
            except Exception:
                self._logger.exception("%s on_recognition_segment callback failed", self._log_prefix)
