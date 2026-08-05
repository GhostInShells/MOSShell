"""VoiceController implementation — wires capture + ASR + state machine + event dispatch."""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Callable

import numpy as np

from ghoshell_moss.host.listener.capture import VoiceCapture
from ghoshell_moss.host.listener.contracts import (
    Disposer,
    EventHandler,
    VoiceConfig,
    VoiceController,
    VoiceMode,
    VoiceNodeRuntime,
)
from ghoshell_moss.host.listener.handlers import EventBus
from ghoshell_moss.host.listener.state import VoiceStateMachine

_ASR_SAMPLE_RATE = 16000

__all__ = ["VoiceControllerImpl"]


class VoiceControllerImpl(VoiceController):
    """Host-level implementation — owns capture, state machine, ASR, and event dispatch.

    Does NOT depend on matrix, channel, or transport.  Controller receives a *gate_check*
    callback (provided by the adapter) to gate capture during TTS playback.
    """

    def __init__(
        self,
        config: VoiceConfig,
        *,
        gate_check: Callable[[], bool] | None = None,
        logger: logging.Logger | None = None,
    ):
        self._log = logger or logging.getLogger("moss.voice.controller")
        self._bus = EventBus()
        self._state = VoiceStateMachine(config, self._bus, logger=self._log)
        self._capture = VoiceCapture(config.device, sample_rate=_ASR_SAMPLE_RATE, logger=self._log)
        self._config = config
        self._gate_check = gate_check or (lambda: False)
        self._queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=64)
        self._task: asyncio.Task | None = None
        self._started = False

    # ── VoiceController ──

    async def start(self) -> None:
        if self._started:
            return
        self._capture.start(self._queue)
        self._state.start()
        self._task = asyncio.create_task(self._run())
        self._started = True
        self._log.info("VoiceController started")

    async def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self._capture.stop()
        self._state.stop()
        self._log.info("VoiceController stopped")

    async def set_mode(self, mode: VoiceMode) -> None:
        self._state.set_mode(mode)

    async def set_config(self, config: VoiceConfig) -> None:
        self._state.set_config(config)
        self._config = config

    def add_handler(self, handler: EventHandler) -> Disposer:
        return self._bus.add(handler)

    def snapshot(self) -> VoiceNodeRuntime:
        return self._state.snapshot(
            capture_device=self._capture.device_info(),
            capture_rate=_ASR_SAMPLE_RATE,
        )

    # ── main loop ──

    async def _run(self) -> None:
        """Main recognition loop — capture → ASR → state machine → dispatch."""
        from ghoshell_moss.host.listener._asr_helpers import iter_with_silence_timeout
        from ghoshell_moss.host.listener.volcengine_asr import VolcengineASR, VolcengineASRConfig

        asr = VolcengineASR(
            config=VolcengineASRConfig(
                sample_rate=_ASR_SAMPLE_RATE,
                end_window_size=self._config.asr_end_window_ms,
            ),
            logger=self._log,
        )

        try:
            while self._started:
                # Wait for TTS gate to clear before arming
                while self._gate_check():
                    await asyncio.sleep(0.05)
                self._state.on_arm()

                # Drain stale audio from inter-utterance silence
                self._drain_queue()

                # New utterance — feed raw PCM to ASR, wrap results with silence timeout
                utterance_id = uuid.uuid4().hex[:12]
                audio_gen = self._queue_generator()
                async for result in iter_with_silence_timeout(asr.recognize(audio_gen), self._log):
                    if not result.text:
                        continue
                    self._log.info("ASR %s: %s", "final" if result.is_final else "partial", result.text)
                    if result.is_final:
                        self._state.on_asr_final(utterance_id, result.text)
                        break
                    else:
                        self._state.on_asr_partial(utterance_id, result.text)

        except asyncio.CancelledError:
            self._log.info("Controller loop cancelled")
        except Exception:
            self._log.exception("Controller loop error")
        finally:
            await asr.close()

    def _drain_queue(self) -> None:
        """Discard stale audio buffered between utterances."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _queue_generator(self):
        """Yield int16 samples from the capture queue."""
        while self._started:
            chunk = await self._queue.get()
            yield chunk.ravel().astype(np.int16)
