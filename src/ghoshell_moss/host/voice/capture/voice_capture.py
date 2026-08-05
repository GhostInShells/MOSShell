"""Direct miniaudio capture — node-owned boundary, 16kHz, no transport layer."""
from __future__ import annotations

import asyncio
import logging

import numpy as np

from ghoshell_moss.depends import depend_host

depend_host()
import miniaudio

from ghoshell_moss.host.voice.contracts import DeviceConfig

__all__ = ["VoiceCapture"]


class VoiceCapture:
    """Owns the microphone via miniaudio. Publishes raw int16 PCM to an asyncio.Queue."""

    def __init__(
        self,
        device: DeviceConfig,
        sample_rate: int = 16000,
        frame_duration_ms: int = 40,
        *,
        logger: logging.Logger | None = None,
    ):
        self._device = device
        self._sample_rate = sample_rate
        self._frame_duration_ms = frame_duration_ms
        self._log = logger or logging.getLogger("moss.voice.capture")
        self._capture: miniaudio.CaptureDevice | None = None
        self._queue: asyncio.Queue[np.ndarray] | None = None
        self._running = False

    def _find_device(self) -> int | None:
        """Three-tier device selection: pattern → index → default (None)."""
        if self._device.device_pattern:
            pattern = self._device.device_pattern.lower()
            try:
                for d in miniaudio.Devices().capture:
                    if pattern in d.name.lower():
                        self._log.info("Voice capture device: %s (pattern=%s)", d.name, pattern)
                        return d.id
            except Exception:
                self._log.warning("Device enumeration failed, pattern=%s", pattern)
        if self._device.device_index is not None:
            self._log.info("Voice capture device: index=%s", self._device.device_index)
            return self._device.device_index
        return None

    def device_info(self) -> str:
        if self._capture is None:
            return "not started"
        return f"miniaudio {self._sample_rate}Hz {self._capture.name or 'default'}"

    def start(self, queue: asyncio.Queue[np.ndarray]) -> None:
        """Open the microphone. PCM chunks are put into *queue* (thread-safe)."""
        if self._running:
            return
        self._queue = queue
        device_id = self._find_device()
        self._capture = miniaudio.CaptureDevice(
            input_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=1,
            sample_rate=self._sample_rate,
            buffersize_msec=self._frame_duration_ms,
            device_id=device_id,
        )
        gen = self._generator()
        next(gen)
        self._capture.start(gen)
        self._running = True
        self._log.info("Voice capture started: %s", self.device_info())

    def stop(self) -> None:
        if not self._running:
            return
        if self._capture is not None:
            self._capture.stop()
            self._capture.close()
            self._capture = None
        self._running = False
        self._log.info("Voice capture stopped")

    @property
    def running(self) -> bool:
        return self._running

    def _generator(self):
        loop = asyncio.get_event_loop()
        queue = self._queue

        def _feed():
            while True:
                raw = yield
                try:
                    samples = np.frombuffer(raw, dtype=np.int16).copy()
                    loop.call_soon_threadsafe(queue.put_nowait, samples)
                except asyncio.QueueFull:
                    pass
                except Exception:
                    pass

        return _feed()
