"""
MiniAudio-based audio capture — system audio → raw PCM, fanned out locally.

Capture owns the microphone device and fans each PCM frame out to in-process
consumers (callbacks + a bounded queue). It does not publish PCM to Zenoh:
audio stays local, so single-process listen+speak (and AEC) never round-trips
through the session bus, and capture's lifecycle is no longer tied to the Matrix
session.
"""
import collections
import contextlib
import logging
import re
import time
from typing import Callable

from ghoshell_moss.depends import depend_host

depend_host()
import janus
import miniaudio
import numpy as np

from ghoshell_moss.contracts.audio import (
    AudioCaptureConfig,
    AudioCaptureSource,
    AudioChunk,
    AudioFrameMeta,
    AudioPullLatest,
    AudioSequentialConsumer,
)
from ghoshell_moss.contracts.workspace import Workspace
from ghoshell_common.contracts import LoggerItf

__all__ = [
    "MiniAudioCaptureSource",
    "MiniAudioSequentialConsumer",
]

_SILENCE_THRESHOLD_DB = -50.0

#: 设备锁 key 前缀 —— 锁的是具体设备, 不是全局 "audio_capture".
_LOCK_KEY_PREFIX = "audio_capture"


def _compute_frame_meta(samples: np.ndarray) -> AudioFrameMeta:
    """Compute RMS + 3-band energy + silence flag from raw PCM."""
    f32 = samples.astype(np.float64) / 32768.0
    rms = float(np.sqrt(np.mean(f32**2)))
    rms_db = 20.0 * np.log10(max(rms, 1e-10))

    fft = np.abs(np.fft.rfft(f32))
    n = len(fft)
    if n >= 6:
        bass = 20.0 * np.log10(max(float(np.mean(fft[:n // 6])), 1e-10))
        mid = 20.0 * np.log10(max(float(np.mean(fft[n // 6:2 * n // 3])), 1e-10))
        high = 20.0 * np.log10(max(float(np.mean(fft[2 * n // 3:])), 1e-10))
    else:
        bass = mid = high = rms_db

    return AudioFrameMeta(
        rms_db=round(rms_db, 1),
        bands={"bass": round(bass, 1), "mid": round(mid, 1), "high": round(high, 1)},
        is_silent=rms_db < _SILENCE_THRESHOLD_DB,
    )


def _device_lock_key(device_id) -> str:
    """锁 key 由设备标识派生, 符合 workspace.lock 的 `^[a-zA-Z0-9_-]+$` 约束."""
    if device_id is None:
        return f"{_LOCK_KEY_PREFIX}_default"
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", str(device_id))
    return f"{_LOCK_KEY_PREFIX}_{s}" or f"{_LOCK_KEY_PREFIX}_default"


class MiniAudioCaptureSource(AudioCaptureSource):
    """Capture system audio via miniaudio, fan out locally to in-process consumers."""

    def __init__(
            self,
            *,
            config: AudioCaptureConfig,
            workspace: Workspace,
            logger: LoggerItf | None = None,
    ):
        self._config = config
        self._workspace = workspace
        self._logger = logger or logging.getLogger("moss")
        self._capture: miniaudio.CaptureDevice | None = None
        self._locker = None
        self._observers: list[Callable[[AudioChunk], None]] = []
        self._seq = 0
        self._started = False
        self._closing = False

    @property
    def sample_rate(self) -> int:
        return self._config.sample_rate

    @property
    def channels(self) -> int:
        return self._config.channels

    # -- lifecycle --

    def on_audio_chunk(self, callback: Callable[[AudioChunk], None]) -> Callable[[], None]:
        """注册一个音频帧观察者, 返回 disposer (调用即摘除)."""
        self._observers.append(callback)

        def _dispose() -> None:
            with contextlib.suppress(ValueError):
                self._observers.remove(callback)

        return _dispose

    async def start(self) -> None:
        if self._started:
            return

        device_id = self._find_device()
        lock_key = _device_lock_key(device_id)
        self._locker = self._workspace.lock(lock_key)
        if not self._locker.acquire(timeout=0):
            self._logger.warning(
                "Audio capture device locked by another process (%s), skipping start", lock_key)
            self._locker = None
            self._started = True
            return

        if device_id is not None:
            self._logger.info("Audio capture using device id=%s", device_id)
        else:
            self._logger.info("Audio capture using default input device")

        self._capture = miniaudio.CaptureDevice(
            input_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=self._config.channels,
            sample_rate=self._config.sample_rate,
            buffersize_msec=self._config.frame_duration_ms,
            device_id=device_id,
        )

        gen = self._make_capture_generator()
        next(gen)
        self._capture.start(gen)

        self._started = True
        self._logger.info("Audio capture started (device=%s)", self.device_explain())

    def device_explain(self) -> str:
        if self._capture is None:
            return "not started"
        return f"miniaudio capture, {self._config.sample_rate}Hz, " \
               f"{self._config.channels}ch, {self._config.format}"

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True

        if self._capture is not None:
            self._capture.stop()
            self._capture.close()
            self._capture = None

        if self._locker is not None:
            self._locker.release()
            self._locker = None

        self._started = False
        self._logger.info("Audio capture closed")

    def is_running(self) -> bool:
        return self._started

    # -- consumer factories --

    def new_consumer(self, ring_buffer_frames: int = 64) -> AudioPullLatest:
        return _MiniAudioPullLatest(
            capture=self,
            maxlen=ring_buffer_frames,
            logger=self._logger,
        )

    def new_sequential_consumer(self, max_queue_frames: int = 128) -> AudioSequentialConsumer:
        return MiniAudioSequentialConsumer(
            capture=self,
            maxsize=max_queue_frames,
            logger=self._logger,
        )

    # -- internals --

    def _find_device(self):
        pattern = self._config.device_pattern.lower()
        try:
            for d in miniaudio.Devices().get_captures():
                if pattern in d['name'].lower():
                    return d['id']
        except Exception as e:
            self._logger.warning("Device enumeration failed: %s, using default", e)
        return None

    def _fan_out(self, chunk: AudioChunk) -> None:
        """public-internal: 把一帧分发给所有注册的观察者.

        采集 generator 与测试共用. 观察者 (consumer 的入队回调) 必须非阻塞 —
        这里的调用发生在 miniaudio 采集线程, 阻塞它会拖垮设备 buffer.
        """
        for cb in list(self._observers):
            try:
                cb(chunk)
            except Exception:
                self._logger.exception("Error in audio capture observer")

    def _make_capture_generator(self):
        channels = self._config.channels
        logger = self._logger
        seq_ref = [0]
        fan_out = self._fan_out

        def _capture_generator():
            while True:
                data = yield
                try:
                    ts = time.time()
                    # miniaudio 会复用底层 buffer, 必须 copy, 否则下一帧覆盖本帧.
                    samples = np.frombuffer(data, dtype=np.int16).reshape(-1, channels).copy()
                    meta = _compute_frame_meta(samples)
                    chunk = AudioChunk(
                        seq=seq_ref[0], timestamp=ts, samples=samples, meta=meta,
                    )
                    seq_ref[0] += 1
                    fan_out(chunk)
                except Exception:
                    logger.exception("Error in capture callback")

        return _capture_generator()


class _MiniAudioPullLatest(AudioPullLatest):
    """Ring-buffer consumer. Non-blocking, latest frame wins."""

    def __init__(self, *, capture: MiniAudioCaptureSource, maxlen: int, logger):
        self._ring: collections.deque[AudioChunk] = collections.deque(maxlen=maxlen)
        self._logger = logger
        self._closed = False
        self._dispose = capture.on_audio_chunk(self._on_chunk)

    def _on_chunk(self, chunk: AudioChunk) -> None:
        self._ring.append(chunk)

    def pull_latest(self) -> AudioChunk | None:
        if self._closed or not self._ring:
            return None
        return self._ring[-1]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._dispose()


class MiniAudioSequentialConsumer(AudioSequentialConsumer):
    """Ordered queue consumer with backpressure. For ASR, audio recording.

    采集线程 (miniaudio 回调) 把帧 ``put_nowait`` 进有界 janus 队列; 队列满时丢弃
    最新帧 —— 音频是实时流, 采集线程不能阻塞 (否则设备 buffer 溢出), 宁可丢帧.
    ``__anext__`` 在 event loop 侧 ``async_q.get()``, 永不阻塞 loop.
    """

    def __init__(self, *, capture: MiniAudioCaptureSource, maxsize: int, logger):
        self._capture = capture
        self._maxsize = maxsize
        self._logger = logger
        self._queue: janus.Queue | None = None
        self._dispose: Callable[[], None] | None = None
        self._started = False
        self._shutdown = False

    def shutdown(self, immediately: bool = False) -> None:
        self._shutdown = True
        if self._queue is not None:
            with contextlib.suppress(Exception):
                self._queue.sync_q.put_nowait(None)

    async def __aenter__(self) -> "MiniAudioSequentialConsumer":
        if not self._started:
            self._queue = janus.Queue(maxsize=self._maxsize)
            self._dispose = self._capture.on_audio_chunk(self._on_chunk)
            self._started = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._dispose is not None:
            self._dispose()
            self._dispose = None
        self._started = False

    def _on_chunk(self, chunk: AudioChunk) -> None:
        if self._queue is not None and not self._shutdown:
            with contextlib.suppress(Exception):
                self._queue.sync_q.put_nowait(chunk)

    def __aiter__(self) -> "MiniAudioSequentialConsumer":
        if not self._started:
            raise RuntimeError("Consumer not started — enter async context first")
        return self

    async def __anext__(self) -> AudioChunk:
        if self._shutdown or self._queue is None:
            raise StopAsyncIteration
        item = await self._queue.async_q.get()
        if item is None or self._shutdown:
            raise StopAsyncIteration
        return item
