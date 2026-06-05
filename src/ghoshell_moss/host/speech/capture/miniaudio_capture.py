"""
MiniAudio-based audio capture — system audio → raw PCM → Zenoh stream.
"""
import asyncio
import collections
import json
import logging
import struct
import time
from typing import Callable

import miniaudio
import numpy as np

from ghoshell_moss.contracts.audio import (
    AudioCaptureConfig,
    AudioCaptureSource,
    AudioChunk,
    AudioFrameMeta,
    AudioPullLatest,
    AudioRuntimeInfo,
    AudioSequentialConsumer,
)
from ghoshell_moss.contracts.workspace import Lock
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.session import Sample

__all__ = [
    "MiniAudioCaptureSource",
    "MiniAudioSequentialConsumer",
    "unpack_chunk",
    "pack_chunk",
]

_STREAM_KEY = "audio/pcm"
_SILENCE_THRESHOLD_DB = -50.0


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


def pack_chunk(chunk: AudioChunk) -> bytes:
    """Serialize AudioChunk to bytes for Zenoh transport.

    Format: [4B meta_json_len(uint32 BE)] [meta_json(UTF-8)] [pcm(int16 LE)]
    """
    meta_json = json.dumps({
        "seq": chunk.seq,
        "timestamp": chunk.timestamp,
        "rms_db": chunk.meta.rms_db,
        "bands": chunk.meta.bands,
        "is_silent": chunk.meta.is_silent,
    }).encode("utf-8")
    header = struct.pack(">I", len(meta_json))
    pcm = chunk.samples.astype(np.int16).tobytes()
    return header + meta_json + pcm


def unpack_chunk(data: bytes) -> AudioChunk:
    """Deserialize bytes back to AudioChunk."""
    meta_len = struct.unpack(">I", data[:4])[0]
    meta_json = data[4:4 + meta_len].decode("utf-8")
    meta_dict = json.loads(meta_json)
    pcm = np.frombuffer(data[4 + meta_len:], dtype=np.int16)

    meta = AudioFrameMeta(
        rms_db=meta_dict["rms_db"],
        bands=meta_dict["bands"],
        is_silent=meta_dict["is_silent"],
    )
    return AudioChunk(
        seq=meta_dict["seq"],
        timestamp=meta_dict["timestamp"],
        samples=pcm,
        meta=meta,
    )


class MiniAudioCaptureSource(AudioCaptureSource):
    """Capture system audio via miniaudio CaptureDevice, publish PCM to Zenoh."""

    def __init__(self, *, matrix: Matrix, config: AudioCaptureConfig):
        self._matrix = matrix
        self._config = config
        self._logger = matrix.logger or logging.getLogger("moss.audio_capture")
        self._session = matrix.session
        self._capture: miniaudio.CaptureDevice | None = None
        self._locker: Lock | None = None
        self._seq = 0
        self._started = False
        self._closing = False

    # ── lifecycle ──────────────────────────────────────────────

    async def start(self) -> None:
        if self._started:
            return

        # Process lock — prevent duplicate device open across processes
        ws = self._matrix.workspace
        self._locker = ws.lock("audio_capture")
        if not self._locker.acquire(timeout=0):
            self._logger.warning("Audio capture lock held by another process, skipping start")
            self._started = True  # mark as started so lifecycle can proceed
            return

        # Device discovery
        device_id = self._find_device()
        if device_id is not None:
            self._logger.info("Audio capture using device id=%s", device_id)
        else:
            self._logger.info("Audio capture using default input device")

        # Build capture device
        self._capture = miniaudio.CaptureDevice(
            input_format=miniaudio.SampleFormat.SIGNED16,
            nchannels=self._config.channels,
            sample_rate=self._config.sample_rate,
            buffersize_msec=self._config.frame_duration_ms,
            device_id=device_id,
        )

        # Start capture via callback generator
        gen = self._make_capture_generator()
        next(gen)  # prime
        self._capture.start(gen)

        # Write runtime info to tmp_storage
        self._write_runtime_info()
        self._started = True
        self._logger.info("Audio capture started (key=%s, device=%s)",
                          _STREAM_KEY, self.device_explain())

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
        elif self._started:
            self._logger.info("Audio capture close (no-op, lock was held by another process)")

        self._started = False
        self._logger.info("Audio capture closed")

    # ── MatrixLifecycleObject ──────────────────────────────────

    async def __aenter__(self) -> "MiniAudioCaptureSource":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # ── consumer factories ─────────────────────────────────────

    def new_consumer(self, ring_buffer_frames: int = 64) -> AudioPullLatest:
        return _MiniAudioPullLatest(
            session=self._session,
            stream_key=_STREAM_KEY,
            maxlen=ring_buffer_frames,
            logger=self._logger,
        )

    def new_sequential_consumer(self, max_queue_frames: int = 128) -> AudioSequentialConsumer:
        return MiniAudioSequentialConsumer(
            session=self._session,
            stream_key=_STREAM_KEY,
            maxsize=max_queue_frames,
            logger=self._logger,
        )

    # ── internals ──────────────────────────────────────────────

    def _find_device(self):
        """Match a capture device by config.device_pattern."""
        pattern = self._config.device_pattern.lower()
        try:
            for d in miniaudio.Devices().capture:
                if pattern in d.name.lower():
                    return d.id
        except Exception as e:
            self._logger.warning("Device enumeration failed: %s, using default", e)
        return None

    def _make_capture_generator(self):
        """Build the miniaudio callback generator.

        miniaudio calls gen.send(raw_bytes) from its internal thread,
        where raw_bytes is SIGNED16 PCM matching our sample rate + channels.
        """
        channels = self._config.channels
        sample_rate = self._config.sample_rate
        frame_duration_ms = self._config.frame_duration_ms
        logger = self._logger
        session = self._session
        stream_key = _STREAM_KEY
        seq_ref = [0]  # mutable ref for closure

        def _capture_generator():
            while True:
                data = yield  # bytes from miniaudio
                try:
                    ts = time.time()
                    samples = np.frombuffer(data, dtype=np.int16).reshape(-1, channels)
                    meta = _compute_frame_meta(samples)
                    seq = seq_ref[0]
                    seq_ref[0] += 1

                    chunk = AudioChunk(seq=seq, timestamp=ts, samples=samples, meta=meta)
                    packed = pack_chunk(chunk)
                    session.pub_stream_delta(stream_key, packed)
                except Exception:
                    logger.exception("Error in capture callback")

        return _capture_generator()

    def _write_runtime_info(self) -> None:
        try:
            devices = miniaudio.Devices()
            device_name = "default"
            for d in devices.capture:
                if d.is_default:
                    device_name = d.name
                    break
        except Exception:
            device_name = "unknown"

        info = AudioRuntimeInfo(
            running=True,
            stream_key=_STREAM_KEY,
            device_name=device_name,
            device_explain=self.device_explain(),
            started_at=time.time(),
            last_heartbeat=time.time(),
        )
        try:
            storage = self._matrix.session.tmp_storage
            storage.put("audio_runtime_info.json", info.model_dump_json().encode("utf-8"))
        except Exception as e:
            self._logger.warning("Failed to write audio runtime info: %s", e)


class _MiniAudioPullLatest(AudioPullLatest):
    """Ring-buffer consumer. Non-blocking, latest frame wins."""

    def __init__(self, *, session, stream_key: str, maxlen: int, logger):
        self._ring: collections.deque[AudioChunk] = collections.deque(maxlen=maxlen)
        self._logger = logger
        self._release: Callable[[], None] | None = None
        self._closed = False

        self._release = session.sub_stream(stream_key, self._on_sample)

    def _on_sample(self, sample: Sample) -> None:
        try:
            chunk = unpack_chunk(sample.payload)
            self._ring.append(chunk)
        except Exception:
            pass  # tolerate deserialize failures

    def pull_latest(self) -> AudioChunk | None:
        if self._closed or not self._ring:
            return None
        return self._ring[-1]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._release is not None:
            self._release()
            self._release = None


class MiniAudioSequentialConsumer(AudioSequentialConsumer):
    """Ordered queue consumer with backpressure. For ASR, audio recording."""

    def __init__(self, *, session, stream_key: str, maxsize: int, logger):
        self._session = session
        self._stream_key = stream_key
        self._maxsize = maxsize
        self._logger = logger
        self._stream = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._stream = self._session.get_stream(self._stream_key, maxsize=self._maxsize)
        await self._stream.__aenter__()
        self._started = True

    async def close(self) -> None:
        if self._stream is not None:
            await self._stream.__aexit__(None, None, None)
            self._stream = None
        self._started = False

    def __aiter__(self) -> "MiniAudioSequentialConsumer":
        if not self._started:
            raise RuntimeError("Consumer not started — call start() first")
        return self

    async def __anext__(self) -> AudioChunk:
        sample = await self._stream.__anext__()  # raises StopAsyncIteration on sentinel
        return unpack_chunk(sample.payload)
