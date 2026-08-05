"""
MiniAudio-based audio capture — system audio → raw PCM → transport stream.
"""
import collections
import json
import struct
import time
from typing import Callable

from ghoshell_moss.depends import depend_host

depend_host()
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
from ghoshell_moss.host.listener.capture.audio_transport import AudioTransport
from ghoshell_moss.topics.audio import AudioRuntimeTopic
from ghoshell_moss.core.blueprint.session import StreamSubscriber

__all__ = [
    "MiniAudioCaptureSource",
    "MiniAudioSequentialConsumer",
    "unpack_chunk",
    "pack_chunk",
]

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
    """Capture system audio via miniaudio CaptureDevice, publish PCM through transport."""

    def __init__(self, *, transport: AudioTransport, config: AudioCaptureConfig):
        self._transport = transport
        self._config = config
        self._logger = transport.logger
        self._capture: miniaudio.CaptureDevice | None = None
        self._seq = 0
        self._started = False
        self._closing = False

    # -- lifecycle --

    async def start(self) -> None:
        if self._started:
            return

        if not self._transport.acquire_lock():
            self._logger.warning("Audio capture lock held by another process, skipping start")
            self._started = True
            return

        device_id = self._find_device()
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
        self._transport.pub_topic(AudioRuntimeTopic(
            running=True,
            device_name=getattr(self._capture, "name", "") or "default",
            device_explain=self.device_explain(),
            started_at=time.time(),
            last_heartbeat=time.time(),
        ))
        self._logger.info("Audio capture started (device=%s)",
                          self.device_explain())

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

        self._transport.pub_topic(AudioRuntimeTopic(
            running=False,
            last_heartbeat=time.time(),
        ))
        self._transport.release_lock()
        self._started = False
        self._logger.info("Audio capture closed")

    # -- consumer factories --

    def new_consumer(self, ring_buffer_frames: int = 64) -> AudioPullLatest:
        return _MiniAudioPullLatest(
            transport=self._transport,
            maxlen=ring_buffer_frames,
            logger=self._logger,
        )

    def new_sequential_consumer(self, max_queue_frames: int = 128) -> AudioSequentialConsumer:
        return MiniAudioSequentialConsumer(
            transport=self._transport,
            maxsize=max_queue_frames,
            logger=self._logger,
        )

    # -- internals --

    def _find_device(self):
        pattern = self._config.device_pattern.lower()
        try:
            for d in miniaudio.Devices().capture:
                if pattern in d.name.lower():
                    return d.id
        except Exception as e:
            self._logger.warning("Device enumeration failed: %s, using default", e)
        return None

    def _make_capture_generator(self):
        channels = self._config.channels
        frame_duration_ms = self._config.frame_duration_ms
        logger = self._logger
        pub = self._transport.pub_pcm
        seq_ref = [0]

        def _capture_generator():
            while True:
                data = yield
                try:
                    ts = time.time()
                    samples = np.frombuffer(data, dtype=np.int16).reshape(-1, channels)
                    meta = _compute_frame_meta(samples)
                    seq = seq_ref[0]
                    seq_ref[0] += 1

                    chunk = AudioChunk(seq=seq, timestamp=ts, samples=samples, meta=meta)
                    packed = pack_chunk(chunk)
                    pub(packed)
                except Exception:
                    logger.exception("Error in capture callback")

        return _capture_generator()


class _MiniAudioPullLatest(AudioPullLatest):
    """Ring-buffer consumer. Non-blocking, latest frame wins."""

    def __init__(self, *, transport: AudioTransport, maxlen: int, logger):
        self._ring: collections.deque[AudioChunk] = collections.deque(maxlen=maxlen)
        self._logger = logger
        self._release: Callable[[], None] | None = None
        self._closed = False

        self._release = transport.sub_pcm_callback(self._on_sample)

    def _on_sample(self, data: bytes) -> None:
        try:
            chunk = unpack_chunk(data)
            self._ring.append(chunk)
        except Exception:
            pass

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

    def __init__(self, *, transport: AudioTransport, maxsize: int, logger):
        self._transport = transport
        self._maxsize = maxsize
        self._logger = logger
        self._stream: StreamSubscriber | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._stream = self._transport.sub_pcm_stream(maxsize=self._maxsize)
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
        sample = await self._stream.__anext__()
        return unpack_chunk(sample.payload)
