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
import hashlib
import logging
import re
import time
from typing import Callable

from ghoshell_moss.depends import depend_host

depend_host()
import janus
import miniaudio
import numpy as np

from .configs import CaptureConfig
from ghoshell_moss.contracts.audio import (
    AcousticEchoCanceller,
    AudioCaptureSource,
    AudioChunk,
    AudioFrameMeta,
    AudioPullLatest,
    AudioSequentialConsumer,
    resample,
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

#: 丢帧 warning 的节流间隔 (秒) —— 队列满时每帧都记会刷屏, 按此间隔聚合出一条.
_DROP_LOG_INTERVAL = 5.0

#: AEC 连续失败多少次后卸载 (自愈: 降级放行 raw 帧, 不再逐帧丢). AEC 失败通常持续
#: (原生库状态坏), 单个阈值足够 —— 不重试以免继续丢帧.
_AEC_MAX_FAILURES = 5


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


def _device_lock_key(device_name: str | None) -> str:
    """锁 key 由**设备名**派生, 符合 workspace.lock 的 `^[a-zA-Z0-9_-]+$` 约束.

    不能用 miniaudio 的设备 id 派生: 那个 id 是 cdata 指针, ``str()`` 出来是
    ``<cdata 'union ma_device_id *' owning 256 bytes>`` —— 对所有设备都一样, 于是
    "锁具体设备"退化成"锁全局采集", 两台不同的麦克风互相打架. 设备名是唯一能区分
    它们、又能被人读懂的东西.
    """
    if not device_name:
        return f"{_LOCK_KEY_PREFIX}_default"
    s = re.sub(r"[^a-zA-Z0-9_-]", "_", str(device_name))
    # 设备名常是纯中文 (「外置麦克风」「内置麦克风」), 净化后只剩一串下划线会互相撞上;
    # 后缀一段名字哈希, 保证"不同设备必不同 key", 前缀保留可读性.
    digest = hashlib.md5(device_name.encode("utf-8")).hexdigest()[:8]
    return f"{_LOCK_KEY_PREFIX}_{s[:32]}_{digest}"


class MiniAudioCaptureSource(AudioCaptureSource):
    """Capture system audio via miniaudio, fan out locally to in-process consumers."""

    def __init__(
            self,
            *,
            config: CaptureConfig,
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
        self._aec: AcousticEchoCanceller | None = None
        self._aec_failures = 0
        # 设备被别的进程占用 → 生命周期已 enter 但无设备空转 (degraded). 见 start().
        self._degraded = False
        # 解析到的设备名 (None = 系统默认输入), 供 device_explain / 日志说清"到底哪台".
        self._device_name: str | None = None

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

    def set_aec(self, aec: AcousticEchoCanceller | None) -> None:
        """挂载/卸载回声消除器. 挂载后采集帧在 meta 前经 process 消回声."""
        self._aec = aec
        self._aec_failures = 0

    def _apply_aec(self, samples: np.ndarray) -> np.ndarray:
        """public-internal: 采集帧过 AEC (int16 → float32 → process → int16).

        在 meta 计算前调用, 保证 meta (rms/is_silent) 描述消回声后的信号 — 静音
        门控读这个 meta, 不重算能量. 无 AEC 时原样返回. AEC 是单声道表面 (契约),
        接线层据此只对 mono 采集挂载.

        AEC 抛异常时**原样放行 raw 帧**(不丢帧), 连续失败达 ``_AEC_MAX_FAILURES`` 后
        卸载 AEC —— 降级为"无回声消除但有声音", 而不是"逐帧丢帧导致全聋".
        """
        aec = self._aec
        if aec is None:
            return samples
        try:
            mono = samples.ravel().astype(np.float32) / 32768.0
            out = aec.process(mono)
            self._aec_failures = 0
            return (out * 32768.0).clip(-32768, 32767).astype(np.int16).reshape(samples.shape)
        except Exception:
            self._aec_failures += 1
            if self._aec_failures >= _AEC_MAX_FAILURES:
                self._logger.error(
                    "AEC failed %d times — disabling AEC, raw audio continues",
                    self._aec_failures,
                )
                self._aec = None
            return samples

    async def start(self) -> None:
        if self._started:
            return

        device_id, device_name = self._find_device()
        self._device_name = device_name
        lock_key = _device_lock_key(device_name)
        self._locker = self._workspace.lock(lock_key)
        if not self._locker.acquire(timeout=0):
            # 设备被另一进程占用: 生命周期已 enter, 但无设备空转 (degraded). is_running()
            # 不再能区分"在采集"与"无设备", 故显式标记 + 记 error, 供上层/自解释面识别.
            self._logger.error(
                "Audio capture device locked by another process (%s) — starting degraded: "
                "lifecycle entered but NO device captured", lock_key)
            self._locker = None
            self._started = True
            self._degraded = True
            return

        self._degraded = False

        # 打设备名而不是 id: id 是 cdata 指针, 对所有设备 repr 都一样, 打出来读不出"哪台".
        self._logger.info(
            "Audio capture using %s",
            repr(device_name) if device_name else "the system default input device",
        )

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
        if self._degraded:
            return "degraded: device locked by another process, no capture"
        if self._capture is None:
            return "not started"
        where = f"device={self._device_name!r}" if self._device_name else "device=system default input"
        return f"miniaudio capture [{where}], {self._config.sample_rate}Hz, " \
               f"{self._config.channels}ch, {self._config.format}"

    def is_degraded(self) -> bool:
        """生命周期已 enter 但无设备空转 (设备被别的进程占用) — 与 is_running() 区分.

        ``is_running()`` 只表示"已 start 且未 close"(生命周期归属判定用), 无法区分
        有无真实采集; 本方法暴露 degraded 真相, 供上层/自解释面识别"耳朵是空的".
        """
        return self._degraded

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

    def new_sequential_consumer(
        self,
        max_queue_frames: int = 128,
        target_sample_rate: int | None = None,
    ) -> AudioSequentialConsumer:
        return MiniAudioSequentialConsumer(
            capture=self,
            maxsize=max_queue_frames,
            target_sample_rate=target_sample_rate,
            logger=self._logger,
        )

    # -- internals --

    def _find_device(self) -> tuple[object | None, str | None]:
        """按 ``device_pattern`` (名字小写子串) 解析输入设备; 返回 (device_id, name).

        **空 pattern 是"没有指定", 不是"匹配一切".** 空串作为子串命中任何名字, 会静默
        拿到枚举里的第一台 —— macOS 上经常是 BlackHole 这类数字静音设备 (实测 -200dB,
        逐帧完全静音), 于是日志写着"匹配成功", 而耳朵是空的. 这里与 player 的
        ``_find_device`` 对齐: 返回 (None, None), 交给 miniaudio 打开系统默认输入.

        匹配到多台时取第一台; 指定了 pattern 却一台都不匹配时记 warning 再退回默认 ——
        静默退回会让"设备发现失败"永远看不见.
        """
        pattern = (self._config.device_pattern or "").strip().lower()
        if not pattern:
            return None, None
        try:
            for d in miniaudio.Devices().get_captures():
                name = d["name"]
                if pattern in name.lower():
                    return d["id"], name
        except Exception as e:
            self._logger.warning("Device enumeration failed: %s, using default", e)
            return None, None
        self._logger.warning(
            "No input device matches %r — falling back to the system default input",
            self._config.device_pattern,
        )
        return None, None

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
        apply_aec = self._apply_aec

        def _capture_generator():
            while True:
                data = yield
                try:
                    ts = time.time()
                    # miniaudio 会复用底层 buffer, 必须 copy, 否则下一帧覆盖本帧.
                    samples = np.frombuffer(data, dtype=np.int16).reshape(-1, channels).copy()
                    samples = apply_aec(samples)
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

    def __init__(
        self,
        *,
        capture: MiniAudioCaptureSource,
        maxsize: int,
        target_sample_rate: int | None,
        logger,
    ):
        self._capture = capture
        self._maxsize = maxsize
        self._target_sample_rate = target_sample_rate
        self._logger = logger
        self._queue: janus.Queue | None = None
        self._dispose: Callable[[], None] | None = None
        self._started = False
        self._shutdown = False
        # 队列满丢弃的最新帧计数 + 上次记日志时间 (节流). 丢帧是实时流设计允许的,
        # 但必须可观测 — 此前 contextlib.suppress 全吞, 无声无息.
        self._dropped = 0
        self._last_drop_log = 0.0

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
        """采集线程入队. 队列满时丢弃最新帧 (实时流不阻塞采集), 计数 + 节流记 warning.

        丢弃是设计允许的 (消费者跟不上), 但**不再无声** — 计数可经 ``dropped_frames()``
        读, 并按 ``_DROP_LOG_INTERVAL`` 节流记一条 warning.
        """
        if self._queue is None or self._shutdown:
            return
        try:
            self._queue.sync_q.put_nowait(chunk)
        except janus.SyncQueueFull:
            self._dropped += 1
            now = time.monotonic()
            if now - self._last_drop_log >= _DROP_LOG_INTERVAL:
                self._last_drop_log = now
                self._logger.warning(
                    "audio consumer queue full — dropped %d frames so far "
                    "(consumer cannot keep up)", self._dropped)

    def dropped_frames(self) -> int:
        """public-internal: 累计因队列满丢弃的帧数 (实时流背压的可观测面)."""
        return self._dropped

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
        return self._maybe_resample(item)

    def _maybe_resample(self, chunk: AudioChunk) -> AudioChunk:
        """声明了 target_sample_rate 且 != capture 原生率时, 在消费侧重采样.

        运行在 event loop 侧 (__anext__), 不碰采集线程 — 采集线程只能入队, 不能做活.
        """
        if self._target_sample_rate is None or self._target_sample_rate == self._capture.sample_rate:
            return chunk
        samples = np.asarray(chunk.samples).ravel().astype(np.int16)
        samples = resample(
            samples,
            origin_rate=self._capture.sample_rate,
            target_rate=self._target_sample_rate,
        )
        return chunk.model_copy(update={"samples": samples})
