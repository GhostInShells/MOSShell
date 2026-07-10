"""
VPIOCaptureSource — macOS VPIO (Voice Processing IO) audio capture with system-level AEC.

Replaces MiniAudioCaptureSource when running on macOS. Opens AVAudioEngine with
`setVoiceProcessingEnabled(true)` on both input and output nodes — this gives us
system-level acoustic echo cancellation (AEC) for free, so TTS playback through
the system default output is automatically subtracted from the mic signal.

Key constraints (see docs/VPIO.md):
1. VPIO audio unit MUST be initialized at 48kHz (or 44.1kHz) — hardware constraint.
2. Both input + output nodes must enable VPIO for AEC to engage.
3. TTS must play through system default output so VPIO can grab far-end reference.
4. AVAudioConverter resamples 48k → 16k for ASR (Volcengine bigmodel_async).
5. Tap callback fires on CoreAudio real-time thread — push bytes to an asyncio
   queue and let the asyncio side do pack_chunk + transport.pub_pcm.

Output format matches MiniAudioCaptureSource: 16kHz / 1ch / int16 PCM, packaged
via the same pack_chunk() wire format. Listener app consumes unchanged.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import time
from typing import Callable

import numpy as np

from ghoshell_moss.contracts.audio import (
    AudioCaptureConfig,
    AudioCaptureSource,
    AudioChunk,
    AudioFrameMeta,
    AudioPullLatest,
    AudioSequentialConsumer,
)
from ghoshell_moss.host.speech.capture.audio_transport import AudioTransport
from ghoshell_moss.host.speech.capture.miniaudio_capture import (
    MiniAudioSequentialConsumer,
    _compute_frame_meta,
    pack_chunk,
    unpack_chunk,
)
from ghoshell_moss.topics.audio import AudioRuntimeTopic

__all__ = ["VPIOCaptureSource"]


# VPIO hardware constraint — must be 48k (or 44.1k). We pick 48k.
_VPIO_NATIVE_SR = 48000
# Output sample rate consumed by ASR (matches listener's _ASR_SAMPLE_RATE).
_OUTPUT_SR = 16000
# Frame duration in ms — same as MiniAudioCaptureSource default.
_FRAME_MS = 50
_MAX_DIAG_CHANNELS = 16


def _channel_mode() -> str:
    mode = os.environ.get("VPIO_CHANNEL_MODE", "best").strip().lower()
    if mode in {"best", "mix", "0"}:
        return mode
    return "best"


class VPIOCaptureSource(AudioCaptureSource):
    """macOS VPIO capture source — system-level AEC, drop-in for MiniAudioCaptureSource.

    Output to transport: 16kHz / mono / int16 PCM, packaged via pack_chunk().
    Consumer side (listener / waveform) is unchanged.
    """

    def __init__(self, *, transport: AudioTransport, config: AudioCaptureConfig | None = None):
        self._transport = transport
        # We override sample_rate to 16k regardless of config — ASR contract.
        # config is kept for compatibility (channels, frame_duration_ms, device_pattern).
        self._config = config or AudioCaptureConfig()
        self._logger = transport.logger
        self._seq = 0
        self._started = False
        self._closing = False

        # AVAudioEngine state
        self._engine = None
        self._input_node = None
        self._output_node = None
        self._input_tap_bus = 0
        self._tap_format = None

        # Thread bridge: CoreAudio real-time thread → asyncio loop
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[np.ndarray | None] | None = None
        self._pump_task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._last_frame_at = 0.0
        self._last_stall_report_at = 0.0
        self._restart_lock: asyncio.Lock | None = None
        self._last_restart_at = 0.0
        self._restart_attempts = 0

    # -- lifecycle --

    async def start(self) -> None:
        if self._started:
            return

        if not self._transport.acquire_lock():
            self._logger.warning("Audio capture lock held by another process, skipping start")
            self._started = True
            return

        self._loop = asyncio.get_running_loop()
        # Bounded queue: if asyncio side falls behind, drop oldest to keep latency bounded.
        self._queue = asyncio.Queue(maxsize=64)
        self._restart_lock = asyncio.Lock()

        try:
            await self._start_engine()
        except Exception:
            await self._cleanup_engine()
            self._transport.release_lock()
            raise

        # Start asyncio pump — drains queue, packs chunks, pub to transport
        self._last_frame_at = time.monotonic()
        self._pump_task = asyncio.create_task(self._pump_loop())
        self._watchdog_task = asyncio.create_task(self._watchdog_loop())

        self._started = True
        self._transport.pub_topic(AudioRuntimeTopic(
            running=True,
            device_name="vpio",
            device_explain=self.device_explain(),
            started_at=time.time(),
            last_heartbeat=time.time(),
        ))
        self._logger.info("VPIO: capture started (%s)", self.device_explain())

    def device_explain(self) -> str:
        if self._engine is None:
            return "not started"
        return (f"macOS VPIO, native={_VPIO_NATIVE_SR}Hz → out={_OUTPUT_SR}Hz, "
                f"1ch, pcm_s16le, AEC=system-level")

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True

        # Stop pump first so we don't publish partial frames during teardown
        if self._pump_task is not None:
            if self._queue is not None:
                try:
                    self._queue.put_nowait(None)  # sentinel
                except asyncio.QueueFull:
                    try:
                        self._queue.get_nowait()
                        self._queue.put_nowait(None)
                    except asyncio.QueueEmpty:
                        pass
            try:
                await asyncio.wait_for(self._pump_task, timeout=2.0)
            except asyncio.TimeoutError:
                self._pump_task.cancel()
            except Exception:
                pass
            self._pump_task = None

        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None

        await self._cleanup_engine()

        self._transport.pub_topic(AudioRuntimeTopic(
            running=False,
            last_heartbeat=time.time(),
        ))
        self._transport.release_lock()
        self._started = False
        self._logger.info("VPIO: capture closed")

    # -- consumer factories (same as MiniAudio — they consume from transport, agnostic to source) --

    def new_consumer(self, ring_buffer_frames: int = 64) -> AudioPullLatest:
        # Reuse the same _MiniAudioPullLatest — it only reads transport stream.
        from ghoshell_moss.host.speech.capture.miniaudio_capture import _MiniAudioPullLatest
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

    async def _start_engine(self) -> None:
        # Lazy import — only fails on non-macOS or missing pyobjc
        try:
            import AVFoundation  # noqa: F401
            from AVFoundation import AVAudioEngine  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "VPIOCaptureSource requires macOS with pyobjc-framework-AVFoundation. "
                f"Import failed: {e}. Install with: uv pip install pyobjc-framework-AVFoundation"
            ) from e

        from AVFoundation import AVAudioEngine

        # 1) Build engine + enable VPIO on BOTH input and output nodes.
        self._engine = AVAudioEngine.new()
        self._input_node = self._engine.inputNode()
        self._output_node = self._engine.outputNode()

        # Enable VPIO on input — this is where AEC lives for the mic side.
        # PyObjC: setVoiceProcessingEnabled_error_(value, error_ptr) → returns BOOL.
        try:
            ok = self._input_node.setVoiceProcessingEnabled_error_(True, None)
            if isinstance(ok, tuple):
                ok, err = ok
            else:
                err = None
            if not ok:
                self._logger.warning("VPIO: inputNode VPIO enable failed (AEC may not engage): %s",
                                     err and err.localizedDescription())
            else:
                self._logger.info("VPIO: inputNode.setVoiceProcessingEnabled = True")
        except Exception as e:
            self._logger.warning("VPIO: inputNode VPIO enable failed (AEC may not engage): %s", e)

        # Enable VPIO on output — required for AEC to engage.
        try:
            ok = self._output_node.setVoiceProcessingEnabled_error_(True, None)
            if isinstance(ok, tuple):
                ok, err = ok
            else:
                err = None
            if not ok:
                self._logger.warning("VPIO: outputNode VPIO enable failed (AEC may not engage): %s",
                                     err and err.localizedDescription())
            else:
                self._logger.info("VPIO: outputNode.setVoiceProcessingEnabled = True")
        except Exception as e:
            self._logger.warning("VPIO: outputNode VPIO enable failed (AEC may not engage): %s", e)

        # Report what VPIO actually applied (settings may downgrade silently).
        self._log_vpio_diagnostics()

        # 2) Native tap format = hardware format of inputNode (typically 48k / 1ch / float32).
        self._tap_format = self._input_node.outputFormatForBus_(self._input_tap_bus)
        native_sr = int(self._tap_format.sampleRate())
        native_ch = int(self._tap_format.channelCount())
        self._logger.info(
            "VPIO: tap format native_sr=%d ch=%d commonFormat=%s",
            native_sr, native_ch, self._tap_format.commonFormat(),
        )

        if native_sr != _VPIO_NATIVE_SR:
            self._logger.warning(
                "VPIO: native_sr=%d is not %d — AEC may fail on some macOS versions",
                native_sr, _VPIO_NATIVE_SR,
            )

        # 3) Resampling is done on the asyncio side via scipy.signal.resample_poly.

        # 4) Install tap on input bus — callback fires on CoreAudio real-time thread.
        self._install_tap()

        # 5) Start engine — startAndReturnError_ returns (BOOL success, NSError* error) tuple.
        try:
            self._engine.prepare()
            ok, err = self._engine.startAndReturnError_(None)
            if not ok:
                raise RuntimeError(f"engine.start failed: {err and err.localizedDescription()}")
        except Exception as e:
            self._logger.exception("VPIO: engine start failed: %s", e)
            await self._cleanup_engine()
            raise

    def _drain_audio_queue(self) -> int:
        if self._queue is None:
            return 0
        drained = 0
        while True:
            try:
                self._queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                return drained

    async def _recover_from_stall(self, age: float) -> None:
        if self._restart_lock is None or self._closing:
            return

        async with self._restart_lock:
            if self._closing:
                return
            fresh_age = time.monotonic() - self._last_frame_at
            if fresh_age < age:
                return

            self._last_restart_at = time.monotonic()
            self._restart_attempts += 1
            self._logger.warning(
                "VPIO stalled: restarting AVAudioEngine after %.1fs without frames (attempt=%d)",
                fresh_age,
                self._restart_attempts,
            )
            self._transport.pub_topic(AudioRuntimeTopic(
                running=False,
                device_name="vpio",
                device_explain=f"state=restarting,no_frames_for={fresh_age:.1f}s,attempt={self._restart_attempts}",
                started_at=time.time(),
                last_heartbeat=time.time(),
            ))

            try:
                await self._cleanup_engine()
                drained = self._drain_audio_queue()
                if drained:
                    self._logger.info("VPIO: drained %d queued audio frames before restart", drained)
                await asyncio.sleep(0.2)
                await self._start_engine()
                self._last_frame_at = time.monotonic()
                self._last_stall_report_at = 0.0
                self._transport.pub_topic(AudioRuntimeTopic(
                    running=True,
                    device_name="vpio",
                    device_explain=f"state=restarted,attempt={self._restart_attempts}",
                    started_at=time.time(),
                    last_heartbeat=time.time(),
                ))
                self._logger.info("VPIO: AVAudioEngine restart completed")
            except Exception as e:
                self._logger.exception("VPIO: AVAudioEngine restart failed: %s", e)
                self._transport.pub_topic(AudioRuntimeTopic(
                    running=False,
                    device_name="vpio",
                    device_explain=f"state=restart_failed,error={type(e).__name__}",
                    started_at=time.time(),
                    last_heartbeat=time.time(),
                ))

    def _log_vpio_diagnostics(self) -> None:
        """Report what VPIO actually applied (may downgrade silently)."""
        try:
            in_vpio = self._input_node.isVoiceProcessingEnabled()
            out_vpio = self._output_node.isVoiceProcessingEnabled()
            self._logger.info(
                "VPIO report · input.vpio=%s · output.vpio=%s · "
                "(both must be True for AEC to engage)",
                in_vpio, out_vpio,
            )
        except Exception as e:
            self._logger.warning("VPIO report failed: %s", e)

    def _install_tap(self) -> None:
        """Install tap on inputNode bus 0 with native format.

        Callback runs on CoreAudio real-time thread. Per VPIO.md §4.6 we keep
        the hot path minimal: copy float32 channels off the realtime buffer
        into a numpy array, hand it to the asyncio loop via
        call_soon_threadsafe. All heavy work (resample 48k→16k, int16
        conversion, FFT meta, pack_chunk, transport.pub_pcm) happens on the
        asyncio side in _pump_loop.
        """
        buf_size = int(_VPIO_NATIVE_SR * _FRAME_MS / 1000)  # 48k * 50ms = 2400 samples
        loop = self._loop

        def _tap_callback(buffer, when):
            # buffer: AVAudioPCMBuffer at native 48k/float32/Nch (post-AEC).
            # Mac input devices can expose multi-channel arrays. Channel 0 is
            # not always the speech-dominant channel, especially with external
            # or aggregate devices, so copy all channels and choose on asyncio.
            # floatChannelData() returns a tuple of objc.varlist objects;
            # varlist[0:n] returns a Python list of floats — fastest path
            # through PyObjC (~0.14ms for 4800 samples). Copy is mandatory
            # because the underlying buffer is owned by CoreAudio.
            try:
                n = int(buffer.frameLength())
                if n == 0:
                    return
                ch_data = buffer.floatChannelData()
                if ch_data is None:
                    return
                channel_count = min(int(buffer.format().channelCount()), len(ch_data), _MAX_DIAG_CHANNELS)
                if channel_count <= 1:
                    arr = np.array(ch_data[0][0:n], dtype=np.float32)
                else:
                    channels = [
                        np.array(ch_data[ch][0:n], dtype=np.float32)
                        for ch in range(channel_count)
                    ]
                    arr = np.stack(channels, axis=0)
                try:
                    loop.call_soon_threadsafe(self._enqueue, arr)
                except RuntimeError:
                    # loop closed during shutdown
                    pass
            except Exception:
                # Don't log on the realtime thread — just swallow
                pass

        self._input_node.installTapOnBus_bufferSize_format_block_(
            self._input_tap_bus,
            buf_size,
            self._tap_format,
            _tap_callback,
        )
        self._logger.info("VPIO: tap installed on bus %d, bufSize=%d", self._input_tap_bus, buf_size)

    def _enqueue(self, pcm: np.ndarray) -> None:
        """Called on the asyncio loop via call_soon_threadsafe — safe to put_nowait."""
        if self._queue is None or self._closing:
            return
        self._last_frame_at = time.monotonic()
        if self._queue.full():
            # Drop oldest to bound latency — better to lose a frame than to lag.
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        try:
            self._queue.put_nowait(pcm)
        except asyncio.QueueFull:
            pass

    async def _watchdog_loop(self) -> None:
        """Publish diagnostics and recover if CoreAudio tap stops producing frames."""
        threshold = float(os.environ.get("VPIO_STALL_SECONDS", "2.5") or "2.5")
        restart_threshold = float(os.environ.get("VPIO_RESTART_STALL_SECONDS", "8.0") or "8.0")
        restart_cooldown = float(os.environ.get("VPIO_RESTART_COOLDOWN_SECONDS", "5.0") or "5.0")
        auto_restart = os.environ.get("VPIO_AUTO_RESTART_ON_STALL", "1").strip() != "0"
        while not self._closing:
            await asyncio.sleep(0.5)
            if not self._started:
                continue
            age = time.monotonic() - self._last_frame_at
            if age < threshold:
                continue
            now = time.monotonic()
            if now - self._last_stall_report_at < threshold:
                continue
            self._last_stall_report_at = now
            diag = f"state=stalled,no_frames_for={age:.1f}s"
            self._logger.warning("VPIO stalled: no frames for %.1fs", age)
            self._transport.pub_topic(AudioRuntimeTopic(
                running=False,
                device_name="vpio",
                device_explain=diag,
                started_at=time.time(),
                last_heartbeat=time.time(),
            ))
            if (
                auto_restart
                and age >= restart_threshold
                and now - self._last_restart_at >= restart_cooldown
            ):
                await self._recover_from_stall(age)

    async def _pump_loop(self) -> None:
        """Drain queue on the asyncio side.

        Per-frame work: resample 48k float32 → 16k int16, build AudioChunk,
        pack_chunk, transport.pub_pcm. All heavy work lives here, not in the
        realtime tap callback.
        """
        import math
        from scipy import signal as _scipy_signal

        assert self._queue is not None
        # 48000 → 16000: gcd=16000, up=1, down=3
        _up = _OUTPUT_SR
        _down = _VPIO_NATIVE_SR
        _g = math.gcd(_up, _down)
        _up_r = _up // _g   # 1
        _down_r = _down // _g  # 3
        stat_next_log = time.monotonic() + 1.0
        stat_frames = 0
        stat_max_rms = 0.0
        stat_max_peak = 0.0
        stat_best_ch_counts: collections.Counter[int] = collections.Counter()
        stat_max_channel_rms: np.ndarray | None = None
        channel_mode = _channel_mode()
        self._logger.info("VPIO: channel mode=%s (env VPIO_CHANNEL_MODE=best|mix|0)", channel_mode)

        while True:
            pcm_f32 = await self._queue.get()
            if pcm_f32 is None:  # shutdown sentinel
                return
            try:
                ts = time.time()
                # Resample 48k → 16k using polyphase filter (matches listener's resampler)
                if pcm_f32.size == 0:
                    continue
                best_ch = 0
                channel_rms = None
                if pcm_f32.ndim == 2:
                    channel_rms = np.sqrt(np.mean(pcm_f32 * pcm_f32, axis=1))
                    best_ch = int(np.argmax(channel_rms)) if channel_rms.size else 0
                    stat_best_ch_counts[best_ch] += 1
                    if stat_max_channel_rms is None or stat_max_channel_rms.shape != channel_rms.shape:
                        stat_max_channel_rms = channel_rms.copy()
                    else:
                        stat_max_channel_rms = np.maximum(stat_max_channel_rms, channel_rms)

                    if channel_mode == "mix":
                        pcm_f32 = np.mean(pcm_f32, axis=0, dtype=np.float32)
                    elif channel_mode == "0":
                        pcm_f32 = pcm_f32[0]
                    else:
                        pcm_f32 = pcm_f32[best_ch]
                resampled = _scipy_signal.resample_poly(
                    pcm_f32.astype(np.float32), _up_r, _down_r,
                )
                resampled_f32 = resampled.astype(np.float32, copy=False)
                rms = float(np.sqrt(np.mean(resampled_f32 * resampled_f32))) if resampled_f32.size else 0.0
                peak = float(np.max(np.abs(resampled_f32))) if resampled_f32.size else 0.0
                stat_frames += 1
                stat_max_rms = max(stat_max_rms, rms)
                stat_max_peak = max(stat_max_peak, peak)
                # float32 [-1.0, 1.0] → int16 [-32768, 32767]
                pcm_i16 = (resampled * 32767.0).clip(-32768, 32767).astype(np.int16)
                # 2D shape (n, 1) — matches MiniAudioCaptureSource convention
                samples = pcm_i16.reshape(-1, 1)
                meta = _compute_frame_meta(samples)
                seq = self._seq
                self._seq += 1
                chunk = AudioChunk(seq=seq, timestamp=ts, samples=samples, meta=meta)
                packed = pack_chunk(chunk)
                self._transport.pub_pcm(packed)
                now = time.monotonic()
                if now >= stat_next_log:
                    top_channels = ""
                    if stat_max_channel_rms is not None:
                        ranked = sorted(
                            enumerate(stat_max_channel_rms.tolist()),
                            key=lambda item: item[1],
                            reverse=True,
                        )[:4]
                        top_channels = ",".join(f"{idx}:{rms:.4f}" for idx, rms in ranked)
                    dominant_ch = stat_best_ch_counts.most_common(1)[0][0] if stat_best_ch_counts else 0
                    diag = (
                        f"mode={channel_mode},best_ch={dominant_ch},"
                        f"ch_rms={top_channels},rms={stat_max_rms:.5f},"
                        f"peak={stat_max_peak:.5f},frames={stat_frames}"
                    )
                    self._logger.info(
                        "VPIO audio stats · frames=%d · max_rms=%.5f · max_peak=%.5f · "
                        "best_ch=%d · ch_rms=%s · mode=%s · seq=%d",
                        stat_frames,
                        stat_max_rms,
                        stat_max_peak,
                        dominant_ch,
                        top_channels or "n/a",
                        channel_mode,
                        seq,
                    )
                    self._transport.pub_topic(AudioRuntimeTopic(
                        running=True,
                        device_name="vpio",
                        device_explain=diag,
                        started_at=time.time(),
                        last_heartbeat=time.time(),
                    ))
                    stat_next_log = now + 1.0
                    stat_frames = 0
                    stat_max_rms = 0.0
                    stat_max_peak = 0.0
                    stat_best_ch_counts.clear()
                    stat_max_channel_rms = None
            except Exception:
                self._logger.exception("VPIO: error in pump loop")

    async def _cleanup_engine(self) -> None:
        if self._input_node is not None:
            try:
                self._input_node.removeTapOnBus_(self._input_tap_bus)
                self._logger.info("VPIO: tap removed")
            except Exception as e:
                self._logger.warning("VPIO: removeTap failed: %s", e)
            self._input_node = None

        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass
            try:
                self._engine.reset()
            except Exception:
                pass
            self._engine = None

        self._output_node = None
        self._tap_format = None
