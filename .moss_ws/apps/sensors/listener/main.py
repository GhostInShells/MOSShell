"""Listener App — ASR consumer.

Consumes PCM stream from audio_capture, feeds to Volcengine ASR,
publishes SpeechTopic on final recognition, emits AudioSignal to mindflow.

Usage:
    moss apps test sensors/listener
    moss apps start sensors/listener
"""
import asyncio
import json
import logging
import math
import os
import re
import time
from collections.abc import AsyncIterable
from dataclasses import dataclass
from pathlib import Path

import dotenv
import numpy as np

dotenv.load_dotenv(Path(__file__).resolve().parent / ".env")
from scipy import signal

from ghoshell_moss.contracts.asr import ASRResult
from ghoshell_moss.contracts.audio import (
    AudioCaptureConfig,
    AudioChunk,
)
from ghoshell_moss.core.mindflow.audio_signal import AudioAction, AudioSignal
from ghoshell_moss.host.speech.capture.audio_transport import AudioTransport
from ghoshell_moss.topics.audio import AudioRuntimeTopic, SpeechTopic
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.host.speech.capture.matrix_audio_transport import MatrixAudioTransport
from ghoshell_moss.host.speech.capture.miniaudio_capture import MiniAudioCaptureSource
from ghoshell_moss.host.speech.volcengine_asr import VolcengineASR, VolcengineASRConfig
from ghoshell_moss.message import Message
from ghoshell_moss.core.blueprint.mindflow import Signal, Priority, unique_id

# ASR 期望的采样率 (16kHz 是语音识别的行业标准)
_ASR_SAMPLE_RATE = 16000
_VOLCENGINE_ASR_ERROR_PREFIX = "__VOLCENGINE_ASR_ERROR__:"
_WAKE_WORDS = (
    "立刻停下",
    "立即停下",
    "马上停下",
    "快停下",
    "停下",
    "别说了",
    "不要说了",
    "先停",
    "闭嘴",
)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _wake_word_hit(text: str) -> bool:
    normalized = re.sub(r"[\s,，。.!！?？、；;：:「」『』“”\"'`~～-]", "", text)
    return any(word in normalized for word in _WAKE_WORDS)


def _normalized_text_len(text: str) -> int:
    return len(re.sub(r"[\s,，。.!！?？、；;：:「」『』“”\"'`~～-]", "", text))


def _asr_diag_payload(source: str = "volcengine_asr", **kwargs) -> str:
    return json.dumps(
        {
            "source": source,
            **kwargs,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


@dataclass
class _ASRControlState:
    mode: str
    enabled: bool
    last_started_at: float = 0.0
    last_heartbeat: float = 0.0


def _initial_asr_control() -> _ASRControlState:
    mode = os.environ.get("LISTENER_ASR_MODE", "continuous").strip().lower()
    if mode not in {"continuous", "manual"}:
        mode = "continuous"
    return _ASRControlState(mode=mode, enabled=(mode == "continuous"))


def _refresh_asr_control(runtime_window, state: _ASRControlState) -> _ASRControlState:
    """Refresh frontend ASR control mode and keep it sticky.

    ``continuous`` keeps the current behavior: listener opens ASR sessions
    continuously. ``manual`` only opens ASR while enabled=True.

    The control topic is a command, not a transient runtime diagnostic. VPIO and
    ASR diagnostics can evict it from a small topic window within seconds, so
    absence of a recent control topic must not reset the listener to defaults.
    """
    for topic in reversed(list(runtime_window.values())):
        if getattr(topic, "device_name", "") != "asr_control":
            continue
        started_at = float(getattr(topic, "started_at", 0.0) or 0.0)
        heartbeat = float(getattr(topic, "last_heartbeat", 0.0) or 0.0)
        if (started_at, heartbeat) <= (state.last_started_at, state.last_heartbeat):
            break
        try:
            payload = json.loads(getattr(topic, "device_explain", "") or "{}")
        except Exception:
            break
        next_mode = str(payload.get("mode", state.mode)).strip().lower()
        if next_mode in {"continuous", "manual"}:
            state.mode = next_mode
        state.enabled = bool(payload.get("enabled", state.mode == "continuous"))
        state.last_started_at = started_at
        state.last_heartbeat = heartbeat
        break
    if state.mode == "continuous":
        state.enabled = True
    return state


def _resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """重采样音频到目标采样率。使用 scipy.signal.resample_poly 保证质量。"""
    if orig_sr == target_sr:
        return samples
    # 44100 -> 16000: up=160, down=441
    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return signal.resample_poly(samples.astype(np.float32), up, down).astype(np.int16)


async def _audio_generator(
    consumer,
    orig_sr: int,
    target_sr: int,
    runtime_window,
    control_state: _ASRControlState,
    abort_event: asyncio.Event,
    logger: logging.Logger,
    initial_chunk: AudioChunk | None = None,
    frame_timeout: float = 2.0,
) -> AsyncIterable[np.ndarray]:
    """Yield resampled np.ndarray samples from AudioSequentialConsumer.

    Uses an internal asyncio.Queue buffer so that ``aclose()`` (called when
    ``asr.recognize()`` finishes) does NOT reach ``consumer.__anext__()``
    and silently drop a chunk.

    NOTE: TTS playback no longer aborts the generator. Instead, ASR continues
    recognizing during TTS. In Aether mode VPIO provides system AEC, so
    non-wake-word user speech during TTS must still become a normal turn.
    """
    # Unbounded queue: pump must never block on put(), otherwise cancellation
    # can land inside put() and the None sentinel never reaches the reader.
    buffer: asyncio.Queue[AudioChunk | None] = asyncio.Queue()

    async def _pump() -> None:
        """Read from consumer into buffer. Stops on cancellation only."""
        try:
            if initial_chunk is not None:
                buffer.put_nowait(initial_chunk)
            async for chunk in consumer:
                buffer.put_nowait(chunk)
        except asyncio.CancelledError:
            pass
        finally:
            # Sentinel so the generator side exits cleanly.
            # put_nowait is used so cancellation cannot intercept us.
            buffer.put_nowait(None)

    pump_task = asyncio.create_task(_pump())
    try:
        while True:
            control = _refresh_asr_control(runtime_window, control_state)
            if control.mode == "manual" and not control.enabled:
                logger.info("ASR manual gate closed; ending current audio stream")
                abort_event.set()
                break
            try:
                chunk = await asyncio.wait_for(buffer.get(), timeout=frame_timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "ASR audio input stalled for %.1fs; ending current audio stream",
                    frame_timeout,
                )
                abort_event.set()
                break
            if chunk is None:
                break
            yield _resample_audio(chunk.samples, orig_sr, target_sr)
    finally:
        pump_task.cancel()
        try:
            await pump_task
        except asyncio.CancelledError:
            pass


async def _iter_with_silence_timeout(
    agen,
    logger: logging.Logger,
    patience: float = 5.0,
    min_timeout_final_chars: int = 2,
    first_result_timeout: float = 90.0,
) -> AsyncIterable:
    """Wrap an async generator with a silence timeout.

    After the first non-empty result, if no subsequent non-empty result
    arrives within *patience* seconds, the iteration stops.  Empty-text
    results (server keep-alive / VAD status) do NOT reset the timer.

    If the server never sends ``is_final=True`` before the timeout fires,
    this wrapper synthesizes a final result from the last partial text.
    Without this, the utterance is silently lost — no SpeechTopic published,
    no SPEECH_FINAL emitted — and the next recognition loop starts fresh.
    """
    timeout = first_result_timeout
    last_result: ASRResult | None = None
    try:
        while True:
            try:
                result = await asyncio.wait_for(agen.__anext__(), timeout=timeout)
                if result.text:
                    last_result = result
                    timeout = patience
                yield result
            except asyncio.TimeoutError:
                if last_result is None:
                    logger.warning(
                        "ASR first-result timeout after %.1fs, restarting recognition",
                        first_result_timeout,
                    )
                elif not last_result.is_final:
                    logger.info("ASR silence timeout after %.1fs, finalizing", patience)
                    if _normalized_text_len(last_result.text) >= min_timeout_final_chars:
                        logger.info(
                            "Server never sent is_final=True — synthesizing from last partial: %s",
                            last_result.text,
                        )
                        yield ASRResult(text=last_result.text, is_final=True)
                    else:
                        logger.info(
                            "ASR timeout partial too short, dropping fragment: %s",
                            last_result.text,
                        )
                break
            except StopAsyncIteration:
                break
    finally:
        await agen.aclose()


async def _drain_consumer(consumer, timeout: float = 0.1, max_chunks: int = 5) -> int:
    """Discard queued audio chunks to clear TTS residue.

    Limits both timeout-per-read and total chunks to avoid draining user speech.
    Returns the number of chunks drained.
    """
    drained = 0
    while drained < max_chunks:
        try:
            await asyncio.wait_for(consumer.__anext__(), timeout=timeout)
            drained += 1
        except asyncio.TimeoutError:
            break
        except StopAsyncIteration:
            break
    return drained


def _is_tts_playing(runtime_window, logger: logging.Logger | None = None) -> bool:
    """检查 TTS 扬声器是否正在播放中。

    AudioRuntimeTopic 是状态快照。从最新往最旧查，找到 speaker
    的最新状态即可；旧的状态可能已被 running=False 覆盖。

    环境变量 ``LISTENER_DISABLE_TTS_GATE=1`` 可关闭此门控。
    Aether 的 VPIO AEC 场景默认允许 TTS 播放时继续接收用户语音；如果
    需要旧的保守回声过滤，可设置 ``LISTENER_GATE_DURING_TTS=1``。
    """
    if os.environ.get("LISTENER_DISABLE_TTS_GATE") == "1":
        return False
    for topic in reversed(runtime_window.values()):
        if topic.device_name == "speaker":
            if logger and topic.running:
                logger.info("TTS gate: speaker running=%s (window size=%d)", topic.running, len(runtime_window))
            return topic.running
    return False


async def main(matrix: Matrix) -> None:
    logger = matrix.logger or logging.getLogger("moss.listener")
    logger.info("Listener app starting")

    # -- transport & source (consumer only, do not start capture) --
    transport: AudioTransport = MatrixAudioTransport(matrix=matrix)
    capture_config = AudioCaptureConfig()
    # Aether's vpio_capture publishes 16kHz mono PCM. The legacy MiniAudio
    # capture path used AudioCaptureConfig.sample_rate (44.1k by default), but
    # applying that default to VPIO double-resamples 16k audio and corrupts ASR
    # timing. Keep this env-tunable for non-VPIO listener modes.
    input_sample_rate = _env_int("LISTENER_INPUT_SAMPLE_RATE", _ASR_SAMPLE_RATE)
    source = MiniAudioCaptureSource(transport=transport, config=capture_config)
    consumer = source.new_sequential_consumer(max_queue_frames=128)
    await consumer.start()
    logger.info("Audio sequential consumer started (input_sample_rate=%d)", input_sample_rate)

    # -- Subscribe to AudioRuntimeTopic for TTS gating --
    runtime_window = transport.topic_window(AudioRuntimeTopic, max_size=256)
    logger.info("Subscribed to AudioRuntimeTopic window for TTS gating and ASR control")

    # -- ASR (16kHz 是语音识别的标准采样率; 如果 capture 不是 16kHz 则重采样) --
    # end_window_size: 服务端静音判停阈值。火山官方建议 800ms 或 1000ms；
    # 过小会切碎句子，过大则明显拖慢 listen -> think。
    asr_end_window_ms = _env_int("LISTENER_ASR_END_WINDOW_MS", 1000)
    silence_patience = _env_float("LISTENER_SILENCE_PATIENCE", 3.2)
    logger.info(
        "ASR segmentation config: end_window_size=%dms, silence_patience=%.1fs",
        asr_end_window_ms,
        silence_patience,
    )
    asr_source = "volcengine_asr"
    asr_config = VolcengineASRConfig(
        sample_rate=_ASR_SAMPLE_RATE,
        end_window_size=asr_end_window_ms,
        force_to_speech_time=_env_int("VOLCENGINE_BM_ASR_FORCE_TO_SPEECH_TIME_MS", 1000),
    )
    asr = VolcengineASR(config=asr_config, logger=logger)
    logger.info("ASR backend selected: %s", asr_source)

    # -- main recognition loop --
    try:
        consecutive_asr_errors = 0
        asr_control = _initial_asr_control()
        while True:
            asr_control = _refresh_asr_control(runtime_window, asr_control)
            if asr_control.mode == "manual" and not asr_control.enabled:
                if consecutive_asr_errors:
                    consecutive_asr_errors = 0
                transport.pub_topic(AudioRuntimeTopic(
                    running=False,
                    device_name="asr",
                    device_explain=_asr_diag_payload(
                        source=asr_source,
                        state="manual_idle",
                        mode=asr_control.mode,
                    ),
                    started_at=time.monotonic(),
                    last_heartbeat=time.monotonic(),
                ))
                await asyncio.sleep(0.08)
                continue

            logger.info("Waiting for speech...")
            preflight_timeout = _env_float("LISTENER_PRE_ASR_AUDIO_TIMEOUT", 2.0)
            try:
                first_chunk = await asyncio.wait_for(consumer.__anext__(), timeout=preflight_timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "ASR preflight: no audio frame for %.1fs; not opening ASR connection",
                    preflight_timeout,
                )
                transport.pub_topic(AudioRuntimeTopic(
                    running=False,
                    device_name="asr",
                    device_explain=_asr_diag_payload(
                        source=asr_source,
                        state="audio_stalled",
                        timeout=preflight_timeout,
                    ),
                    started_at=time.monotonic(),
                    last_heartbeat=time.monotonic(),
                ))
                await asyncio.sleep(0.2)
                continue
            except StopAsyncIteration:
                logger.warning("ASR preflight: audio consumer stopped")
                await asyncio.sleep(0.2)
                continue

            # NOTE: 不再在 TTS 播放时 hold ASR。
            # VPIO AEC 已经在系统层抑制扬声器回声；如果仍在这里把
            # speaker running 时的非唤醒词结果丢弃，用户在 speak 期间说的话
            # 就永远不会发布 SpeechTopic，前端会表现成 listen 后直接 idle。

            # Fresh abort flag and utterance id for this utterance.
            abort_event = asyncio.Event()
            utterance_id = unique_id()
            started_emitted = False
            asr_running_published = False

            utterance_published = False

            # Each recognize call handles one utterance.
            # The ASR backend (end_window_size) splits on silence.
            audio_gen = _audio_generator(
                consumer,
                input_sample_rate,
                _ASR_SAMPLE_RATE,
                runtime_window,
                asr_control,
                abort_event,
                logger,
                initial_chunk=first_chunk,
                frame_timeout=_env_float("LISTENER_AUDIO_FRAME_TIMEOUT", 2.0),
            )
            async for result in _iter_with_silence_timeout(
                asr.recognize(audio_gen),
                logger,
                patience=silence_patience,
                first_result_timeout=60.0,
            ):
                if result.text.startswith(_VOLCENGINE_ASR_ERROR_PREFIX):
                    raw = result.text.removeprefix(_VOLCENGINE_ASR_ERROR_PREFIX)
                    code, _, message = raw.partition("|")
                    consecutive_asr_errors += 1
                    backoff = min(20.0, 2.0 * consecutive_asr_errors)
                    logger.warning(
                        "ASR server error %s; message=%s; backing off %.1fs before reconnect (consecutive=%d)",
                        code,
                        message[:300],
                        backoff,
                        consecutive_asr_errors,
                    )
                    transport.pub_topic(AudioRuntimeTopic(
                        running=False,
                        device_name="asr",
                        device_explain=_asr_diag_payload(
                            source=asr_source,
                            error="server_error",
                            code=code,
                            message=message,
                            backoff=backoff,
                            consecutive=consecutive_asr_errors,
                        ),
                        started_at=time.monotonic(),
                        last_heartbeat=time.monotonic(),
                    ))
                    await asyncio.sleep(backoff)
                    break

                if result.text:
                    logger.info("ASR partial: %s (final=%s)", result.text, result.is_final)
                    consecutive_asr_errors = 0
                    asr_explain = _asr_diag_payload(
                        source=asr_source,
                        text=result.text,
                        final=bool(result.is_final),
                    )
                    if not asr_running_published:
                        transport.pub_topic(AudioRuntimeTopic(
                            running=True,
                            device_name="asr",
                            device_explain=asr_explain,
                            started_at=time.monotonic(),
                            last_heartbeat=time.monotonic(),
                        ))
                        asr_running_published = True
                    else:
                        transport.pub_topic(AudioRuntimeTopic(
                            running=True,
                            device_name="asr",
                            device_explain=asr_explain,
                            started_at=time.monotonic(),
                            last_heartbeat=time.monotonic(),
                        ))

                # 全双工核心：检测明确停止意图 → 立刻打断。
                # 不再强依赖 speaker running topic；TTS 状态上报可能滞后于 ASR partial。
                tts_active = _is_tts_playing(runtime_window, logger)
                if result.text and _wake_word_hit(result.text):
                    logger.info("★ Wake word detected: %s — BARGE-IN!", result.text)
                    # 1) 视觉信号：前端切 interrupt 状态
                    interrupt_topic = AudioRuntimeTopic(
                        running=True,
                        device_name="interrupt",
                        device_explain="wake_word_barge_in",
                        started_at=time.monotonic(),
                        last_heartbeat=time.monotonic(),
                    )
                    transport.pub_topic(interrupt_topic)
                    # 2) ghost 中断信号：发 interrupt signal 到 ghost 主进程 (通过 Zenoh 跨进程)
                    #    → mindflow.InterruptNucleus → FATAL impulse → shell.clear() → 停 TTS + 停 LLM
                    #    listener 是独立子进程，不能直接访问 ghost 的 Mindflow，
                    #    必须通过 session.add_signal 走 Zenoh 发布。
                    try:
                        from ghoshell_moss.core.mindflow.interrupt_nucleus import new_interrupt_signal
                        sig = new_interrupt_signal(
                            "立刻停下",
                            description="用户喊'立刻停下'，全双工 barge-in",
                        )
                        matrix.session.add_signal(sig)
                        logger.info("★ Interrupt signal sent via zenoh (shell.clear will fire)")
                    except Exception as e:
                        logger.warning("Failed to send interrupt signal: %s", e)
                    # 中断当前 ASR 识别
                    abort_event.set()
                    break

                # 旧的保守模式：TTS 播放时只保留 wake word，其他结果丢弃。
                # Aether 默认关闭这条门控，保证真正全双工。
                if tts_active and os.environ.get("LISTENER_GATE_DURING_TTS") == "1":
                    continue

                # Emit SPEECH_STARTED on first non-empty intermediate result for
                # attention preemption (incomplete impulse with interrupt=True).
                if not result.is_final and result.text and not started_emitted:
                    started_meta = AudioSignal(action=AudioAction.SPEECH_STARTED)
                    sig = Signal(
                        id=utterance_id,
                        name=started_meta.signal_name(),
                        priority=Priority.WARNING,
                        messages=[Message.new().with_content(result.text)],
                        description=f"Speech: {result.text}",
                        metadata=started_meta.model_dump(exclude_defaults=True, exclude_none=True),
                        complete=False,
                    )
                    matrix.session.add_signal(sig)
                    started_emitted = True
                    logger.info("Emitted SPEECH_STARTED signal (utterance=%s)", utterance_id)

                if result.is_final and result.text:
                    # Publish SpeechTopic
                    speech_topic = SpeechTopic(
                        text=result.text,
                        speaker_id="human",
                        speaker_name="User",
                        role="human",
                        timestamp=time.monotonic(),
                    )
                    transport.pub_topic(speech_topic)
                    logger.info("Published SpeechTopic: %s", result.text)

                    # Emit AudioSignal (SPEECH_FINAL) to mindflow
                    audio_meta = AudioSignal(
                        action=AudioAction.SPEECH_FINAL,
                        speech_topic=speech_topic,
                    )
                    sig = Signal(
                        id=utterance_id,
                        name=audio_meta.signal_name(),
                        priority=Priority.WARNING,
                        messages=[Message.new().with_content(result.text)],
                        description=f"Speech: {result.text}",
                        metadata=audio_meta.model_dump(exclude_defaults=True, exclude_none=True),
                        complete=True,
                    )
                    matrix.session.add_signal(sig)
                    logger.info("Emitted SPEECH_FINAL signal (utterance=%s)", utterance_id)
                    utterance_published = True
                    break

            if asr_running_published:
                transport.pub_topic(AudioRuntimeTopic(
                    running=False,
                    device_name="asr",
                    device_explain=_asr_diag_payload(source=asr_source, state="idle"),
                    started_at=time.monotonic(),
                    last_heartbeat=time.monotonic(),
                ))

            # Cooldown: after publishing a speech result, hold ASR briefly until
            # the ghost starts TTS (covers LLM thinking time). With DeepSeek V4
            # Flash (TTFT ~1.1s) + streaming TTS, 2s is enough.
            if utterance_published:
                for _ in range(40):  # up to 2s
                    if _is_tts_playing(runtime_window, logger):
                        logger.info("TTS detected during post-utterance cooldown, holding")
                        break
                    await asyncio.sleep(0.05)

            # NOTE: We intentionally do NOT drain post-utterance here.
            # Any leftover chunks are either:
            #   - ambient noise (ASR VAD will ignore)
            #   - user's next utterance started early (must NOT discard)
            # Pre-call gate above handles TTS residue when TTS is actually playing.

    except asyncio.CancelledError:
        logger.info("Listener app cancelled")
    except Exception:
        logger.exception("Listener app error")
    finally:
        await consumer.close()
        await asr.close()
        logger.info("Listener app stopped")


if __name__ == "__main__":
    Matrix.discover().run(main)
