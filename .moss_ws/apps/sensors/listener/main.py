"""Listener App — ASR consumer.

Consumes PCM stream from audio_capture, feeds to Volcengine ASR,
publishes SpeechTopic on final recognition, emits AudioSignal to mindflow.

Usage:
    moss apps test sensors/listener
    moss apps start sensors/listener
"""
import asyncio
import logging
import math
import os
import time
from collections.abc import AsyncIterable

import dotenv
import numpy as np

dotenv.load_dotenv()
from scipy import signal

from ghoshell_moss.contracts.asr import ASRResult
from ghoshell_moss.contracts.audio import (
    AudioCaptureConfig,
    AudioChunk,
)
from ghoshell_moss.core.mindflow.audio_signal import AudioAction, AudioSignal
from ghoshell_moss.host.listener.capture.audio_transport import AudioTransport
from ghoshell_moss.topics.audio import AudioRuntimeTopic, SpeechTopic
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.host.listener.capture.matrix_audio_transport import MatrixAudioTransport
from ghoshell_moss.host.listener.capture.miniaudio_capture import MiniAudioCaptureSource
from ghoshell_moss.host.listener.volcengine_asr import VolcengineASR, VolcengineASRConfig
from ghoshell_moss.message import Message
from ghoshell_moss.core.blueprint.mindflow import Signal, Priority, unique_id

# ASR 期望的采样率 (16kHz 是语音识别的行业标准)
_ASR_SAMPLE_RATE = 16000


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
    abort_event: asyncio.Event,
    logger: logging.Logger,
) -> AsyncIterable[np.ndarray]:
    """Yield resampled np.ndarray samples from AudioSequentialConsumer.

    Uses an internal asyncio.Queue buffer so that ``aclose()`` (called when
    ``asr.recognize()`` finishes) does NOT reach ``consumer.__anext__()``
    and silently drop a chunk.

    Monitors TTS playback state in real time. If TTS starts speaking mid-feed,
    sets ``abort_event`` and stops yielding so ASR receives an early EOF.
    """
    # Unbounded queue: pump must never block on put(), otherwise cancellation
    # can land inside put() and the None sentinel never reaches the reader.
    buffer: asyncio.Queue[AudioChunk | None] = asyncio.Queue()

    async def _pump() -> None:
        """Read from consumer into buffer. Stops on TTS or cancellation."""
        try:
            async for chunk in consumer:
                if _is_tts_playing(runtime_window, logger):
                    logger.info("TTS started during pump, aborting")
                    abort_event.set()
                    break
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
            chunk = await buffer.get()
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
    timeout = None  # No timeout on first iteration (wait for speech)
    last_result: ASRResult | None = None
    try:
        while True:
            try:
                if timeout is not None:
                    result = await asyncio.wait_for(agen.__anext__(), timeout=timeout)
                else:
                    result = await agen.__anext__()
                if result.text:
                    last_result = result
                    timeout = patience
                yield result
            except asyncio.TimeoutError:
                logger.info("ASR silence timeout after %.1fs, finalizing", patience)
                if last_result is not None and not last_result.is_final:
                    logger.info(
                        "Server never sent is_final=True — synthesizing from last partial: %s",
                        last_result.text,
                    )
                    yield ASRResult(text=last_result.text, is_final=True)
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

    环境变量 ``LISTENER_DISABLE_TTS_GATE=1`` 可关闭此门控，
    用于需要 ASR 与 TTS 同时工作的调试或特殊场景。
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
    source = MiniAudioCaptureSource(transport=transport, config=capture_config)
    consumer = source.new_sequential_consumer(max_queue_frames=128)
    await consumer.start()
    logger.info("Audio sequential consumer started")

    # -- Subscribe to AudioRuntimeTopic for TTS gating --
    runtime_window = transport.topic_window(AudioRuntimeTopic, max_size=10)
    logger.info("Subscribed to AudioRuntimeTopic window for TTS gating")

    # -- ASR (16kHz 是语音识别的标准采样率; 如果 capture 不是 16kHz 则重采样) --
    # end_window_size: 静音判停阈值。默认 500ms 对对话场景太短，用户稍微换气就被切断。
    # 1500ms 允许正常句子间停顿，同时不会让用户等太久。
    asr_config = VolcengineASRConfig(
        sample_rate=_ASR_SAMPLE_RATE,
        end_window_size=500,
    )
    asr = VolcengineASR(config=asr_config, logger=logger)

    # -- main recognition loop --
    try:
        while True:
            logger.info("Waiting for speech...")

            # Pre-call gate: don't start ASR while TTS is playing.
            # Drain residual audio while waiting so TTS echoes don't pile up.
            # Limit drain to avoid clearing user's new speech.
            while _is_tts_playing(runtime_window, logger):
                logger.debug("TTS is playing, holding ASR...")
                drained = await _drain_consumer(consumer, timeout=0.05, max_chunks=3)
                if drained:
                    logger.debug("Drained %d residual chunk(s) while TTS active", drained)
                await asyncio.sleep(0.05)

            # Fresh abort flag and utterance id for this utterance.
            abort_event = asyncio.Event()
            utterance_id = unique_id()
            started_emitted = False

            utterance_published = False

            # Each recognize call handles one utterance.
            # The ASR backend (end_window_size) splits on silence.
            audio_gen = _audio_generator(
                consumer,
                capture_config.sample_rate,
                _ASR_SAMPLE_RATE,
                runtime_window,
                abort_event,
                logger,
            )
            async for result in _iter_with_silence_timeout(asr.recognize(audio_gen), logger):
                if result.text:
                    logger.info("ASR partial: %s (final=%s)", result.text, result.is_final)

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
                    # If TTS started mid-feed, the generator aborted early.
                    # The ASR may still return a partial result — drop it.
                    if abort_event.is_set():
                        logger.info(
                            "Gated ASR result — TTS interfered during feed, dropping: %s",
                            result.text,
                        )
                        break

                    # Post-call gate: TTS may have started *after* generator finished
                    # but before result arrived (narrow race).
                    if _is_tts_playing(runtime_window, logger):
                        logger.info(
                            "Gated ASR result — TTS started during recognition, dropping: %s",
                            result.text,
                        )
                        break

                    # 1. Publish SpeechTopic
                    speech_topic = SpeechTopic(
                        text=result.text,
                        speaker_id="human",
                        speaker_name="User",
                        role="human",
                        timestamp=time.monotonic(),
                    )
                    transport.pub_topic(speech_topic)
                    logger.info("Published SpeechTopic: %s", result.text)

                    # 2. Emit AudioSignal (SPEECH_FINAL) to mindflow
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

            # Cooldown: after publishing a speech result, wait briefly for the
            # ghost to start TTS so the pre-call gate can block the next ASR
            # session. Without this, ASR starts before the ghost begins
            # speaking and captures the ghost's own voice for several seconds.
            if utterance_published:
                for _ in range(10):  # up to 500ms
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
