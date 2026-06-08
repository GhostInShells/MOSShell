"""PTT Listener App — push-to-talk audio capture.

Hold a key to record, release to send to ASR.
Much simpler than continuous listening + TTS gating.

Usage:
    moss apps test sensors/ptt_listener
    moss apps start sensors/ptt_listener

Env:
    PTT_KEY — keyboard key name (default: t). Single chars (t, a, space)
              or pynput special keys (media_play_pause, f1, ctrl, etc.)
"""
import asyncio
import math
import os
import time

import numpy as np
from pynput import keyboard
from scipy import signal

from ghoshell_moss.contracts.audio import (
    AudioAction,
    AudioCaptureConfig,
    AudioSignal,
    AudioTransport,
)
from ghoshell_moss.contracts.speech import SpeechTopic
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.host.speech.capture.matrix_audio_transport import MatrixAudioTransport
from ghoshell_moss.host.speech.capture.miniaudio_capture import MiniAudioCaptureSource
from ghoshell_moss.host.speech.volcengine_asr import VolcengineASR, VolcengineASRConfig
from ghoshell_moss.message import Message

_ASR_SAMPLE_RATE = 16000


def _match_key(key, ptt_key_name: str) -> bool:
    """Match pynput key event against configured PTT key.

    Supports single-char keys ('t', 'a') via key.char,
    and special keys ('media_play_pause', 'f1') via keyboard.Key.
    """
    if isinstance(key, keyboard.KeyCode) and key.char is not None:
        return key.char.lower() == ptt_key_name.lower()
    expected = getattr(keyboard.Key, ptt_key_name, None)
    return key == expected


def _resample_audio(samples: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return samples
    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    return signal.resample_poly(samples.astype(np.float32), up, down).astype(np.int16)


async def main(matrix: Matrix) -> None:
    logger = matrix.logger
    if logger is None:
        import logging
        logger = logging.getLogger("moss.ptt_listener")

    ptt_key_name = os.getenv("PTT_KEY", "t")
    logger.info("=" * 50)
    logger.info("  PTT Listener")
    logger.info("=" * 50)
    logger.info("  Key    : %r", ptt_key_name)
    logger.info("  Action : hold to record, release to recognize")
    logger.info("  Stop   : Ctrl+C")
    logger.info("=" * 50)

    transport: AudioTransport = MatrixAudioTransport(matrix=matrix)
    capture_config = AudioCaptureConfig()
    source = MiniAudioCaptureSource(transport=transport, config=capture_config)
    consumer = source.new_sequential_consumer(max_queue_frames=128)
    await consumer.start()

    asr = VolcengineASR(
        config=VolcengineASRConfig(
            sample_rate=_ASR_SAMPLE_RATE,
            end_window_size=1000,
        ),
        logger=logger,
    )

    press_event = asyncio.Event()
    release_event = asyncio.Event()

    def on_press(key) -> None:
        if _match_key(key, ptt_key_name) and not press_event.is_set():
            press_event.set()
            logger.info("[recording...]")

    def on_release(key) -> None:
        if _match_key(key, ptt_key_name) and not release_event.is_set():
            release_event.set()
            logger.info("[recognizing...]")

    kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    kb_listener.start()

    try:
        while True:
            # Wait for press
            await press_event.wait()
            press_event.clear()

            chunks = []
            release_time = None

            # Collect until release + 1.2s tail
            async for chunk in consumer:
                chunks.append(
                    _resample_audio(
                        chunk.samples, capture_config.sample_rate, _ASR_SAMPLE_RATE
                    )
                )

                if release_event.is_set():
                    release_event.clear()
                    release_time = time.monotonic()

                if release_time is not None and time.monotonic() - release_time >= 1.2:
                    break

            # Mis-click guard: total duration < 500ms
            total_ms = len(chunks) * capture_config.frame_duration_ms
            if total_ms < 500:
                logger.info("[too short, ignored]")
                continue

            # Skip first 800ms (lead-in silence / prep time)
            skip = int(800 / capture_config.frame_duration_ms)
            valid = chunks[skip:]
            if not valid:
                logger.info("[no valid audio]")
                continue

            # ASR
            async def _audio_gen():
                for c in valid:
                    yield c

            async for result in asr.recognize(_audio_gen()):
                if result.is_final and result.text:
                    speech_topic = SpeechTopic(
                        text=result.text,
                        speaker_id="human",
                        speaker_name="User",
                        role="human",
                        timestamp=time.monotonic(),
                    )
                    transport.pub_topic(speech_topic)

                    sig = AudioSignal(
                        action=AudioAction.SPEECH_FINAL,
                        speech_topic=speech_topic,
                    ).to_signal(
                        Message.new().with_content(result.text),
                        description=f"PTT: {result.text}",
                    )
                    matrix.session.add_signal(sig)

                    logger.info("-> %s", result.text)
                    break

            logger.info("[done]")

    except asyncio.CancelledError:
        pass
    finally:
        logger.info("[shutting down...]")
        kb_listener.stop()
        await consumer.close()
        await asr.close()
        logger.info("[stopped]")


if __name__ == "__main__":
    Matrix.discover().run(main)
