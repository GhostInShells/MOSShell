"""Aether VPIO audio capture app — macOS system-level AEC.

Opens AVAudioEngine with setVoiceProcessingEnabled(True) on both input + output
nodes → system-level AEC subtracts TTS playback from the mic signal → ASR only
hears the user's voice. Publishes 16kHz / mono / int16 PCM to Zenoh stream
(audio/pcm) using the same pack_chunk() wire format as MiniAudioCaptureSource,
so listener / waveform apps consume unchanged.

Usage:
    moss apps test aether/vpio_capture
    moss apps start aether/vpio_capture

Replaces sensors/audio_capture when running on macOS for full-duplex conversation.
"""
import asyncio
import logging
import sys
from pathlib import Path

from ghoshell_moss.contracts.audio import AudioCaptureConfig
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.host.speech.capture.matrix_audio_transport import MatrixAudioTransport

# Local import — the VPIOCaptureSource lives next to main.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vpio_capture import VPIOCaptureSource  # noqa: E402


async def main(matrix: Matrix) -> None:
    logger = matrix.logger or logging.getLogger("moss.vpio_capture")
    logger.info("VPIO audio capture app starting (macOS system-level AEC)")

    transport = MatrixAudioTransport(matrix=matrix)
    # AudioCaptureConfig defaults (sample_rate=44100 etc.) are ignored —
    # VPIOCaptureSource forces native 48k internally and outputs 16k for ASR.
    config = AudioCaptureConfig()
    capture = VPIOCaptureSource(transport=transport, config=config)

    try:
        await capture.start()
        logger.info("VPIO audio capture app started (device: %s)", capture.device_explain())
    except RuntimeError as e:
        # Non-macOS or pyobjc missing — let app die cleanly, listener will fall back
        logger.error("VPIO start failed: %s", e)
        raise

    stop_event = asyncio.Event()

    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        logger.info("VPIO audio capture app cancelled, shutting down")
    except KeyboardInterrupt:
        pass
    finally:
        await capture.close()
        logger.info("VPIO audio capture app stopped")


if __name__ == "__main__":
    Matrix.discover().run(main)
