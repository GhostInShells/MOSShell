"""Audio capture app — opens miniaudio CaptureDevice, publishes PCM through transport.

Usage:
    moss apps test sensors/audio_capture
    moss apps start sensors/audio_capture
"""
import asyncio
import logging

from ghoshell_moss.contracts.audio import AudioCaptureConfig
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.host.speech.capture.miniaudio_capture import MiniAudioCaptureSource
from ghoshell_moss.host.speech.capture.matrix_audio_transport import MatrixAudioTransport


async def main(matrix: Matrix) -> None:
    logger = matrix.logger or logging.getLogger("moss.audio_capture")
    logger.info("Audio capture app starting")

    transport = MatrixAudioTransport(matrix=matrix)
    config = AudioCaptureConfig()
    capture = MiniAudioCaptureSource(transport=transport, config=config)

    await capture.start()
    logger.info("Audio capture app started (device: %s)", capture.device_explain())

    stop_event = asyncio.Event()

    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        logger.info("Audio capture app cancelled, shutting down")
    except KeyboardInterrupt:
        pass
    finally:
        await capture.close()
        logger.info("Audio capture app stopped")


if __name__ == "__main__":
    Matrix.discover().run(main)
