from typing import Optional

import numpy as np
from ghoshell_common.contracts import LoggerItf

try:
    import pyaudio
except ImportError as e:
    raise ImportError(f"failed to import audio dependencies, please try to install ghoshell-shell[audio]: {e}")

from ghoshell_moss.speech.player.base_player import BaseAudioStreamPlayer

__all__ = ["PyAudioStreamPlayer"]


# author: deepseek v3.1


class PyAudioStreamPlayer(BaseAudioStreamPlayer):
    """
    基础的 AudioStream
    使用单独的线程处理音频输出，通过 asyncio 队列进行通信
    """

    def __init__(
        self,
        *,
        device_index: int = 0,
        sample_rate: int = 44100,
        channels: int = 1,
        logger: LoggerItf | None = None,
        safety_delay: float = 0.1,
    ):
        """
        基于 PyAudio 的异步音频播放器实现
        使用单独的线程处理阻塞的音频输出操作
        """
        self._device_index = device_index
        self._pyaudio_instance: Optional[pyaudio.PyAudio] = None
        self._stream: Optional[pyaudio.Stream] = None
        super().__init__(
            sample_rate=sample_rate,
            channels=channels,
            logger=logger,
            safety_delay=safety_delay,
        )
        self.safety_delay = safety_delay

    def _audio_stream_start(self):
        self._pyaudio_instance = pyaudio.PyAudio()
        self._stream = self._pyaudio_instance.open(
            format=pyaudio.paInt16,
            channels=self.channels,  # 固定为单声道
            rate=self.sample_rate,  # 固定采样率
            output=True,
            frames_per_buffer=1024,
        )

    def _audio_stream_stop(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pyaudio_instance:
            self._pyaudio_instance.terminate()

    def _audio_stream_write(self, data: np.ndarray):
        if self._stream:
            try:
                self._stream.write(data.tobytes())
            except Exception:
                self.logger.exception("Write audio stream failed")
