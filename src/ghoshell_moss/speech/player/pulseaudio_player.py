import numpy as np
from ghoshell_common.contracts import LoggerItf

try:
    import pulsectl

except Exception as e:
    raise ImportError(f"failed to import audio dependencies, please try to install ghoshell-shell[audio]: {e}")

from ghoshell_moss.speech.player.base_player import BaseAudioStreamPlayer

__all__ = ["PulseAudioStreamPlayer"]


class PulseAudioStreamPlayer(BaseAudioStreamPlayer):
    """
    基于 PulseAudio 的异步音频播放器实现
    使用单独的线程处理音频输出，通过队列进行通信
    """

    def __init__(
        self,
        *,
        name: str = "moss-audio-player",
        sink_name: str | None = None,
        sample_rate: int = 16000,
        channels: int = 1,
        safety_delay: float = 0.1,
        logger: LoggerItf | None = None,
    ):
        """
        基于 PulseAudio 的异步音频播放器实现
        使用单独的线程处理阻塞的音频输出操作
        """
        self._client_name = name
        self._sink_name = sink_name
        super().__init__(
            sample_rate=sample_rate,
            logger=logger,
            channels=channels,
            safety_delay=safety_delay,
        )

    def _audio_stream_start(self):
        # 创建 PulseAudio 连接
        self.pulse = pulsectl.Pulse(self._client_name)
        # 获取接收器（如果没有指定，使用默认接收器）
        if self._sink_name is None:
            self._sink_name = self.pulse.server_info().default_sink_name
            self.logger.info("使用默认音频设备: %s", self._sink_name)

        # 创建音频流
        stream_props = {
            "media.name": "MOSShell Audio Stream",
            "application.name": "MOSShell",
        }

        self.stream = self.pulse.stream_connect_playback(
            device=self._sink_name,
            stream_name="moshell-audio-stream",
            format="s16le",  # PulseAudio 使用字符串格式
            rate=self.sample_rate,
            channels=self.channels,
            properties=stream_props,
        )
        self.logger.info("PulseAudio 输出流已创建")

    def _audio_stream_stop(self):
        self.pulse.stream_kill(self.stream)
        self.pulse.close()

    def _audio_stream_write(self, data: np.ndarray):
        # 写入音频数据（阻塞调用）
        self.pulse.stream_write(self.stream, data.tobytes())
