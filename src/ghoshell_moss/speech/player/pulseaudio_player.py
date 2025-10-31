import asyncio
import numpy as np
from ghoshell_moss.depends import check_pulseaudio
from ghoshell_moss.concepts.speech import AudioType
from ghoshell_common.contracts import LoggerItf

if check_pulseaudio():
    import pulsectl

from ghoshell_moss.speech.player.base_player import BaseAudioStreamPlayer

__all__ = ['PulseAudioStreamPlayer']


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
            self.logger.info(f"使用默认音频设备: {self._sink_name}")

        # 创建音频流
        stream_props = {
            'media.name': 'MOSShell Audio Stream',
            'application.name': 'MOSShell',
        }

        self.stream = self.pulse.stream_connect_playback(
            device=self._sink_name,
            stream_name='moshell-audio-stream',
            format='s16le',  # PulseAudio 使用字符串格式
            rate=self.sample_rate,
            channels=self.channels,
            properties=stream_props
        )
        self.logger.info("PulseAudio 输出流已创建")

    def _audio_stream_stop(self):
        self.pulse.stream_kill(self.stream)
        self.pulse.close()

    def _audio_stream_write(self, data: bytes):
        # 写入音频数据（阻塞调用）
        self.pulse.stream_write(self.stream, data)


async def test_pulse_audio_player():
    """测试 PulseAudio 播放器"""
    player = PulseAudioStreamPlayer(
        sink_name=None,  # 使用默认设备
        sample_rate=16000,
        channels=1,
        safety_delay=0.1
    )

    try:
        async with player:
            # 生成测试音频：1秒的440Hz正弦波
            sample_rate = 16000
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            sine_wave = 0.3 * np.sin(2 * np.pi * 440 * t)

            # 添加音频片段
            end_time = await player.add(sine_wave, AudioType.PCM_F32LE, 0, sample_rate)
            print(f"预计完成时间: {end_time}")

            # 等待播放完成
            await player.wait_play_done()

            # 检查状态
            print(f"是否仍在播放: {player.is_playing()}")
            print(f"是否已关闭: {player.is_closed()}")

    except Exception as e:
        print(f"测试失败: {e}")


if __name__ == "__main__":
    asyncio.run(test_pulse_audio_player())
