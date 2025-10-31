import asyncio
import numpy as np
from typing import Optional
from ghoshell_moss.depends import check_pyaudio
from ghoshell_moss.concepts.speech import AudioType
from ghoshell_common.contracts import LoggerItf

if check_pyaudio():
    import pyaudio

from ghoshell_moss.speech.player.base_player import BaseAudioStreamPlayer

__all__ = ['PyAudioStreamPlayer']


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
            sample_rate: int = 16000,
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
            frames_per_buffer=1024
        )

    def _audio_stream_stop(self):
        if self._stream:
            self._stream.stop_stream()
            self._stream.close()
        if self._pyaudio_instance:
            self._pyaudio_instance.terminate()

    def _audio_stream_write(self, data: bytes):
        if self._stream:
            try:
                self._stream.write(data)
            except Exception as e:
                self.logger.exception(e)


# 测试代码
async def test_pyaudio_player():
    """测试 PyAudio 播放器"""
    player = PyAudioStreamPlayer(safety_delay=0.1)

    try:
        async with player:
            # 生成测试音频：1秒的440Hz正弦波
            sample_rate = 44100
            duration = 1.0
            t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
            sine_wave = 0.3 * np.sin(2 * np.pi * 440 * t)  # 减小音量避免爆音

            # 添加音频片段
            end_time = await player.add(sine_wave, audio_type=AudioType.PCM_F32LE, channel=0, rate=sample_rate)
            print(f"预计完成时间: {end_time}")

            # 等待播放完成
            await player.wait_play_done()

            # 检查状态
            print(f"是否仍在播放: {player.is_playing()}")
            print(f"是否已关闭: {player.is_closed()}")

    except Exception as e:
        print(f"测试失败: {e}")


async def test_multiple_chunks():
    """测试连续添加多个片段"""
    player = PyAudioStreamPlayer(safety_delay=0.1)

    async with player:
        sample_rate = 44100

        # 添加3个短片段
        for i in range(3):
            duration = 0.3
            samples = int(sample_rate * duration)
            # 生成不同频率的正弦波
            freq = 440 + (i * 100)
            t = np.linspace(0, duration, samples, endpoint=False)
            chunk = 0.2 * np.sin(2 * np.pi * freq * t)  # 减小音量

            end_time = await player.add(chunk, audio_type=AudioType.PCM_F32LE, channel=1, rate=sample_rate)
            print(f"片段 {i + 1} ({freq}Hz) 预计完成时间: {end_time}")

        # 等待所有播放完成
        await player.wait_play_done()
        print("所有片段播放完成")


if __name__ == "__main__":
    print("=== 测试 PyAudio 播放器 ===")
    asyncio.run(test_pyaudio_player())

    print("\n=== 测试多个片段 ===")
    asyncio.run(test_multiple_chunks())
