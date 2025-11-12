import asyncio
import numpy as np
from ghoshell_moss.concepts.speech import AudioFormat
from ghoshell_moss.speech.player.pyaudio_player import PyAudioStreamPlayer


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
            end_time = player.add(sine_wave, audio_type=AudioFormat.PCM_F32LE, channels=0, rate=sample_rate)
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

            end_time = player.add(chunk, audio_type=AudioFormat.PCM_F32LE, channels=1, rate=sample_rate)
            print(f"片段 {i + 1} ({freq}Hz) 预计完成时间: {end_time}")

        # 等待所有播放完成
        await player.wait_play_done()
        print("所有片段播放完成")


if __name__ == "__main__":
    print("=== 测试 PyAudio 播放器 ===")
    asyncio.run(test_pyaudio_player())

    print("\n=== 测试多个片段 ===")
    asyncio.run(test_multiple_chunks())
