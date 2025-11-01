from ghoshell_moss.speech.player.pulseaudio_player import PulseAudioStreamPlayer, AudioFormat
import numpy as np
import asyncio


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
            end_time = player.add(sine_wave, audio_type=AudioFormat.PCM_F32LE, channels=1, rate=sample_rate)
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
