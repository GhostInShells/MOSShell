import asyncio

import numpy as np
import pytest

from ghoshell_moss.contracts.speech import AudioFormat
from ghoshell_moss.host.speech.player import VirtualStreamPlayer


def _make_sine(duration: float, sample_rate: int, freq: float = 440.0, amplitude: float = 0.3) -> np.ndarray:
    """生成一段正弦波 int16 音频数据."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * (32767 * amplitude)).astype(np.int16)


@pytest.mark.asyncio
async def test_basic_lifecycle():
    """start → close 基本生命周期."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    assert not player.is_closed()

    await player.start()
    assert player._thread is not None

    await player.close()
    assert player.is_closed()


@pytest.mark.asyncio
async def test_add_and_wait():
    """add 音频片段后 wait_play_done 正常返回."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    await player.start()

    audio = _make_sine(0.05, 44100)
    player.add(audio, audio_type=AudioFormat.PCM_S16LE, rate=44100)

    done = await player.wait_play_done(timeout=2.0)
    assert done

    await player.close()


@pytest.mark.asyncio
async def test_add_pcm_f32le():
    """PCM_F32LE 格式自动转为 int16."""
    player = VirtualStreamPlayer(sample_rate=16000, channels=1)
    await player.start()

    t = np.linspace(0, 0.05, int(16000 * 0.05), endpoint=False)
    audio_f32 = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float32)
    player.add(audio_f32, audio_type=AudioFormat.PCM_F32LE, rate=16000)

    await player.wait_play_done(timeout=2.0)
    await player.close()


@pytest.mark.asyncio
async def test_add_resample():
    """不同采样率自动重采样."""
    player = VirtualStreamPlayer(sample_rate=16000, channels=1)
    await player.start()

    audio_24k = _make_sine(0.05, 24000)
    player.add(audio_24k, audio_type=AudioFormat.PCM_S16LE, rate=24000)

    await player.wait_play_done(timeout=2.0)
    await player.close()


@pytest.mark.asyncio
async def test_is_playing():
    """add 后 is_playing() 返回 True，播放完毕后返回 False."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    await player.start()

    assert not player.is_playing()

    audio = _make_sine(0.1, 44100)
    player.add(audio, audio_type=AudioFormat.PCM_S16LE, rate=44100)
    assert player.is_playing()

    await player.wait_play_done(timeout=2.0)
    assert not player.is_playing()

    await player.close()


@pytest.mark.asyncio
async def test_clear():
    """clear 清空播放队列并立即停止."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    await player.start()

    audio = _make_sine(0.2, 44100)
    player.add(audio, audio_type=AudioFormat.PCM_S16LE, rate=44100)

    await player.clear()
    assert not player.is_playing()

    await player.close()


@pytest.mark.asyncio
async def test_multiple_add_streaming():
    """多次 add 模拟流式播放."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    await player.start()

    for _ in range(10):
        chunk = _make_sine(0.01, 44100)
        player.add(chunk, audio_type=AudioFormat.PCM_S16LE, rate=44100)

    await player.wait_play_done(timeout=3.0)
    await player.close()


@pytest.mark.asyncio
async def test_estimated_end_time_monotonic():
    """estimated_end_time 随 add 单调递增."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    await player.start()

    prev_time = 0.0
    for _ in range(5):
        audio = _make_sine(0.02, 44100)
        end_time = player.add(audio, audio_type=AudioFormat.PCM_S16LE, rate=44100)
        assert end_time > prev_time
        prev_time = end_time

    await player.wait_play_done(timeout=2.0)
    await player.close()


@pytest.mark.asyncio
async def test_double_start_idempotent():
    """重复 start 不创建第二个线程."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    await player.start()

    thread_id_1 = player._thread.ident
    await player.start()
    thread_id_2 = player._thread.ident

    assert thread_id_1 == thread_id_2

    await player.close()


@pytest.mark.asyncio
async def test_add_after_close_is_noop():
    """close 后 add 不抛异常."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    await player.start()
    await player.close()

    audio = _make_sine(0.05, 44100)
    end_time = player.add(audio, audio_type=AudioFormat.PCM_S16LE, rate=44100)
    assert end_time > 0


@pytest.mark.asyncio
async def test_clear_interrupts_playback():
    """clear 后 is_playing 立即返回 False，wait_play_done 不阻塞."""
    player = VirtualStreamPlayer(sample_rate=44100, channels=1)
    await player.start()

    # 添加一段长音频
    audio = _make_sine(2.0, 44100)
    player.add(audio, audio_type=AudioFormat.PCM_S16LE, rate=44100)
    assert player.is_playing()

    # clear 立即中断
    await player.clear()
    assert not player.is_playing()

    # wait_play_done 应立即返回（无需等 2 秒）
    done = await player.wait_play_done(timeout=0.5)
    assert done

    await player.close()


def test_add_zero_rate_raises():
    """rate <= 0 应抛出 ValueError."""
    player = VirtualStreamPlayer(sample_rate=16000, channels=1)
    audio = np.zeros(0, dtype=np.int16)
    with pytest.raises(ValueError, match="rate must be greater than 0"):
        player.add(audio, audio_type=AudioFormat.PCM_S16LE, rate=0)


def test_add_zero_duration_returns_float():
    """空音频片段 (duration==0) 应有返回值，不返回 None."""
    player = VirtualStreamPlayer(sample_rate=16000, channels=1)
    audio = np.zeros(0, dtype=np.int16)
    result = player.add(audio, audio_type=AudioFormat.PCM_S16LE, rate=16000)
    assert isinstance(result, float)
    assert result >= 0.0
