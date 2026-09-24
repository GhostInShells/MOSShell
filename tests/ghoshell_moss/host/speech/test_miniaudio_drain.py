"""MiniAudioStreamPlayer 在自己的 wait_play_done 里补设备排空 (无需音频硬件).

基类的 wait_play_done 只等 worker 输入队列空; miniaudio 的 ``_audio_stream_write`` 只是
把数据交给设备, 设备回调要一个周期后才真的播出, 所以基类事件比"设备播完"早至少一块.
本实现覆写契约方法 wait_play_done: 返回即代表最后一帧真实音频已经播出.

测试直接驱动 generator 与播放头, 不启真实设备 — 验的是排空判据, 不是具体数值.
"""
import asyncio
import time

import numpy as np
import pytest

from ghoshell_moss.host.speech.player.miniaudio_player import MiniAudioStreamPlayer


class _FakeDevice:
    """占位设备 — 排空判据只检查设备是否存在."""


def _driver_player() -> MiniAudioStreamPlayer:
    player = MiniAudioStreamPlayer(sample_rate=16000, channels=1, safety_delay=0.0)
    player._start_playback = lambda: None  # 不启真实设备
    player._audio_stream_start()  # 初始化缓冲与播放头计数
    player._playback = _FakeDevice()
    return player


def test_generator_tracks_playhead_and_last_real_frame():
    """交付真实音频 + 补静音时, 播放头与真实音频终点被正确记账."""
    player = _driver_player()
    gen = player._make_generator()
    next(gen)  # prime

    player._audio_stream_write(np.zeros(1600, dtype=np.int16))  # 0.1s 真实音频
    out = gen.send(3200)  # 设备请求 0.2s 的块

    # 真实音频在块头, 不足的补零 — 回调开始时上一块刚播完, 故播放头还是 0.
    assert len(out) == 3200 * 2
    assert player._playhead_frames == 0
    assert player._last_real_frame_pos == 1600
    assert player._data_queue.empty()
    assert player._buf == b""


def test_is_playing_stays_true_until_device_plays_out():
    """队列空但设备还在放尾音时, is_playing 不能报停."""
    player = _driver_player()
    gen = player._make_generator()
    next(gen)
    player._audio_stream_write(np.zeros(1600, dtype=np.int16))
    gen.send(3200)  # 交付 1600 真实帧 + 补静音
    player._play_done_event.set()  # 模拟 worker 队列已空 (基类此刻就会置事件)

    assert player.is_playing() is True
    gen.send(3200)  # 下一次回调: 播放头推过真实音频末尾
    assert player.is_playing() is False


@pytest.mark.asyncio
async def test_wait_play_done_waits_for_device_to_play_out():
    """设备还没播完最后一帧真实音频时, wait_play_done 不能返回."""
    player = _driver_player()
    gen = player._make_generator()
    next(gen)
    player._audio_stream_write(np.zeros(1600, dtype=np.int16))
    gen.send(3200)
    player._play_done_event.set()

    waiter = asyncio.create_task(player.wait_play_done(timeout=2.0))
    await asyncio.sleep(0.05)
    assert not waiter.done(), "设备还没播完最后一帧真实音频, 不应返回"

    gen.send(3200)  # 下一次回调: 播放头推过真实音频末尾
    assert await asyncio.wait_for(waiter, timeout=1.0) is True


@pytest.mark.asyncio
async def test_wait_play_done_gives_up_when_device_stalls():
    """设备不再回调推进播放头时, 排空等待必须放弃而不是挂死."""
    player = _driver_player()
    player._DRAIN_STALL_TIMEOUT = 0.1
    gen = player._make_generator()
    next(gen)
    player._audio_stream_write(np.zeros(1600, dtype=np.int16))
    gen.send(3200)  # 交付后设备就不再回调
    player._play_done_event.set()

    started = time.monotonic()
    assert await player.wait_play_done(timeout=2.0) is True
    assert time.monotonic() - started < 1.0
