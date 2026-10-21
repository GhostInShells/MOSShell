"""BaseTTSSpeech.clear() 停嘴契约测试.

Speech.clear() 的契约是"清空所有输出中的 output" — 对一个正在播放的 TTS speech,
这意味着立刻停止播放, 而不是只清账本. 回归: 曾有一版 clear() 只 copy+clear
``_outputted`` 账本, 不碰 player, 导致任何不经 cancel 的 clear 路径都漏嘴.
"""
import asyncio

import numpy as np
import pytest

from ghoshell_moss.contracts.speech import (
    TTS,
    TTSBatch,
    TTSInfo,
    TTSItem,
)
from ghoshell_moss.core.speech.stream_tts_speech import BaseTTSSpeech
from ghoshell_moss.core.speech.virtual_player import VirtualStreamPlayer


def _pcm(seconds: float, sample_rate: int = 8000) -> np.ndarray:
    samples = int(seconds * sample_rate)
    return (np.sin(np.linspace(0, 2 * np.pi * 440 * seconds, samples)) * 8000).astype(np.int16)


class _LongTTSBatch(TTSBatch):
    """产出 20 段 0.5s 音频 (共 10s), 让播放持续足够久以观测 clear 前后的停嘴."""

    def __init__(self, batch_id: str):
        self._id = batch_id
        self._text = ""
        self._started = False

    def batch_id(self) -> str:
        return self._id

    def with_callback(self, callback) -> None:
        pass

    def feed(self, text: str) -> None:
        self._text += text

    def commit(self) -> None:
        pass

    async def start(self) -> None:
        self._started = True

    async def close(self) -> None:
        pass

    def is_committed(self) -> bool:
        return True

    def is_closed(self) -> bool:
        return False

    def is_started(self) -> bool:
        return self._started

    async def wait_done(self) -> None:
        pass

    async def items(self):
        pcm = _pcm(0.5)
        for _ in range(20):
            yield TTSItem(text=self._text, audio=pcm, sample_rate=8000,
                          audio_format="s16le", channels=1, tone="", voice={})


class _LongTTS(TTS):
    def __init__(self):
        self._info = TTSInfo(sample_rate=8000, channels=1)

    def new_batch(self, batch_id: str = "", *, callback=None, tone=None, voice=None) -> TTSBatch:
        return _LongTTSBatch(batch_id or "long-batch")

    async def clear(self) -> None:
        pass

    def get_info(self) -> TTSInfo:
        return self._info

    def use_tone(self, config_key: str) -> None:
        pass

    def current_tone(self) -> str:
        return ""

    def set_voice(self, config: dict) -> None:
        pass

    def get_voice(self) -> dict:
        return {}

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_clear_stops_playback():
    player = VirtualStreamPlayer(sample_rate=8000, channels=1)
    speech = BaseTTSSpeech(player=player, tts=_LongTTS())
    await speech.start()

    stream = speech.new_segment()
    stream.feed("这句话会播很久")
    stream.commit()

    play_task = asyncio.create_task(stream.play([]))

    # 播放真正开始 (有音频入队, is_playing 变 True).
    for _ in range(500):
        if player.is_playing():
            break
        await asyncio.sleep(0.01)
    assert player.is_playing() is True, "播放没有开始, 测试前提不成立"

    # 播放进行中 clear, 应立刻停嘴.
    await speech.clear()
    assert player.is_playing() is False, "clear 后 player 仍在播放"

    # stream 已被 close, 播放任务应提前结束而非等 10s 播完.
    await asyncio.wait_for(play_task, timeout=2.0)

    await speech.close()


class _HangingTTSBatch(_LongTTSBatch):
    """wait_done 永不返回 —— 模拟一个再也不回来的 TTS 会话.

    D21 现场: say 命令卡在 wait_played 的 ``_tts_batch.wait_done()`` 上, 而 clear() 置的是
    stream 自己的 closed / play_done 事件, 两个不相干 —— 于是 clear 救不回来, 整条 channel
    队列跟着停摆, 只能重启.
    """

    async def wait_done(self) -> None:
        await asyncio.Event().wait()


class _HangingTTS(_LongTTS):
    def new_batch(self, batch_id: str = "", *, callback=None, tone=None, voice=None) -> TTSBatch:
        return _HangingTTSBatch(batch_id or "hanging-batch")


@pytest.mark.asyncio
async def test_clear_frees_a_say_stuck_on_the_tts_batch():
    """clear() 必须能解掉一个卡在等 batch 收尾的 say, 否则它会把命令队列一起带走."""
    player = VirtualStreamPlayer(sample_rate=8000, channels=1)
    speech = BaseTTSSpeech(player=player, tts=_HangingTTS())
    await speech.start()

    stream = speech.new_segment()
    stream.feed("这句话永远合不完")
    stream.commit()

    play_task = asyncio.create_task(stream.play([]))
    await asyncio.sleep(0.05)
    assert not play_task.done(), "测试前提: batch 不收尾, play 应该还在等"

    await speech.clear()
    await asyncio.wait_for(play_task, timeout=2.0)
    assert stream.is_closed() is True

    await speech.close()


@pytest.mark.asyncio
async def test_clear_closes_a_stream_that_never_started_synthesis():
    """没开始合成的流也必须在 clear 里真关掉, 不能只从账本里摘掉."""
    player = VirtualStreamPlayer(sample_rate=8000, channels=1)
    speech = BaseTTSSpeech(player=player, tts=_LongTTS())
    await speech.start()

    stream = speech.new_segment()
    stream.feed("只准备, 还没开始")
    assert stream.is_closed() is False

    await speech.clear()
    assert stream.is_closed() is True, "未合成的流漏关了: batch 没关, is_closed() 永远是 False"
    assert speech.streams() == []

    await speech.close()
