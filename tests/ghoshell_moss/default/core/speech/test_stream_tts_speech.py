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
