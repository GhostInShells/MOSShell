"""TTSSpeechStream.play collects real playback samples into the caller's list.

Regression: play(samples)/speak(samples) must register the sample collector even
when the caller passes an empty list (which is falsy) — guard with
``if samples is not None``, not ``if samples``. Otherwise say/content commands
return None (no audible progress) instead of the played-seconds description.
"""
import asyncio
import logging

import numpy as np
import pytest

from ghoshell_moss.contracts.speech import (
    AudioFormat,
    SpeechClause,
    TTSBatch,
    TTSItem,
    Word,
)
from ghoshell_moss.core.speech.stream_tts_speech import TTSSpeechStream
from ghoshell_moss.core.speech.virtual_player import VirtualStreamPlayer


def _pcm(seconds: float, sample_rate: int = 8000) -> np.ndarray:
    samples = int(seconds * sample_rate)
    return (np.sin(np.linspace(0, 2 * np.pi * 440 * seconds, samples)) * 8000).astype(np.int16)


class _FakeTTSBatch(TTSBatch):
    """Minimal TTS batch that yields one 0.1s sine item carrying its text."""

    def __init__(self, batch_id: str, text: str):
        self._id = batch_id
        self._text = text
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
        pcm = (np.sin(np.linspace(0, 2 * np.pi * 440 * 0.1, 800)) * 8000).astype(np.int16)
        yield TTSItem(text=self._text, audio=pcm, sample_rate=8000,
                      audio_format="s16le", channels=1, tone="", voice={})


@pytest.mark.asyncio
async def test_play_collects_samples_into_empty_list():
    """play(samples) with an initially empty list must still collect samples."""
    player = VirtualStreamPlayer(sample_rate=8000, channels=1)
    await player.start()
    stream = TTSSpeechStream(
        loop=asyncio.get_running_loop(),
        audio_format=AudioFormat.PCM_S16LE,
        channels=1,
        sample_rate=8000,
        player=player,
        tts_batch=_FakeTTSBatch("bid1", ""),
        logger=logging.getLogger("t"),
    )
    samples = []  # empty — falsy, must not skip on_sample registration
    stream.feed("hello world")
    stream.commit()
    await stream.play(samples)

    assert len(samples) == 1
    assert samples[0].text == "hello world"
    assert samples[0].duration == pytest.approx(0.1, abs=0.02)
    await player.close()


class _ClauseTTSBatch(_FakeTTSBatch):
    """Two clauses over two audio chunks, 服务端字幕的时间轴与 chunk 时长刻意错开.

    第一块音频 0.4s, 但其 clause 末尾只标到 0.1s — 播完这块就等于第一句播完;
    第二句标到 0.9s, 远超总音频 0.5s, 所以它永远不会被判为"已播出".
    """

    def __init__(self):
        super().__init__("clause-batch", "")
        self._clause_list: list[SpeechClause] = []

    def clauses(self) -> list[SpeechClause]:
        return list(self._clause_list)

    async def items(self):
        self._clause_list = []
        specs = [("第一句.", 0.4, 0.1), ("第二句.", 0.1, 0.9)]
        for text, audio_seconds, end_time in specs:
            self._clause_list.append(
                SpeechClause(
                    text=text,
                    words=[Word.model_validate({"word": text, "endTime": end_time})],
                )
            )
            yield TTSItem(text="", audio=_pcm(audio_seconds), sample_rate=8000,
                          audio_format="s16le", channels=1, tone="", voice={})


@pytest.mark.asyncio
async def test_played_text_counts_only_clauses_actually_played():
    """played_text 是真实播出的记账: 未播出时为空, 中断时只含播完的 clause."""
    player = VirtualStreamPlayer(sample_rate=8000, channels=1)
    await player.start()
    stream = TTSSpeechStream(
        loop=asyncio.get_running_loop(),
        audio_format=AudioFormat.PCM_S16LE,
        channels=1,
        sample_rate=8000,
        player=player,
        tts_batch=_ClauseTTSBatch(),
        logger=logging.getLogger("t"),
    )
    stream.feed("第一句.第二句.")
    stream.commit()

    # 已喂入、已合成都不算 — 没有真实播出就是空.
    assert stream.played_text() == ""
    assert stream.buffered() == "第一句.第二句."

    play_task = asyncio.create_task(stream.play([]))
    try:
        for _ in range(400):
            if stream.played_text():
                break
            await asyncio.sleep(0.01)
        play_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await play_task
        # 第一句播完, 第二句没有 — 且文本原样 (中文词间无空格).
        assert stream.played_text() == "第一句."
    finally:
        if not play_task.done():
            play_task.cancel()
        await player.close()
