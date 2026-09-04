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

from ghoshell_moss.contracts.speech import AudioFormat, TTSBatch, TTSItem
from ghoshell_moss.core.speech.stream_tts_speech import TTSSpeechStream
from ghoshell_moss.host.speech.player import VirtualStreamPlayer


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
