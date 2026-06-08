import numpy as np
import pytest

from ghoshell_moss.contracts.speech import ASR, ASRResult


class MockASR(ASR):
    """A mock ASR for testing the ABC default methods."""

    def __init__(self, results: list[ASRResult]):
        self._results = results
        self._closed = False

    async def recognize(self, audio_chunks):
        for r in self._results:
            yield r

    async def close(self):
        self._closed = True


class TestASRResult:
    def test_is_final_default(self):
        r = ASRResult(text="hello")
        assert r.is_final is False

    def test_is_final_true(self):
        r = ASRResult(text="hello", is_final=True)
        assert r.is_final is True


class TestASRABC:
    @pytest.mark.asyncio
    async def test_recognize_once_returns_final(self):
        asr = MockASR([
            ASRResult(text="你", is_final=False),
            ASRResult(text="你好", is_final=False),
            ASRResult(text="你好世界", is_final=True),
        ])
        text = await asr.recognize_once(_empty_audio())
        assert text == "你好世界"

    @pytest.mark.asyncio
    async def test_recognize_once_empty(self):
        asr = MockASR([])
        text = await asr.recognize_once(_empty_audio())
        assert text == ""

    @pytest.mark.asyncio
    async def test_recognize_once_no_final(self):
        asr = MockASR([ASRResult(text="partial", is_final=False)])
        text = await asr.recognize_once(_empty_audio())
        assert text == ""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        asr = MockASR([])
        async with asr:
            pass
        assert asr._closed


async def _empty_audio():
    """Empty async generator for audio chunks."""
    if False:
        yield np.array([], dtype=np.int16)
