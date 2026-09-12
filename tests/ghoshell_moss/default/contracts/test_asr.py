"""ASR 契约 — RecognitionStream / RecognitionEvent / RecognitionSegment 的行为协议.

只测抽象层的外部契约: recognize_once 的默认累积行为 + RecognitionSegment.precise_cut
的切片语义. 不测实现内部, 也不测框架默认值.
"""
import numpy as np
import pytest

from ghoshell_moss.contracts.asr import (
    ASR,
    ASRInfo,
    Clause,
    RecognitionPhase,
    RecognitionEvent,
    RecognitionSegment,
    RecognitionStream,
)


def _clause(text: str, *, stream_id: str = "s1", segment_id: str = "g1") -> RecognitionEvent:
    return RecognitionEvent(
        stream_id=stream_id, segment_id=segment_id,
        phase=RecognitionPhase.CLAUSE, text=text,
        clause=Clause(text=text),
    )


def _tail(text: str, *, stream_id: str = "s1", segment_id: str = "g1") -> RecognitionEvent:
    return RecognitionEvent(
        stream_id=stream_id, segment_id=segment_id,
        phase=RecognitionPhase.TAIL, text=text,
    )


class _MockStream(RecognitionStream):
    def __init__(self, results: list[RecognitionEvent], *, stream_id: str = "s1"):
        self._results = list(results)
        self._id = stream_id

    @property
    def stream_id(self) -> str:
        return self._id

    def on_segment(self, callback) -> None:
        pass

    def commit(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def is_input_done(self) -> bool:
        return True

    def __aiter__(self) -> RecognitionStream:
        return self

    async def __anext__(self) -> RecognitionEvent:
        if not self._results:
            raise StopAsyncIteration
        return self._results.pop(0)


class _MockASR(ASR):
    def __init__(self, results: list[RecognitionEvent]):
        self._results = list(results)

    def get_info(self) -> ASRInfo:
        return ASRInfo()

    def configure(self, params: dict) -> None:
        pass

    def on_error(self, callback) -> None:
        pass

    def recognize(self, audio_chunks, *, stream_id: str | None = None) -> RecognitionStream:
        return _MockStream(self._results, stream_id=stream_id or "s1")

    async def close(self) -> None:
        pass


async def _empty_audio():
    if False:
        yield np.array([], dtype=np.int16)


class TestRecognizeOnce:
    """recognize_once 的默认行为: 尾包返回全文; 无尾包时累积 CLAUSE 文本."""

    @pytest.mark.asyncio
    async def test_returns_tail_text(self):
        asr = _MockASR([_clause("句1"), _clause("句2"), _tail("句1句2")])
        assert await asr.recognize_once(_empty_audio()) == "句1句2"

    @pytest.mark.asyncio
    async def test_accumulates_clauses_when_no_tail(self):
        asr = _MockASR([_clause("句1"), _clause("句2")])
        assert await asr.recognize_once(_empty_audio()) == "句1句2"

    @pytest.mark.asyncio
    async def test_returns_tail_text_when_no_clause(self):
        asr = _MockASR([_tail("你好世界")])
        assert await asr.recognize_once(_empty_audio()) == "你好世界"

    @pytest.mark.asyncio
    async def test_empty_stream_returns_empty(self):
        asr = _MockASR([])
        assert await asr.recognize_once(_empty_audio()) == ""


class TestSegmentPreciseCut:
    """precise_cut 按 start_ms/end_ms 相对 offset_ms 切片, 并 clamp 到 audio 边界."""

    def test_slices_audio_by_timing(self):
        audio = np.zeros(16000, dtype=np.int16)
        seg = RecognitionSegment(
            id="g1", stream_id="s1", text="hello",
            start_ms=100, end_ms=500, sample_rate=16000,
            audio=audio, offset_ms=0,
        )
        assert len(seg.precise_cut()) == 6400  # (500-100)ms × 16kHz

    def test_clamps_to_audio_bounds(self):
        audio = np.zeros(1600, dtype=np.int16)
        seg = RecognitionSegment(
            id="g1", stream_id="s1",
            start_ms=0, end_ms=10000, sample_rate=16000,
            audio=audio, offset_ms=0,
        )
        assert len(seg.precise_cut()) == 1600  # end_ms 超出 audio 长度, clamp 到边界
