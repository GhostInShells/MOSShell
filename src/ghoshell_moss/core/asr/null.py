"""NullASR — 降级耳朵: 空转消费音频, 不产任何 recognition event.

对称 NullSpeech (说侧). ASR config 无效 (缺 api_key) 时 AudioASRProvider 返回它,
listener 照常装线 (capture 跑、waveform topic 正常), 但永远听不到识别结果.
"""
from typing import AsyncIterable, Awaitable, Callable

from ghoshell_moss.contracts.asr import (
    ASR,
    ASRInfo,
    RecognitionEvent,
    RecognitionSegment,
    RecognitionStream,
)
from ghoshell_moss.contracts.audio import AudioChunk

__all__ = ["NullASR"]


class _NullRecognitionStream(RecognitionStream):
    """空转消费音频、不产 event 的识别流."""

    def __init__(self, audio_chunks: AsyncIterable[AudioChunk]) -> None:
        self._audio_chunks = audio_chunks
        self._done = False

    @property
    def stream_id(self) -> str:
        return "null"

    def on_segment(self, callback: Callable[[RecognitionSegment], None]) -> None:
        pass

    def on_event_creating(
            self,
            callback: Callable[[RecognitionEvent], Awaitable[None] | None],
    ) -> None:
        pass

    def commit(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def is_input_done(self) -> bool:
        return self._done

    def __aiter__(self) -> "_NullRecognitionStream":
        return self

    async def __anext__(self) -> RecognitionEvent:
        # 空转消费音频: 抽干输入, 不产任何 event.
        async for _ in self._audio_chunks:
            pass
        self._done = True
        raise StopAsyncIteration


class NullASR(ASR):
    """降级耳朵 — 空转消费音频, 不产 event (对称 NullSpeech)."""

    def get_info(self) -> ASRInfo:
        return ASRInfo()

    def configure(self, params: dict) -> None:
        pass

    def on_error(self, callback: Callable[[Exception], None]) -> None:
        pass

    def recognize(
            self,
            audio_chunks: AsyncIterable[AudioChunk],
            *,
            stream_id: str | None = None,
            gate_factory=None,
    ) -> RecognitionStream:
        return _NullRecognitionStream(audio_chunks)

    async def close(self) -> None:
        pass
