"""
ASR contracts — audio speech recognition abstractions.

ASR is the ear: raw audio -> text stream. Kept separate from speech (the mouth)
to avoid cross-infection in the contract layer.

Self-describing surface (mirrors TTSInfo): get_info() exposes the audio input
contract (sample_rate/bits/channel) plus the JSON schema and current values of the
tunable behavior params; configure() sets those params for the next recognize().
Model identity is fixed at creation by the factory/provider — it is not part of the
runtime self-describing surface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterable, Callable

import numpy as np
from pydantic import BaseModel, Field

__all__ = [
    "ASR",
    "ASRInfo",
    "RecognitionStream",
    "RecognitionPhase",
    "RecognitionResult",
    "RecognitionSegment",
]


class RecognitionPhase(str, Enum):
    """Three result phases of a recognition stream — aligned with volcengine response semantics.

    - PARTIAL: intermediate result (utterance ``definite=false``), emitted while speaking.
    - CLAUSE:   stable sentence (utterance ``definite=true``).
    - TAIL:     segment tail (frame-level ``is_last_package=true``) — marks the end of a
                segment (one turn) and triggers a segment cut.

    ``definite`` only marks "this sentence is stable", not the end of the stream. A
    stream holds many segments, each ending with its own TAIL. Stream end is a
    separate fact (the audio input is exhausted), not a TAIL.
    """

    PARTIAL = "partial"
    CLAUSE = "clause"
    TAIL = "tail"


@dataclass
class RecognitionResult:
    """One result produced at a phase of the recognition stream (text axis).

    ``text`` is the full accumulated text — consistent across phases, always
    updating. ``clause_text`` carries the stable clause text (only meaningful for
    CLAUSE phase). ``segment_id`` links to the RecognitionSegment of the same
    segment (its ``id``).
    """

    stream_id: str
    segment_id: str  # segment id
    phase: RecognitionPhase
    text: str
    start_ms: int = 0
    end_ms: int = 0
    error: str = ""
    clause_text: str = ""

    @property
    def last_clause_text(self) -> str:
        """最后一句分句的文本; 无分句时回退到全文."""
        return self.clause_text or self.text


@dataclass
class RecognitionSegment:
    """Audio archive of one segment (one turn) — the audio axis.

    Cut once per tail — ``text`` is the accumulated text of the whole segment,
    ``audio`` is the accumulated audio. ``start_ms``/``end_ms`` are stream-relative
    timestamps of the segment; ``offset_ms`` is the stream-relative start of
    ``audio``, used by ``precise_cut``.

    Linked to RecognitionResult via ``segment_id`` (its ``id``) + ``stream_id``,
    delivered through a separate callback (``on_segment``), not mixed into the text axis.
    """

    id: str  # segment id
    stream_id: str  # stream id
    text: str = ""
    start_ms: int = 0
    end_ms: int = 0
    sample_rate: int = 16000
    bits: int = 16
    channel: int = 1
    audio: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int16))
    offset_ms: int = 0

    def precise_cut(self) -> np.ndarray:
        """Slice the coarse-cut audio precisely by start_ms/end_ms."""
        sr = self.sample_rate or 1
        start = int((self.start_ms - self.offset_ms) * sr / 1000)
        end = int((self.end_ms - self.offset_ms) * sr / 1000)
        start = max(0, min(start, len(self.audio)))
        end = max(start, min(end, len(self.audio)))
        return self.audio[start:end]


class ASRInfo(BaseModel):
    """Runtime self-describing info — mirrors TTSInfo.

    The model reads get_info() first: the audio input contract (sample_rate/bits/channel)
    plus the JSON schema and current values of the tunable behavior params. Then
    configure() turns the behavior knobs for the next recognize(). Each implementation
    exposes its own params BaseModel; the contract only carries the schema and current
    values as dicts.
    """

    sample_rate: int = Field(default=16000, description="sample rate the ASR expects")
    bits: int = Field(default=16, description="bit depth")
    channel: int = Field(default=1, description="channel count")

    params_schema: dict = Field(default_factory=dict,
                                description="json schema of tunable behavior params (each implementation exposes its own BaseModel)")
    params: dict = Field(default_factory=dict, description="current behavior param values")


class RecognitionStream(ABC):
    """Continuous recognition loop — one audio stream -> a sequence of RecognitionResult.

    1 stream = ( n segment = ( m result ) )

    Each segment (one turn) ends with a TAIL, which triggers a segment cut. The stream
    keeps running across segments until the audio input is exhausted; at that point a
    final commit cuts the last segment and the loop ends naturally.

    Single entry only (``__aiter__`` is not re-entrant). ``is_input_done()`` reports
    whether the audio input has stopped.
    """

    @property
    @abstractmethod
    def stream_id(self) -> str:
        ...

    @abstractmethod
    def on_segment(self, callback: Callable[[RecognitionSegment], None]) -> None:
        """Register a callback invoked at each tail cut with the audio-axis result (RecognitionSegment)."""

    @abstractmethod
    def commit(self) -> None:
        """Notify the cloud to produce a tail now — marks the end of the current segment. Does not close the stream."""

    @abstractmethod
    def is_input_done(self) -> bool:
        """Whether the audio input loop has stopped."""

    @abstractmethod
    def __aiter__(self) -> "RecognitionStream":
        """Start sending audio; runs until the audio stream ends. Single entry only."""

    @abstractmethod
    async def __anext__(self) -> RecognitionResult:
        ...


class ASR(ABC):
    """Audio perception organ — the ear. Symmetric to TTS (the mouth).

    Input: a 1-D int16 PCM audio stream (the caller resamples to match ASRInfo).
    Output: a continuous recognition loop (RecognitionStream) of partial/clause/tail results.
    """

    @abstractmethod
    def get_info(self) -> ASRInfo:
        """Return runtime self-describing info — audio contract + schema/values of tunable params."""

    @abstractmethod
    def configure(self, params: dict) -> None:
        """Set behavior params for the next recognize(). Validation/domain lives in each impl's params BaseModel."""

    @abstractmethod
    def on_error(self, callback: Callable[[Exception], None]) -> None:
        """Runtime error observation — long-running faults (e.g. connection drops) reported here, orthogonal to per-result error fields."""

    @abstractmethod
    def recognize(
            self,
            audio_chunks: AsyncIterable[np.ndarray],
            *,
            stream_id: str | None = None,
    ) -> RecognitionStream:
        """Start a continuous recognition loop consuming the audio stream; returns a RecognitionStream."""

    async def recognize_once(self, audio_chunks: AsyncIterable[np.ndarray]) -> str:
        """Recognize a complete audio stream, return the accumulated text. Default implementation."""
        texts: list[str] = []
        async for result in self.recognize(audio_chunks):
            if result.phase == RecognitionPhase.TAIL:
                texts.append(result.text)
                break
            if result.phase == RecognitionPhase.CLAUSE:
                texts.append(result.text)
        return "".join(texts)

    @abstractmethod
    async def close(self) -> None:
        """Release ASR resources."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
