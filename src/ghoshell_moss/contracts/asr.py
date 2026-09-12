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
from typing import AsyncIterable, Awaitable, Callable

import numpy as np
from pydantic import BaseModel, Field

__all__ = [
    "ASR",
    "ASRInfo",
    "RecognitionStream",
    "RecognitionPhase",
    "Clause",
    "RecognitionEvent",
    "RecognitionSegment",
]


class RecognitionPhase(str, Enum):
    """Result phases of a recognition stream.

    - FIRST:   the first meaningful packet — the first result whose ``text`` is non-empty.
               It is NOT a segment/turn marker, just "the recognizer has text now". The
               engine does not emit it; the recognizer derives it. Emitted at most once
               per stream.
    - PARTIAL: intermediate result (utterance ``definite=false``), emitted while speaking.
               Only emitted when the text actually changed — the engine re-sends the same
               accumulated text on every audio package, identical consecutive text is not
               re-emitted.
    - CLAUSE:  stable sentence (utterance ``definite=true``). Each clause is emitted once,
               never suppressed by partial dedup.
    - TAIL:    segment tail (frame-level ``is_last_package=true``) — marks the end of a
               segment and triggers a segment cut.

    ``definite`` only marks "this sentence is stable", not the end of the stream.
    Stream end is a separate fact (the audio input is exhausted), not a TAIL.
    """

    FIRST = "first"
    PARTIAL = "partial"
    CLAUSE = "clause"
    TAIL = "tail"


@dataclass
class Clause:
    """A stable sentence finalized by the engine's VAD 判停 (text axis).

    ``text`` is the clause's own text — NOT the full accumulated text (that lives
    on ``RecognitionEvent.text``). ``additional`` carries the engine's raw
    utterance ``additions`` (说话人 / 情绪 / 音量 / 语速 / 语种...), passed
    through untouched so downstream keeps the full surface.
    """

    text: str = ""
    start_ms: int = 0
    end_ms: int = 0
    additional: dict = field(default_factory=dict)


@dataclass
class RecognitionEvent:
    """One event on the text axis of a recognition stream.

    ``text`` is the full accumulated text — full-replace, consistent across
    phases (FIRST carries the stream's first meaningful text). ``clause`` is
    present only for CLAUSE phase, holding the just-finalized sentence (its own
    text / timing / additional). ``segment_id`` links to the RecognitionSegment
    of the same segment (its ``id``).
    """

    stream_id: str
    segment_id: str  # segment id
    phase: RecognitionPhase
    text: str
    clause: Clause | None = None
    error: str = ""


@dataclass
class RecognitionSegment:
    """Audio archive of one segment (one turn) — the audio axis.

    Cut once per tail — ``text`` is the accumulated text of the whole segment,
    ``audio`` is the accumulated audio. ``start_ms``/``end_ms`` are stream-relative
    timestamps of the segment; ``offset_ms`` is the stream-relative start of
    ``audio``, used by ``precise_cut``.

    Linked to RecognitionEvent via ``segment_id`` (its ``id``) + ``stream_id``,
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

    The model reads get_info() first: the audio input contract (sample_rate/bits/channel),
    the core standard behavior param (vad_end_window_ms — cross-implementation, params-free),
    plus the JSON schema and current values of the tunable implementation-specific params.
    Then configure() turns the behavior knobs for the next recognize(). Each implementation
    exposes its own params BaseModel; the contract only carries the schema and current
    values as dicts.
    """

    sample_rate: int = Field(default=16000, description="sample rate the ASR expects")
    bits: int = Field(default=16, description="bit depth")
    channel: int = Field(default=1, description="channel count")

    vad_end_window_ms: int = Field(
        default=0,
        description="分句判停阈值 (ms) — 连续静音达该值判定一句结束。跨实现核心标准参数, "
                    "与具体实现的 params 无关; 0 表示实现未定义/不支持",
    )

    params_schema: dict = Field(default_factory=dict,
                                description="json schema of tunable behavior params (each implementation exposes its own BaseModel)")
    params: dict = Field(default_factory=dict, description="current behavior param values")


class RecognitionStream(ABC):
    """One recognition stream — one continuous audio input -> a sequence of RecognitionEvent.

    1 stream = n segments (each segment = one turn = one WS). The stream is bounded by
    the audio input, not by a single turn: ``commit()`` ends the current segment with a
    TAIL and the recognizer opens the next segment, until the audio input is exhausted
    or the stream is closed (``close()``).

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
    def on_event_creating(
            self,
            callback: Callable[[RecognitionEvent], Awaitable[None] | None],
    ) -> None:
        """Register a commit-decision hook, invoked the moment a RecognitionEvent is
        parsed — before it is enqueued. Fires on FIRST/PARTIAL/CLAUSE, not on TAIL
        (TAIL is already the commit result).

        Two callback shapes, decided by the callback's own return:

        - returns an Awaitable (async def): the recognizer awaits it inline — a
          blocking consumption point. The receive loop stalls until it returns, so
          keep it short (e.g. deciding to ``commit()``). If the callback needs
          concurrency it must spawn its own task.
        - returns a plain value (sync def): the recognizer offloads it to a thread
          (``asyncio.to_thread``) — non-blocking, naturally parallel.

        The commit mechanism itself is ``commit()``; this hook only decides when to
        call it."""

    @abstractmethod
    def commit(self) -> None:
        """End the current segment — the recognizer produces a TAIL, then opens the next segment (does not close the stream)."""

    @abstractmethod
    async def close(self) -> None:
        """Actively stop the stream — no further results, no TAIL."""

    @abstractmethod
    def is_input_done(self) -> bool:
        """Whether the audio input loop has stopped."""

    @abstractmethod
    def __aiter__(self) -> "RecognitionStream":
        """Start sending audio; runs until the audio stream ends. Single entry only."""

    @abstractmethod
    async def __anext__(self) -> RecognitionEvent:
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
                return result.text
            if result.phase == RecognitionPhase.CLAUSE and result.clause is not None:
                texts.append(result.clause.text)
        return "".join(texts)

    @abstractmethod
    async def close(self) -> None:
        """Release ASR resources."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
