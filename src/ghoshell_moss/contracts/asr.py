"""
ASR contracts — audio speech recognition abstractions.

ASR is the ear: raw audio → text stream. Kept separate from speech (the mouth)
to avoid cross-infection in the contract layer.
"""
from abc import ABC, abstractmethod
from typing import AsyncIterable, NamedTuple

import numpy as np

__all__ = [
    "ASR",
    "ASRResult",
]


class ASRResult(NamedTuple):
    """ASR recognition result fragment."""

    text: str
    is_final: bool = False


class ASR(ABC):
    """Audio perception organ — ear. Symmetric to TTS (mouth).

    Input: audio stream. Output: text stream.
    Intermediate results have is_final=False; tail packet has is_final=True.
    """

    @abstractmethod
    async def recognize(
        self,
        audio_chunks: AsyncIterable[np.ndarray],
    ) -> AsyncIterable[ASRResult]:
        """Streaming recognition. Yields intermediate results; last one is is_final=True."""

    async def recognize_once(self, audio_chunks: AsyncIterable[np.ndarray]) -> str:
        """Recognize a complete audio segment, return final text. Default implementation."""
        async for result in self.recognize(audio_chunks):
            if result.is_final:
                return result.text
        return ""

    @abstractmethod
    async def close(self) -> None:
        """Release ASR resources."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
