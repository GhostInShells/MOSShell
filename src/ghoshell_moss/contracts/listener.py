from abc import ABC, abstractmethod
from typing import Callable
from typing_extensions import Self

from .asr import RecognitionResult, RecognitionSegment
from .audio import AudioChunk

Discard = Callable[[], None]


class Listener(ABC):

    @property
    @abstractmethod
    def state(self) -> 'ListenerState | None':
        ...

    async def close_state(self, immediately: bool = False) -> bool:
        if state := self.state:
            await state.close(immediately=immediately)
            return True
        return False

    @abstractmethod
    async def listen(self, *, max_size: int = 10) -> 'ListenerState':
        ...

    @abstractmethod
    def on_audio_chunk(self, callback: Callable[[AudioChunk], None]) -> Discard:
        ...

    @abstractmethod
    def on_recognition_result(self, callback: Callable[[RecognitionResult], None]) -> Discard:
        ...

    @abstractmethod
    def on_recognition_segment(self, callback: Callable[[RecognitionSegment], None]) -> Discard:
        ...

    @abstractmethod
    def is_running(self) -> bool:
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...


class ListenerState(ABC):

    @abstractmethod
    def is_running(self) -> bool:
        ...

    @abstractmethod
    async def close(self, *, immediately: bool = False) -> None:
        ...

    @abstractmethod
    def commit(self) -> None:
        ...

    @abstractmethod
    def segments(self) -> list[RecognitionSegment]:
        ...
