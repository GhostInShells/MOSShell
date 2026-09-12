from abc import ABC, abstractmethod
from typing import Callable
from typing_extensions import Self

from .asr import RecognitionEvent, RecognitionSegment
from .audio import AudioChunk

Discard = Callable[[], None]


class Listener(ABC):
    """缝合 capture + asr 的"耳朵"器官 — 对称 Speech (嘴).

    持有一个 capture source (设备) 和一个 asr (注入), 拥有两者的生命周期.
    ``listen()`` 产出一条 listening session (ListenerState), 同一时刻至多一条 —
    再次 listen cancel 前一条.
    """

    @property
    @abstractmethod
    def state(self) -> 'ListenerState | None':
        """当前活跃 session, 无则 None."""

    @abstractmethod
    async def listen(self) -> 'ListenerState':
        """开一条 listening session (未启动), cancel 前一条."""

    @abstractmethod
    async def close(self) -> None:
        """关闭 Listener: 关 asr + capture, 结束活跃 session."""

    def is_listening(self) -> bool:
        state = self.state
        return state is not None and state.is_running()

    # 三个 on_*: 自动装线到当前 session (跨 session 稳定订阅)
    @abstractmethod
    def on_audio_chunk(self, callback: Callable[[AudioChunk], None]) -> Discard:
        ...

    @abstractmethod
    def on_recognition_result(self, callback: Callable[[RecognitionEvent], None]) -> Discard:
        ...

    @abstractmethod
    def on_recognition_segment(self, callback: Callable[[RecognitionSegment], None]) -> Discard:
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...


class ListenerState(ABC):
    """一条 listening session 的独立生命周期 — 对称 SpeechStream.

    ``async with state`` 启动后台 pump (订阅 capture consumer + asr.recognize),
    ``__aexit__`` 停 pump 并结束 recognition. 三个 on_* 观察本条 session.
    """

    @abstractmethod
    async def __aenter__(self) -> Self:
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...

    @abstractmethod
    def commit(self) -> None:
        """透传 recognition.commit() — 发负序号讯号, 切一个 segment (不关流)."""

    @abstractmethod
    def is_running(self) -> bool:
        ...

    @abstractmethod
    def on_audio_chunk(self, callback: Callable[[AudioChunk], None]) -> Discard:
        ...

    @abstractmethod
    def on_recognition_result(self, callback: Callable[[RecognitionEvent], None]) -> Discard:
        ...

    @abstractmethod
    def on_recognition_segment(self, callback: Callable[[RecognitionSegment], None]) -> Discard:
        ...
