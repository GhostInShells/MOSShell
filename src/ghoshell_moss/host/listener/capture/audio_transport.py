"""
Audio transport abstraction — isolates audio core from Matrix/Session/Zenoh.

This lives in the host layer (not contracts) because it couples to
core types: StreamSubscriber, TopicModel, TopicWindow. Contracts stay
clean; implementations swap transport by replacing the adapter.
"""
import logging
from abc import ABC, abstractmethod
from typing import Callable

from ghoshell_moss.core.blueprint.session import StreamSubscriber
from ghoshell_moss.core.concepts.topic import TOPIC_MODEL, TopicModel, TopicWindow

__all__ = ["AudioTransport"]


class AudioTransport(ABC):
    """Transport abstraction isolating audio core from Matrix/Session/Zenoh.

    Audio capture only needs: publish PCM, subscribe to PCM, process lock,
    topic broadcast, and a logger. How those are implemented (Zenoh, FileLocker,
    TopicService) is the adapter's concern — not audio core's.

    Single coupling point: MatrixAudioTransport in host/voice/capture/.
    """

    # -- PCM stream --
    @abstractmethod
    def pub_pcm(self, chunk: bytes) -> None:
        """Publish a raw PCM chunk to the audio stream."""
        ...

    @abstractmethod
    def sub_pcm_callback(self, on_chunk: Callable[[bytes], None]) -> Callable[[], None]:
        """Subscribe to PCM stream via callback. Returns a release handle."""
        ...

    @abstractmethod
    def sub_pcm_stream(self, maxsize: int) -> StreamSubscriber:
        """Subscribe to PCM stream as an async iterable."""
        ...

    # -- process lock --
    @abstractmethod
    def acquire_lock(self) -> bool:
        """Acquire cross-process lock for exclusive device access."""
        ...

    @abstractmethod
    def release_lock(self) -> None:
        """Release the process lock."""
        ...

    # -- topic broadcast --
    @abstractmethod
    def pub_topic(self, topic: TopicModel) -> None:
        """Publish a topic via the transport's topic service."""
        ...

    @abstractmethod
    def topic_window(self, model: type[TOPIC_MODEL], max_size: int) -> TopicWindow[TOPIC_MODEL]:
        """Create a bounded sliding window over a topic stream."""
        ...

    # -- logger --
    @property
    @abstractmethod
    def logger(self) -> logging.Logger:
        """Logger for audio capture diagnostics."""
        ...
