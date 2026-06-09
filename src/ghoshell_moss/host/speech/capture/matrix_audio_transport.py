"""
MatrixAudioTransport — adapts Matrix/Session to AudioTransport ABC.

Single coupling point between audio core and Matrix runtime. Replace this
implementation to switch transport (e.g. independent Zenoh session on port 20775).
"""
import logging
from typing import Callable

from ghoshell_moss.host.speech.capture.audio_transport import AudioTransport
from ghoshell_moss.contracts.workspace import Lock
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.session import Sample, StreamSubscriber
from ghoshell_moss.core.concepts.topic import TOPIC_MODEL, TopicModel, TopicWindow

__all__ = ["MatrixAudioTransport"]

_STREAM_KEY = "audio/pcm"
_LOCK_KEY = "audio_capture"


class MatrixAudioTransport(AudioTransport):
    """Adapt Matrix/Session/Zenoh to the AudioTransport contract."""

    def __init__(self, *, matrix: Matrix):
        self._matrix = matrix
        self._locker: Lock | None = None

    # -- PCM stream --

    def pub_pcm(self, chunk: bytes) -> None:
        self._matrix.session.pub_stream_delta(_STREAM_KEY, chunk)

    def sub_pcm_callback(self, on_chunk: Callable[[bytes], None]) -> Callable[[], None]:
        def _adapt(sample: Sample) -> None:
            on_chunk(sample.payload)

        return self._matrix.session.sub_stream(_STREAM_KEY, _adapt)

    def sub_pcm_stream(self, maxsize: int) -> StreamSubscriber:
        return self._matrix.session.get_stream(_STREAM_KEY, maxsize=maxsize)

    # -- process lock --

    def acquire_lock(self) -> bool:
        self._locker = self._matrix.workspace.lock(_LOCK_KEY)
        return self._locker.acquire(timeout=0)

    def release_lock(self) -> None:
        if self._locker is not None:
            self._locker.release()
            self._locker = None

    # -- topic broadcast --

    def pub_topic(self, topic: TopicModel) -> None:
        self._matrix.session.topics.pub(topic)

    def topic_window(self, model: type[TOPIC_MODEL], max_size: int) -> TopicWindow[TOPIC_MODEL]:
        return self._matrix.session.topics.create_window_for(model, max_size=max_size)

    # -- logger --

    @property
    def logger(self) -> logging.Logger:
        return self._matrix.logger or logging.getLogger("moss.audio_capture")
