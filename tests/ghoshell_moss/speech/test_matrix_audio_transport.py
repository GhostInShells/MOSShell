"""Unit tests for MatrixAudioTransport — verify delegation to Matrix/Session/Workspace."""
from unittest.mock import MagicMock

from ghoshell_moss.contracts.audio import AudioTransport
from ghoshell_moss.host.speech.capture.matrix_audio_transport import MatrixAudioTransport


def _make_transport():
    m = MagicMock()
    return MatrixAudioTransport(matrix=m), m


class TestPcmStream:
    """PCM pub/sub delegation."""

    def test_pub_pcm_delegates_to_session(self):
        transport, matrix = _make_transport()
        transport.pub_pcm(b"raw pcm data")
        matrix.session.pub_stream_delta.assert_called_once_with("audio/pcm", b"raw pcm data")

    def test_sub_pcm_callback_returns_release_handle(self):
        transport, matrix = _make_transport()
        matrix.session.sub_stream.return_value = "release_42"
        release = transport.sub_pcm_callback(lambda data: None)
        matrix.session.sub_stream.assert_called_once()
        assert release == "release_42"

    def test_sub_pcm_callback_extracts_payload_from_sample(self):
        transport, matrix = _make_transport()
        captured = []

        def fake_sub(key, callback):
            from ghoshell_moss.core.blueprint.session import Sample
            callback(Sample(relative_key=key, payload=b"hello"))
            return lambda: None

        matrix.session.sub_stream = fake_sub
        transport.sub_pcm_callback(captured.append)
        assert captured == [b"hello"]

    def test_sub_pcm_stream_delegates_to_session(self):
        transport, matrix = _make_transport()
        matrix.session.get_stream.return_value = "stream_42"
        result = transport.sub_pcm_stream(maxsize=128)
        matrix.session.get_stream.assert_called_once_with("audio/pcm", maxsize=128)
        assert result == "stream_42"


class TestProcessLock:
    """Lock acquire/release delegation."""

    def test_acquire_lock_creates_and_acquires(self):
        transport, matrix = _make_transport()
        transport.acquire_lock()
        matrix.workspace.lock.assert_called_once_with("audio_capture")
        matrix.workspace.lock.return_value.acquire.assert_called_once_with(timeout=0)

    def test_acquire_lock_returns_false_when_held(self):
        transport, matrix = _make_transport()
        matrix.workspace.lock.return_value.acquire.return_value = False
        assert transport.acquire_lock() is False

    def test_release_lock_before_acquire_is_safe(self):
        transport, matrix = _make_transport()
        transport.release_lock()  # no-op, no exception

    def test_release_lock_after_acquire(self):
        transport, matrix = _make_transport()
        transport.acquire_lock()
        transport.release_lock()
        matrix.workspace.lock.return_value.release.assert_called_once()


class TestTopicBroadcast:
    """Topic pub/window delegation."""

    def test_pub_topic_delegates_to_topic_service(self):
        transport, matrix = _make_transport()
        topic = MagicMock()
        transport.pub_topic(topic)
        matrix.session.topics.pub.assert_called_once_with(topic)

    def test_topic_window_creates_via_topic_service(self):
        transport, matrix = _make_transport()
        model = MagicMock()
        transport.topic_window(model, max_size=10)
        matrix.session.topics.create_window_for.assert_called_once_with(model, max_size=10)


class TestLogger:
    """Logger property."""

    def test_logger_returns_matrix_logger(self):
        transport, matrix = _make_transport()
        assert transport.logger is matrix.logger
