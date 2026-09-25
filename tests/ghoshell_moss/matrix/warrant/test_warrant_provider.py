"""Tests for SessionWarrantProvider: the host/non-host branch (v8 gap #1)."""

from __future__ import annotations

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.session.mock_session import MockSession
from ghoshell_moss.matrix.providers.warrant_provider import SessionWarrantProvider
from ghoshell_moss.matrix.warrant import SessionWarrant, TopicWarrant


class _FakeMatrix:
    def __init__(self, is_host: bool) -> None:
        self.is_host = is_host


class _StubCon:
    def __init__(self, session: Session, matrix: Matrix | None) -> None:
        self._session = session
        self._matrix = matrix

    def force_fetch(self, cls):
        if cls is Session:
            return self._session
        raise LookupError(cls)

    def get(self, cls):
        if cls is Matrix:
            return self._matrix
        return None


def _provider() -> SessionWarrantProvider:
    return SessionWarrantProvider()


def test_provider_branches_to_session_warrant_for_host():
    con = _StubCon(MockSession(), _FakeMatrix(is_host=True))
    assert isinstance(_provider().factory(con), SessionWarrant)


def test_provider_branches_to_topic_warrant_for_non_host():
    con = _StubCon(MockSession(), _FakeMatrix(is_host=False))
    assert isinstance(_provider().factory(con), TopicWarrant)


def test_provider_falls_back_to_host_without_matrix():
    # Matrix absent (KD7 fail-open) -> keep the old write-storage behaviour.
    con = _StubCon(MockSession(), matrix=None)
    assert isinstance(_provider().factory(con), SessionWarrant)
