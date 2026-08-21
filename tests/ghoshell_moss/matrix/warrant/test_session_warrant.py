"""Tests for SessionWarrant — concrete Warrant backed by Session.

Uses MockSession + JanusQAManager for fully in-process tests.
Each test validates a behaviour contract of the Warrant ABC, not
SessionWarrant internals.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel

from ghoshell_moss.core.blueprint.warrant import (
    Permission, PermissionStateData, AuthorizationResult,
)
from ghoshell_moss.core.concepts.qa import Question, Answer
from ghoshell_moss.core.qa.janus_qa import JanusQAManager
from ghoshell_moss.core.session.mock_session import MockSession
from ghoshell_moss.matrix.warrant import SessionWarrant, WARRANT_NAMESPACE


# ── fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def warrant_dir() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _new_warrant(session: MockSession, warrant_dir: Path) -> SessionWarrant:
    return SessionWarrant(session, states_dir=warrant_dir)


# ── test permissions ────────────────────────────────────────────


class AutoPassState(BaseModel):
    counter: int = 0


class AutoPassPermission(Permission[AutoPassState]):
    """Always passes — check returns None."""

    @classmethod
    def key(cls) -> str:
        return "test.auto_pass"

    @classmethod
    def type(cls) -> str:
        return "test.auto_pass"

    def default(self) -> AutoPassState:
        return AutoPassState(counter=0)

    def check(self, state: AutoPassState) -> Question | None:
        return None

    def replied(self, answer: Answer) -> AuthorizationResult[AutoPassState]:
        return AuthorizationResult(allowed=True, state=AutoPassState(counter=1))


class AskApprovalState(BaseModel):
    approved: bool = False


class AskApprovalPermission(Permission[AskApprovalState]):
    """Asks an approval question (kind=apply); approved if not rejected."""

    @classmethod
    def key(cls) -> str:
        return "test.ask_approval"

    @classmethod
    def type(cls) -> str:
        return "test.ask_approval"

    def default(self) -> AskApprovalState:
        return AskApprovalState(approved=False)

    def check(self, state: AskApprovalState) -> Question | None:
        return Question(content="Proceed?", kind="apply")

    def replied(self, answer: Answer) -> AuthorizationResult[AskApprovalState]:
        approved = not answer.rejected
        return AuthorizationResult(
            allowed=approved,
            state=AskApprovalState(approved=approved),
            reason=None if approved else "Denied by user",
        )


class AlwaysDenyState(BaseModel):
    asked: bool = False


class AlwaysDenyPermission(Permission[AlwaysDenyState]):
    """Asks then always returns denied in replied, no state stored."""

    @classmethod
    def key(cls) -> str:
        return "test.always_deny"

    @classmethod
    def type(cls) -> str:
        return "test.always_deny"

    def default(self) -> AlwaysDenyState:
        return AlwaysDenyState(asked=False)

    def check(self, state: AlwaysDenyState) -> Question | None:
        return Question(content="Deny me?", kind="apply")

    def replied(self, answer: Answer) -> AuthorizationResult[AlwaysDenyState]:
        return AuthorizationResult(
            allowed=False,
            reason="Always denied",
            state=None,  # no state → not stored
        )


# ── helpers ──────────────────────────────────────────────────────


def _make_approve_answer() -> Answer:
    return Answer(content="ok", rejected=False)


def _make_reject_answer() -> Answer:
    return Answer(content="no", rejected=True)


async def _answer_next_question(qa_mgr: JanusQAManager, namespace: str, answer: Answer) -> None:
    """Watch *namespace*, reply to the next question with *answer*."""
    watcher = qa_mgr.watch(namespace)
    loop = asyncio.get_running_loop()
    done = loop.create_future()

    def _on_question(qa):
        if not done.done():
            done.set_result(qa)

    watcher.on_question(_on_question)
    qa = await done
    qa.reply(answer)


# ── lifecycle ────────────────────────────────────────────────────


def test_not_running_before_enter(warrant_dir: Path):
    session = MockSession()
    warrant = _new_warrant(session, warrant_dir)
    assert not warrant.is_running()


@pytest.mark.asyncio
async def test_running_after_enter(warrant_dir: Path):
    session = MockSession()
    async with _new_warrant(session, warrant_dir) as w:
        assert w.is_running()


@pytest.mark.asyncio
async def test_not_running_after_exit(warrant_dir: Path):
    session = MockSession()
    warrant = _new_warrant(session, warrant_dir)
    async with warrant:
        pass
    assert not warrant.is_running()


# ── require ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_auto_pass_permission(warrant_dir: Path):
    """check returns None → allowed=True immediately, no QA roundtrip, no state."""
    session = MockSession()
    async with _new_warrant(session, warrant_dir) as warrant:
        result = await warrant.require(AutoPassPermission())
        assert result.allowed is True
        assert result.state is None  # no QA → no replied → no state


@pytest.mark.asyncio
async def test_approval_approved(warrant_dir: Path):
    """check returns Question → watcher approves → allowed=True, state stored."""
    async with JanusQAManager(issuer="test") as qa_mgr:
        session = MockSession(qa_manager=qa_mgr)
        async with _new_warrant(session, warrant_dir) as warrant:
            task = asyncio.ensure_future(
                _answer_next_question(qa_mgr, WARRANT_NAMESPACE, _make_approve_answer()),
            )
            result = await warrant.require(AskApprovalPermission())
            await task
            assert result.allowed is True
            assert result.state is not None
            assert result.state.approved is True


@pytest.mark.asyncio
async def test_approval_rejected(warrant_dir: Path):
    """check returns Question → watcher rejects → allowed=False."""
    async with JanusQAManager(issuer="test") as qa_mgr:
        session = MockSession(qa_manager=qa_mgr)
        async with _new_warrant(session, warrant_dir) as warrant:
            task = asyncio.ensure_future(
                _answer_next_question(qa_mgr, WARRANT_NAMESPACE, _make_reject_answer()),
            )
            result = await warrant.require(AskApprovalPermission())
            await task
            assert result.allowed is False
            assert result.reason is not None


@pytest.mark.asyncio
async def test_deny_no_state_stored(warrant_dir: Path):
    """replied returns state=None → nothing persisted in cache."""
    async with JanusQAManager(issuer="test") as qa_mgr:
        session = MockSession(qa_manager=qa_mgr)
        async with _new_warrant(session, warrant_dir) as warrant:
            task = asyncio.ensure_future(
                _answer_next_question(qa_mgr, WARRANT_NAMESPACE, _make_approve_answer()),
            )
            result = await warrant.require(AlwaysDenyPermission())
            await task
            assert result.allowed is False
            assert result.state is None
            assert AlwaysDenyPermission.key() not in warrant.states()


@pytest.mark.asyncio
async def test_cancelled_error_propagates(warrant_dir: Path):
    """CancelledError during ask_question propagates, not swallowed."""
    async with JanusQAManager(issuer="test") as qa_mgr:
        session = MockSession(qa_manager=qa_mgr)
        async with _new_warrant(session, warrant_dir) as warrant:
            async def ask():
                return await warrant.require(AskApprovalPermission())

            t = asyncio.ensure_future(ask())
            await asyncio.sleep(0.01)
            t.cancel()
            with pytest.raises(asyncio.CancelledError):
                await t


# ── state persistence ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_state_reloaded_after_restart(warrant_dir: Path):
    """Store a state, exit, re-enter — state is loaded from storage."""
    session = MockSession()
    state = PermissionStateData(
        key="test.auto_pass",
        data=AutoPassState(counter=5).model_dump(mode="json"),
    )

    async with _new_warrant(session, warrant_dir) as w:
        w.store(state)

    async with _new_warrant(session, warrant_dir) as w:
        restored = w.get_permission_state(AutoPassPermission())
        assert restored.counter == 5


@pytest.mark.asyncio
async def test_get_permission_state_fallback(warrant_dir: Path):
    """No stored state → fallback to permission.default()."""
    session = MockSession()
    async with _new_warrant(session, warrant_dir) as w:
        state = w.get_permission_state(AutoPassPermission())
        assert state == AutoPassPermission().default()
        assert state.counter == 0


# ── seq & on_flushed ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_seq_round_trips_through_storage(warrant_dir: Path):
    """A PermissionStateData's seq survives store → flush → reload."""
    session = MockSession()
    state = PermissionStateData(
        key="test.auto_pass",
        seq=7,
        data=AutoPassState(counter=3).model_dump(mode="json"),
    )

    async with _new_warrant(session, warrant_dir) as w:
        w.store(state)

    async with _new_warrant(session, warrant_dir) as w:
        restored = w.states().get("test.auto_pass")
        assert restored is not None
        assert restored.seq == 7


@pytest.mark.asyncio
async def test_on_flushed_fires_after_real_flush(warrant_dir: Path):
    """on_flushed callback fires once, after the state is actually persisted."""
    session = MockSession()
    event = asyncio.Event()
    fired: list[PermissionStateData] = []

    async with _new_warrant(session, warrant_dir) as w:
        def _on_flush(state: PermissionStateData) -> None:
            fired.append(state)
            event.set()

        w.on_flushed(_on_flush)
        w.store(PermissionStateData(key="test.auto_pass", data={"counter": 1}))
        await asyncio.wait_for(event.wait(), timeout=2.0)

    assert len(fired) == 1
    assert fired[0].key == "test.auto_pass"


@pytest.mark.asyncio
async def test_on_flushed_unsubscribe(warrant_dir: Path):
    """The unsubscribe handle stops a callback from firing."""
    session = MockSession()
    kept: list[PermissionStateData] = []
    dropped: list[PermissionStateData] = []
    event = asyncio.Event()

    async with _new_warrant(session, warrant_dir) as w:
        def _kept(state: PermissionStateData) -> None:
            kept.append(state)
            event.set()

        def _dropped(state: PermissionStateData) -> None:
            dropped.append(state)

        w.on_flushed(_kept)
        unsub = w.on_flushed(_dropped)
        unsub()

        w.store(PermissionStateData(key="test.auto_pass", data={"counter": 1}))
        await asyncio.wait_for(event.wait(), timeout=2.0)

    assert len(kept) == 1
    assert len(dropped) == 0
