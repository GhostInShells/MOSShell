"""Tests for TopicWarrant — non-host concrete (topic mode).

Uses QueueBasedTopicService (in-process, no zenoh) to verify the storage
wiring: store() publishes a write-request on the write topic, and an incoming
truth reconciles the cache + fires on_flushed. The peer (host) side is
simulated on the same service.
"""

from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio
from pydantic import BaseModel

from ghoshell_moss.core.blueprint.warrant import (
    Permission, PermissionStateData, AuthorizationResult,
)
from ghoshell_moss.core.concepts.qa import Question, Answer
from ghoshell_moss.core.qa.janus_qa import JanusQAManager
from ghoshell_moss.core.session.mock_session import MockSession
from ghoshell_moss.core.topic.queue_based import QueueBasedTopicService
from ghoshell_moss.matrix.warrant import TopicWarrant, WARRANT_NAMESPACE
from ghoshell_moss.matrix.warrant.topics import WarrantTruth, WarrantWriteRequest


# ── fixtures ─────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def topics():
    svc = QueueBasedTopicService(sender="test")
    await svc.start()
    yield svc
    await svc.close()


@pytest_asyncio.fixture
async def qa_mgr():
    async with JanusQAManager(issuer="test") as mgr:
        yield mgr


# ── test permission ──────────────────────────────────────────────


class AskState(BaseModel):
    approved: bool = False


class AskPermission(Permission[AskState]):
    @classmethod
    def key(cls) -> str:
        return "test.ask"

    @classmethod
    def type(cls) -> str:
        return "test.ask"

    def default(self) -> AskState:
        return AskState(approved=False)

    def check(self, state: AskState) -> Question | None:
        return Question(content="Proceed?", kind="apply")

    def replied(self, answer: Answer) -> AuthorizationResult[AskState]:
        approved = not answer.rejected
        return AuthorizationResult(
            allowed=approved,
            state=AskState(approved=approved),
            reason=None if approved else "Denied by user",
        )


def _approve_answer() -> Answer:
    return Answer(content="ok", rejected=False)


async def _answer_next(qa_mgr: JanusQAManager, namespace: str, answer: Answer) -> None:
    watcher = qa_mgr.watch(namespace)
    loop = asyncio.get_running_loop()
    done = loop.create_future()

    def _on_question(qa):
        if not done.done():
            done.set_result(qa)

    watcher.on_question(_on_question)
    qa = await done
    qa.reply(answer)


# ── store -> write-request ───────────────────────────────────────


@pytest.mark.asyncio
async def test_store_publishes_write_request(topics, qa_mgr):
    session = MockSession(topics=topics, qa_manager=qa_mgr)
    sub = topics.subscribe_model(WarrantWriteRequest)
    async with sub:
        async with TopicWarrant(session) as w:
            w.store(PermissionStateData(key="test.k", data={"n": 1}))
            req = await sub.poll_model(timeout=1.0)
    assert req is not None
    assert req.key == "test.k"
    assert req.seq == 1
    assert req.data == {"n": 1}


@pytest.mark.asyncio
async def test_store_assigns_monotonic_seq(topics, qa_mgr):
    session = MockSession(topics=topics, qa_manager=qa_mgr)
    sub = topics.subscribe_model(WarrantWriteRequest)
    async with sub:
        async with TopicWarrant(session) as w:
            w.store(PermissionStateData(key="test.k", data={"n": 1}))
            req1 = await sub.poll_model(timeout=1.0)
            w.store(PermissionStateData(key="test.k", data={"n": 2}))
            req2 = await sub.poll_model(timeout=1.0)
    assert req1.seq == 1
    assert req2.seq == 2


# ── truth reconciliation ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_truth_reconciles_cache_and_fires_on_flushed(topics, qa_mgr):
    session = MockSession(topics=topics, qa_manager=qa_mgr)
    event = asyncio.Event()
    fired: list[PermissionStateData] = []

    async with TopicWarrant(session) as w:
        def _on_flush(state: PermissionStateData) -> None:
            fired.append(state)
            event.set()

        w.on_flushed(_on_flush)
        topics.pub(WarrantTruth(key="test.k", seq=7, data={"n": 7}))
        await asyncio.wait_for(event.wait(), timeout=2.0)

    assert len(fired) == 1
    assert fired[0].seq == 7
    assert w.states()["test.k"].seq == 7
    assert w.states()["test.k"].data == {"n": 7}


# ── require via qa (authorization loop over topic wiring) ─────────


@pytest.mark.asyncio
async def test_require_with_qa_loop(topics, qa_mgr):
    session = MockSession(topics=topics, qa_manager=qa_mgr)

    async with TopicWarrant(session) as w:
        task = asyncio.ensure_future(
            _answer_next(qa_mgr, WARRANT_NAMESPACE, _approve_answer()),
        )
        result = await w.require(AskPermission())
        await task
        assert result.allowed is True
        assert result.state is not None and result.state.approved is True
