"""Tests for SessionWarrant's host receive-side (reject-retry + truth broadcast, 2b).

Host + non-host talk over one QueueBasedTopicService (in-process, no zenoh).
The reject-retry table: accept (seq == current+1), duplicate (seq == current),
首读 (seq == 1 on an existing key -> rebroadcast truth), stale (seq < current
and seq >= 2 -> drop), gap (seq > current+1 -> rebroadcast current truth).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import pytest_asyncio

from ghoshell_moss.core.blueprint.warrant import PermissionStateData
from ghoshell_moss.core.qa.janus_qa import JanusQAManager
from ghoshell_moss.core.session.mock_session import MockSession
from ghoshell_moss.core.topic.queue_based import QueueBasedTopicService
from ghoshell_moss.matrix.warrant import SessionWarrant, TopicWarrant
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


@pytest.fixture
def warrant_dir(tmp_path) -> Path:
    return tmp_path / "warrants"


def _seed_host_state(warrant_dir: Path, key: str, seq: int, data: dict) -> None:
    warrant_dir.mkdir(parents=True, exist_ok=True)
    state = PermissionStateData(key=key, seq=seq, data=data)
    (warrant_dir / f"{key}.json").write_text(state.model_dump_json(indent=2))


def _host(topics, qa_mgr, warrant_dir) -> SessionWarrant:
    return SessionWarrant(
        MockSession(topics=topics, qa_manager=qa_mgr),
        states_dir=warrant_dir,
    )


def _non_host(topics, qa_mgr) -> TopicWarrant:
    return TopicWarrant(MockSession(topics=topics, qa_manager=qa_mgr))


# ── accept: full loop ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_accept_loop(topics, qa_mgr, warrant_dir):
    host = _host(topics, qa_mgr, warrant_dir)
    non_host = _non_host(topics, qa_mgr)
    event = asyncio.Event()

    async with host, non_host:
        def _on_flush(state: PermissionStateData) -> None:
            if state.key == "test.k":
                event.set()

        non_host.on_flushed(_on_flush)
        non_host.store(PermissionStateData(key="test.k", data={"n": 1}))
        await asyncio.wait_for(event.wait(), timeout=3.0)

    assert host.states()["test.k"].seq == 1
    assert non_host.states()["test.k"].seq == 1
    assert non_host.states()["test.k"].data == {"n": 1}


# ── 首读: seq=1 on an existing key rebroadcasts truth ────────────


@pytest.mark.asyncio
async def test_first_read_rebroadcasts_truth(topics, qa_mgr, warrant_dir):
    _seed_host_state(warrant_dir, "test.k", seq=5, data={"n": 5})
    host = _host(topics, qa_mgr, warrant_dir)
    non_host = _non_host(topics, qa_mgr)
    event = asyncio.Event()
    reconciled: list[int] = []

    async with host, non_host:
        def _on_flush(state: PermissionStateData) -> None:
            if state.key == "test.k":
                reconciled.append(state.seq)
                event.set()

        non_host.on_flushed(_on_flush)
        non_host.store(PermissionStateData(key="test.k", data={"n": 99}))
        await asyncio.wait_for(event.wait(), timeout=3.0)

    assert reconciled == [5]
    assert non_host.states()["test.k"].seq == 5
    assert non_host.states()["test.k"].data == {"n": 5}
    assert host.states()["test.k"].seq == 5  # seq=1 was not accepted


# ── gap: seq too high rebroadcasts current truth ─────────────────


@pytest.mark.asyncio
async def test_gap_broadcasts_current_truth(topics, qa_mgr, warrant_dir):
    host = _host(topics, qa_mgr, warrant_dir)
    truth_sub = topics.subscribe_model(WarrantTruth)

    async with host:
        async with truth_sub:
            topics.pub(WarrantWriteRequest(key="test.k", seq=3, data={"n": 3}))
            truth = await truth_sub.poll_model(timeout=3.0)

    assert truth is not None
    assert truth.key == "test.k"
    assert truth.seq == 0
    assert truth.data == {}


# ── stale + duplicate: no regression ─────────────────────────────


@pytest.mark.asyncio
async def test_stale_and_duplicate_do_not_regress(topics, qa_mgr, warrant_dir):
    _seed_host_state(warrant_dir, "test.k", seq=5, data={"n": 5})
    host = _host(topics, qa_mgr, warrant_dir)
    truth_sub = topics.subscribe_model(WarrantTruth)

    async with host:
        async with truth_sub:
            # stale (3 < 5) and duplicate (5 == 5) — both must be no-ops.
            topics.pub(WarrantWriteRequest(key="test.k", seq=3, data={"n": 3}))
            topics.pub(WarrantWriteRequest(key="test.k", seq=5, data={"n": 5}))
            # barrier: accepted only if 3 and 5 were ignored (else it would gap).
            topics.pub(WarrantWriteRequest(key="test.k", seq=6, data={"n": 6}))
            truth = await truth_sub.poll_model(timeout=3.0)

    assert truth is not None
    assert truth.seq == 6
    assert host.states()["test.k"].seq == 6
    assert host.states()["test.k"].data == {"n": 6}
