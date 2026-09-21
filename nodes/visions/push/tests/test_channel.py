import asyncio
import time

import pytest

from push_node.channel import build_push_channel
from push_node.session import SessionState
from push_node.store import AcceptAll, PushStore

from fakes import FakeSubprocesses, Recorder

_URL = "http://127.0.0.1:9"


@pytest.fixture
def store():
    return PushStore()


def _channel(store, processes, rec=None, accept_all=None, surface_url=None):
    rec = rec or Recorder()
    accept_all = accept_all or AcceptAll()
    chan = build_push_channel(
        store,
        accept_all,
        processes,
        surface=rec,
        signaler=rec,
        surface_url=surface_url or (lambda: _URL),
    )
    return chan, rec


async def _until(predicate, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


@pytest.mark.asyncio
async def test_command_set_is_exposed(store):
    chan, _ = _channel(store, FakeSubprocesses())
    async with chan.bootstrap() as runtime:
        for name in ("request", "sessions", "read", "stop", "stop_all"):
            assert runtime.get_command(name) is not None, name


@pytest.mark.asyncio
async def test_request_returns_a_receipt_and_spawns_nothing(store):
    processes = FakeSubprocesses()
    chan, rec = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        receipt = await runtime.execute_command(
            "request", args=("screen",), kwargs={"label": "desk"}
        )
        assert "awaiting approval" in receipt, "the model must not block on the human"
        assert processes.spawned == [], "nothing streams before a verdict"
        session = store.sessions()[-1]
        assert session.state == SessionState.PENDING
        assert rec.types() == ["session"], "the request is announced on the surface"


@pytest.mark.asyncio
async def test_accept_spawns_the_producer_and_signals_the_url(store):
    processes = FakeSubprocesses()
    chan, rec = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("request", args=("screen",), kwargs={"label": "desk"})
        session = store.sessions()[-1]

        store.settle(session.id, "accept")
        await _until(lambda: session.state == SessionState.LIVE)

        assert len(processes.spawned) == 1
        assert "ffmpeg" in processes.spawned[0].meta.command
        joined = " ".join(str(s.messages[0].to_content_string()) for s in rec.signals)
        assert f"{_URL}/stream/{session.id}" in joined, "the live URL is signalled"


@pytest.mark.asyncio
async def test_deny_spawns_nothing(store):
    processes = FakeSubprocesses()
    chan, _ = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("request", args=("screen",))
        session = store.sessions()[-1]

        store.settle(session.id, "deny")
        await _until(lambda: session.state == SessionState.DENIED)
        assert processes.spawned == []


@pytest.mark.asyncio
async def test_the_model_cannot_stop_a_human_session(store):
    chan, _ = _channel(store, FakeSubprocesses())
    async with chan.bootstrap() as runtime:
        human = store.request("camera", owner="human")
        store.set_state(human.id, SessionState.LIVE)

        out = await runtime.execute_command("stop", args=(human.id,))
        assert "not yours" in out
        assert human.state == SessionState.LIVE, "the human's stream is untouched"


@pytest.mark.asyncio
async def test_the_model_stops_its_own_live_session(store):
    processes = FakeSubprocesses()
    chan, _ = _channel(store, processes)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("request", args=("screen",))
        session = store.sessions()[-1]
        store.settle(session.id, "accept")
        await _until(lambda: session.state == SessionState.LIVE)

        out = await runtime.execute_command("stop", args=(session.id,))
        assert "stopped" in out
        assert session.state == SessionState.STOPPED
        assert processes.spawned[0].stopped is True


@pytest.mark.asyncio
async def test_accept_all_skips_the_verdict(store):
    processes = FakeSubprocesses()
    accept_all = AcceptAll()
    accept_all.enabled = True
    chan, _ = _channel(store, processes, accept_all=accept_all)
    async with chan.bootstrap() as runtime:
        receipt = await runtime.execute_command("request", args=("screen",))
        assert "accept-all" in receipt
        session = store.sessions()[-1]
        await _until(lambda: session.state == SessionState.LIVE)
        assert len(processes.spawned) == 1
