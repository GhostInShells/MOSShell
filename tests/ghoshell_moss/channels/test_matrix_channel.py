"""Tests for matrix channel — surface tiers (cold / warm / hot).

matrix_channel owns the local + network cell governance surface. What is pinned
here is the delivery tier of each surface, not the governance verbs. Both
channels are all-warm: state rides notice, which the kernel diffs by text.

- nodes: running / recently-exited cells. The notice text must be byte-stable
  between refreshes — a drifting field (uptime, age) would defeat the diff and
  re-emit the whole section every refresh.
- mesh: a bounded tail of recent cell events. The tail must be capped, and it
  must not re-emit while no new event has arrived.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ghoshell_moss.channels.matrix_channel import (
    new_mesh_channel,
    new_nodes_channel,
)
from ghoshell_moss.core.blueprint.cell import AutoAcceptPolicy, CellEvent, CellEventLevel

_UID_A = "01J8ZZZZZZAAAAAAAAAAAAAA1"
_UID_B = "01J8ZZZZZZBBBBBBBBBBBBBB2"


def _handle(uid: str, pid: int, *, exit_code=None):
    runtime = SimpleNamespace(
        address=f"node/vision/{uid}",
        cell=SimpleNamespace(
            category="vision", name="vision", fullname="node/vision",
            providing=[], home="/tmp",
        ),
        event_level=CellEventLevel.INFO,
    )
    meta = SimpleNamespace(
        pid=pid, exit_code=exit_code, created=0.0, updated=0.0, cwd="/tmp",
    )
    return SimpleNamespace(
        runtime=runtime, address=runtime.address,
        process=SimpleNamespace(meta=meta, output=None),
    )


class _NodesCatalog:
    def list_nodes(self, refresh=False, paths=None, installed=None):
        return {
            "nodes/visions/camera": SimpleNamespace(
                name="camera", category="vision", installed=True,
                description="camera stream",
            ),
        }


class _StubMatrix:
    def __init__(self):
        self.project = SimpleNamespace(nodes=_NodesCatalog())
        self.handled: dict = {}
        self.dead: list = []

    def handled_cells(self):
        return self.handled

    def dead_cells(self):
        return self.dead


class _StubMesh:
    def __init__(self):
        self._on_event = None
        self._policy = AutoAcceptPolicy(local=True, foreign=False)

    def on_event(self, callback):
        self._on_event = callback
        return lambda: None

    def channel_proxies(self):
        return {}

    def auto_accept(self):
        return self._policy

    def set_auto_accept(self, *, local=None, foreign=None):
        self._policy = AutoAcceptPolicy(
            local=self._policy.local if local is None else local,
            foreign=self._policy.foreign if foreign is None else foreign,
        )

    def recent_events(self, *, limit=20):
        return []

    def cell_events(self, address, *, limit=20):
        return []

    def emit(self, content: str, level=CellEventLevel.INFO):
        self._on_event(
            CellEvent(
                address=f"node/vision/{_UID_A}", content=content, event_level=level,
            )
        )


# ---- nodes: warm state rides notice, not the hot band ---- #

@pytest.mark.asyncio
async def test_nodes_state_rides_notice_not_context():
    matrix = _StubMatrix()
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        handle = _handle(_UID_A, pid=4242)
        matrix.handled[handle.address] = handle
        await runtime.refresh_metas()
        meta = runtime.self_meta()

        assert meta.context == []
        assert "running (1)" in meta.notice
        assert "installed nodes (1)" in meta.notice


@pytest.mark.asyncio
async def test_nodes_notice_text_is_stable_across_refreshes():
    matrix = _StubMatrix()
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        handle = _handle(_UID_A, pid=4242)
        matrix.handled[handle.address] = handle
        await runtime.refresh_metas()
        first = runtime.self_meta().notice

        await runtime.refresh_metas()
        second = runtime.self_meta().notice
        assert first == second, (
            "notice text drifted without a state change — the facade diff "
            "would re-emit it on every refresh"
        )


@pytest.mark.asyncio
async def test_nodes_notice_reports_exit_without_age_text():
    matrix = _StubMatrix()
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        matrix.dead.append(_handle(_UID_B, pid=4243, exit_code=1))
        await runtime.refresh_metas()
        notice = runtime.self_meta().notice

        assert "recently exited (1)" in notice
        assert "exit=1" in notice
        assert "ago" not in notice, "a drifting age field defeats the facade diff"
        assert "uptime" not in notice


# ---- mesh: the event tail is warm state, capped ---- #

def _net(stub):
    """Mesh channel only needs ``matrix.network()`` at startup + refresh."""
    async def _network():
        return stub

    return SimpleNamespace(network=_network)


@pytest.mark.asyncio
async def test_mesh_events_ride_notice_not_context():
    stub = _StubMesh()
    chan = new_mesh_channel(_net(stub))
    async with chan.bootstrap() as runtime:
        stub.emit("first")
        stub.emit("second")
        await runtime.refresh_metas()
        meta = runtime.self_meta()

        assert meta.context == [], "matrix has no perception-tier data"
        assert "first" in meta.notice and "second" in meta.notice


@pytest.mark.asyncio
async def test_mesh_notice_shows_policy_without_events():
    stub = _StubMesh()
    chan = new_mesh_channel(_net(stub))
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        notice = runtime.self_meta().notice

        assert "auto_accept: local=True, foreign=False" in notice
        assert "recent events" not in notice


@pytest.mark.asyncio
async def test_mesh_notice_is_stable_without_new_events():
    stub = _StubMesh()
    chan = new_mesh_channel(_net(stub))
    async with chan.bootstrap() as runtime:
        stub.emit("first")
        await runtime.refresh_metas()
        first = runtime.self_meta().notice

        await runtime.refresh_metas()
        assert runtime.self_meta().notice == first, (
            "an unchanged tail must not re-emit — only a new event may change it"
        )


@pytest.mark.asyncio
async def test_mesh_event_tail_is_capped():
    stub = _StubMesh()
    chan = new_mesh_channel(_net(stub))
    async with chan.bootstrap() as runtime:
        for i in range(12):
            stub.emit(f"flood-{i}")
        await runtime.refresh_metas()
        notice = runtime.self_meta().notice

        assert "flood-11" in notice, "the newest survive"
        assert "flood-3" not in notice, "the tail window is bounded"
        assert "more, events() for the tail" in notice, "overspill points at the pull path"


# ---- mesh: trust-op visibility follows the actual policy ---- #

def _command_available(runtime, name: str) -> bool:
    for cmd in runtime.self_meta().commands:
        if cmd.name == name:
            return cmd.available
    return False


@pytest.mark.asyncio
async def test_accept_reject_hidden_when_auto_accept_covers_all():
    stub = _StubMesh()
    stub._policy = AutoAcceptPolicy(local=True, foreign=True)
    chan = new_mesh_channel(_net(stub))
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()

        assert _command_available(runtime, "accept") is False
        assert _command_available(runtime, "reject") is False


@pytest.mark.asyncio
async def test_accept_reject_visible_under_partial_policy():
    stub = _StubMesh()  # default local=True, foreign=False
    chan = new_mesh_channel(_net(stub))
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()

        assert _command_available(runtime, "accept") is True
        assert _command_available(runtime, "reject") is True


@pytest.mark.asyncio
async def test_set_auto_accept_reports_result_not_request():
    stub = _StubMesh()  # foreign=False
    chan = new_mesh_channel(_net(stub))
    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command("set_auto_accept", kwargs={"foreign": True})

        assert "foreign=True" in result, "the reply states the resulting policy"
        assert "foreign=False" not in result
