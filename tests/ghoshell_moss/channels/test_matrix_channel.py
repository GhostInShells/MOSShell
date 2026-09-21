"""Tests for matrix channel — surface tiers (cold / warm / hot).

matrix_channel owns the local + network cell governance surface. What is pinned
here is the delivery tier of each surface, not the governance verbs. All state
is warm; nothing rides the hot context band.

- nodes: state rides named_notices fragments, each diffed independently, so one
  fragment moving does not re-send the others. The text must be byte-stable
  between refreshes — a drifting field (uptime, age) would defeat the diff.
  running / exited keep a bounded tail; installed is a bare count (the catalog
  is a pull via list()).
- mesh: a bounded tail of recent cell events. The tail must be capped, and it
  must not re-emit while no new event has arrived.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ghoshell_moss.channels.matrix_channel import (
    CellAliasRegistry,
    new_mesh_channel,
    new_nodes_channel,
)
from ghoshell_moss.core.blueprint.cell import (
    AutoAcceptPolicy,
    CELL_EVENT_CHANNEL_ADDED,
    CellEvent,
    CellEventLevel,
)
from ghoshell_moss.core.blueprint.channel_builder import CommandUtil

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


def _manifest(name, category, description=""):
    return SimpleNamespace(
        name=name, category=category, installed=True, description=description,
    )


class _NodesCatalog:
    def __init__(self, nodes=None):
        self._nodes = nodes or {
            "nodes/visions/camera": _manifest("camera", "vision", description="camera stream"),
        }

    def list_nodes(self, refresh=False, paths=None, installed=None):
        return self._nodes


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
        assert meta.notice == ""  # no unnamed notice — state is all fragments now
        assert meta.named_notices["running"].startswith("1 running:")
        assert meta.named_notices["installed"] == "1"


@pytest.mark.asyncio
async def test_nodes_notice_text_is_stable_across_refreshes():
    matrix = _StubMatrix()
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        handle = _handle(_UID_A, pid=4242)
        matrix.handled[handle.address] = handle
        await runtime.refresh_metas()
        first = runtime.self_meta().named_notices

        await runtime.refresh_metas()
        second = runtime.self_meta().named_notices
        assert first == second, (
            "fragment text drifted without a state change — the facade diff "
            "would re-emit it on every refresh"
        )


@pytest.mark.asyncio
async def test_nodes_notice_reports_exit_without_age_text():
    matrix = _StubMatrix()
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        matrix.dead.append(_handle(_UID_B, pid=4243, exit_code=1))
        await runtime.refresh_metas()
        exited = runtime.self_meta().named_notices["exited"]

        assert "recently exited" in exited
        assert "exit=1" in exited
        assert "ago" not in exited, "a drifting age field defeats the facade diff"
        assert "uptime" not in exited


@pytest.mark.asyncio
async def test_nodes_installed_is_count_only():
    matrix = _StubMatrix()
    matrix.project.nodes = _NodesCatalog({
        f"nodes/n{i}": _manifest(f"node{i}", "tools", description=f"desc {i}")
        for i in range(20)
    })
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        meta = runtime.self_meta()

        assert meta.named_notices["installed"] == "20"
        # the catalog's descriptions and paths never ride the warm band
        warm = "\n".join(meta.named_notices.values())
        assert "desc " not in warm
        assert "nodes/n0" not in warm


# ---- mesh: the event tail is warm state, capped ---- #

def _net(stub):
    """Mesh channel needs ``matrix.network()`` at startup + refresh, and
    ``matrix.handled_cells()`` to prune pending names."""
    async def _network():
        return stub

    return SimpleNamespace(network=_network, handled_cells=lambda: {})


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


# ---- naming / alias ---- #


def test_alias_registry_mints_unique_names():
    reg = CellAliasRegistry()
    a = reg.reserve("node/vision/a", "vision")
    b = reg.reserve("node/vision/b", "vision")
    c = reg.reserve("node/vision/c", "vision")
    d = reg.reserve("node/camera/d", "front_camera")

    assert a == "vision", "the first allocation keeps the bare name"
    assert (b, c) == ("vision_2", "vision_3"), "duplicates suffix monotonically"
    assert d == "front_camera", "distinct bases do not collide"


def test_alias_registry_consume_is_one_shot():
    reg = CellAliasRegistry()
    reg.reserve("node/vision/a", "vision")

    assert reg.consume("node/vision/a") == "vision"
    assert reg.consume("node/vision/a") is None, "consumed at mount, not re-readable"


def test_alias_registry_prunes_dead_addresses():
    reg = CellAliasRegistry()
    reg.reserve("node/vision/a", "vision")
    reg.reserve("node/sensor/b", "sensor")

    reg.prune({"node/sensor/b"})
    assert reg.consume("node/vision/a") is None, "dead process → name pruned"
    assert reg.consume("node/sensor/b") == "sensor", "live process → name kept"


class _RunMatrix:
    def __init__(self):
        self.project = SimpleNamespace(nodes=_NodesCatalog())
        self._spawns = 0

    async def run_node(self, target, *, extra_args=None):
        self._spawns += 1
        uid = f"01J8ZZZZZZAAAAAAAAAAAAAA{self._spawns}"
        runtime = SimpleNamespace(
            address=f"node/vision/{uid}",
            cell=SimpleNamespace(
                category="vision", name="vision", fullname="node/vision",
                providing=["channel"], home="/tmp",
                event_level=CellEventLevel.INFO,
            ),
        )
        meta = SimpleNamespace(pid=1000 + self._spawns, exit_code=None, cwd="/tmp")
        return SimpleNamespace(
            runtime=runtime, address=runtime.address,
            process=SimpleNamespace(meta=meta, output=None),
        )


@pytest.mark.asyncio
async def test_run_mints_and_reports_alias():
    matrix = _RunMatrix()
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        first = await runtime.execute_command(
            "run", kwargs={"target": "nodes/visions/camera", "name": "vision"},
        )
        assert "alias=vision" in first, "the receipt reports the final name"
        assert "matrix.mesh.vision" in first, "the receipt shows the mount path"

        second = await runtime.execute_command(
            "run", kwargs={"target": "nodes/visions/camera", "name": "vision"},
        )
        assert "alias=vision_2" in second, "a duplicate base name is suffixed"


@pytest.mark.asyncio
async def test_channel_added_event_is_filtered_from_signal(monkeypatch):
    stub = _StubMesh()
    chan = new_mesh_channel(_net(stub))
    sent = []
    monkeypatch.setattr(CommandUtil, "send_signal", lambda s: sent.append(s))

    async with chan.bootstrap() as runtime:
        stub.emit(CELL_EVENT_CHANNEL_ADDED)
        stub.emit("something else")
        await runtime.refresh_metas()

        assert len(sent) == 1, "the producer's 'channel added' self-report is not a signal"
        assert sent[0].name == "cell_event"

        # the ring still records the filtered event — warm data, not an attention signal
        assert "channel added" in runtime.self_meta().notice
