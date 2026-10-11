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

import threading
from types import SimpleNamespace

import pytest

from ghoshell_moss.channels.matrix_channel import (
    new_mesh_channel,
    new_nodes_channel,
)
from ghoshell_moss.core.blueprint.cell import (
    AutoAcceptPolicy,
    CellAddressCodec,
    CellEvent,
    CellEventLevel,
)
from ghoshell_moss.core.blueprint.channel_builder import CommandUtil
from ghoshell_moss.core.concepts.errors import CommandError
from ghoshell_moss.core.py_channel import PyChannel

# Production uids are six characters, lowercase, letter-first (``cell.short_uid``):
# the fixtures must be that shape, or the naming / prefix paths are exercised at a
# scale and alphabet no real address has.
_UID_A = "a1b2c3"
_UID_B = "d4e5f6"


def _handle(uid: str, pid: int, *, exit_code=None):
    runtime = SimpleNamespace(
        address=f"node/vision/{uid}",
        alias="vision",
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
    def __init__(self, nodes=None, spawned=None):
        self._nodes = nodes or {
            "nodes/visions/camera": _manifest("camera", "vision", description="camera stream"),
        }
        # The naming authority's live map: address → alias, emptied when a cell exits.
        self._spawned = spawned or {}

    def list_nodes(self, refresh=False, paths=None, installed=None):
        return self._nodes

    def spawned_nodes(self):
        return dict(self._spawned)


class _RecordingCatalog(_NodesCatalog):
    """Records (refresh, thread) per call — the channel must not scan on the loop."""

    def __init__(self, nodes=None):
        super().__init__(nodes)
        self.calls: list[tuple[bool, str]] = []

    def list_nodes(self, refresh=False, paths=None, installed=None):
        self.calls.append((refresh, threading.current_thread().name))
        return super().list_nodes(refresh=refresh, paths=paths, installed=installed)


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
        self._on_channel_provided = None
        self._policy = AutoAcceptPolicy(local=True, foreign=False)

    def on_event(self, callback):
        self._on_event = callback
        return lambda: None

    def on_channel_provided(self, callback):
        self._on_channel_provided = callback
        return lambda: None

    def channel_proxies(self):
        return {}

    def view(self):
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


class _PathRecordingCatalog(_NodesCatalog):
    """Records the `paths` each scan was given — `list(path=...)` passes a scan root."""

    def __init__(self, nodes=None):
        super().__init__(nodes)
        self.scanned: list = []

    def list_nodes(self, refresh=False, paths=None, installed=None):
        self.scanned.append(paths)
        return super().list_nodes(refresh=refresh, paths=paths, installed=installed)


@pytest.mark.asyncio
async def test_nodes_list_resolves_a_relative_path_against_the_project_root(tmp_path):
    """`list(path=...)` is a project-relative scan root — the channel makes it absolute.

    Empty `path` scans the default roots; a relative path is anchored at the project
    root; an absolute path is used as given.
    """
    matrix = _StubMatrix()
    matrix.project = SimpleNamespace(nodes=_PathRecordingCatalog(), root=tmp_path)
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("list")
        assert matrix.project.nodes.scanned[-1] is None, "no path → default roots"

        await runtime.execute_command("list", kwargs={"path": "nodes/screens"})
        relative = matrix.project.nodes.scanned[-1]
        assert relative == [(tmp_path / "nodes/screens").resolve()]

        await runtime.execute_command("list", kwargs={"path": str(tmp_path / "nodes")})
        assert matrix.project.nodes.scanned[-1] == [tmp_path / "nodes"], "absolute → as given"


@pytest.mark.asyncio
async def test_nodes_catalog_is_warmed_off_loop_at_startup():
    """The warm band reads the catalog synchronously, so the scan must not land there.

    Every read on the notice path has to be a cached, instant one — the first
    filesystem scan is pulled forward to startup and runs off the event loop.
    """
    matrix = _StubMatrix()
    matrix.project.nodes = catalog = _RecordingCatalog()
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        assert catalog.calls, "startup must prime the catalog"
        assert catalog.calls[0][1] != threading.current_thread().name, (
            "the scan ran on the event loop thread — the notice path would have blocked it"
        )
        await runtime.refresh_metas()
        assert all(not refresh for refresh, _ in catalog.calls), (
            "only startup may scan; the notice path reads the cache"
        )


# ---- naming: an alias is a live name; pid and address are the handles ---- #

@pytest.mark.asyncio
async def test_status_calls_an_aliased_live_cell_running():
    """An alias names a live cell. It resolves through the naming authority, never
    as a key of the address-keyed ledger — and not being a key is not evidence of
    death."""
    matrix = _StubMatrix()
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        handle = _handle(_UID_A, pid=4242)
        matrix.handled[handle.address] = handle
        matrix.project.nodes = _NodesCatalog(spawned={handle.address: "vision"})

        by_alias = await runtime.execute_command("status", kwargs={"target": "vision"})
        assert "running" in by_alias
        assert f"[vision ({handle.address}) pid=4242]" in by_alias, (
            "a live row carries the name, the identity and the debug handle"
        )

        by_address = await runtime.execute_command(
            "status", kwargs={"target": handle.address}
        )
        assert "running" in by_address


@pytest.mark.asyncio
async def test_status_still_calls_a_dead_cell_dead_by_pid():
    """An exited cell answers to pid or address — never to a name: the alias dies
    with the process it named."""
    matrix = _StubMatrix()
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        matrix.dead.append(_handle(_UID_A, pid=4242, exit_code=1))

        out = await runtime.execute_command("status", kwargs={"target": "4242"})
        assert "dead" in out, "pid is the debug handle and outlives the process"

        out = await runtime.execute_command(
            "status", kwargs={"target": f"node/vision/{_UID_A}"}
        )
        assert "dead" in out, "the address is the identity and always works"


@pytest.mark.asyncio
async def test_an_exited_cell_answers_to_pid_or_whole_address_only():
    """Neither a minted name nor a shorthand cut from the address reaches a cell that
    has exited — the address has to be copied whole."""

    matrix = _StubMatrix()
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        matrix.dead.append(_handle(_UID_A, pid=4242, exit_code=1))
        # The naming authority dropped the entry — it never names the dead.
        matrix.project.nodes = _NodesCatalog(spawned={})

        with pytest.raises(CommandError, match="not found"):
            await runtime.execute_command("status", kwargs={"target": "vision_2"})
        with pytest.raises(CommandError, match="not found"):
            # a uid prefix resolves a *live* cell; a dead one is not addressed that way
            await runtime.execute_command("status", kwargs={"target": _UID_A})


@pytest.mark.asyncio
async def test_read_output_takes_pid_not_a_name():
    matrix = _StubMatrix()
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        handle = _handle(_UID_A, pid=4242, exit_code=1)
        matrix.dead.append(handle)
        matrix.project.nodes = _NodesCatalog(spawned={handle.address: "vision"})

        out = await runtime.execute_command(
            "read_output", kwargs={"target": "4242", "stream": "stderr"}
        )
        assert "no capture buffer" in out, "the pid resolved to the exited cell"
        assert f"[node/vision/{_UID_A} pid=4242]" in out, (
            "an exited cell's row shows identity + pid, not the name it had"
        )


# ---- mesh: the event tail is warm state, capped ---- #

def _net(stub):
    """Mesh channel needs ``matrix.network()`` at startup, and
    ``matrix.project.nodes.spawned_nodes()`` for branch-name resolution."""
    async def _network():
        return stub

    return SimpleNamespace(
        network=_network,
        handled_cells=lambda: {},
        project=SimpleNamespace(nodes=_NodesCatalog()),
        send_signal_to_ghost=lambda signal: None,
    )


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


@pytest.mark.asyncio
async def test_a_branch_is_named_by_its_alias_alone():
    """The branch name doubles as the CTML path segment (``matrix.mesh.<name>``).

    It is a lookup key, not a label: decorating it — e.g. spelling the address next
    to the alias — makes the branch unresolvable. The address surfaces in the notice
    instead, so the model still reads both namespaces.
    """
    stub = _StubMesh()
    matrix = _net(stub)
    address = f"node/push/{_UID_A}"
    matrix.project.nodes.spawned_nodes = lambda: {address: "push"}
    stub.channel_proxies = lambda: {address: PyChannel(name="push")}
    chan = new_mesh_channel(matrix)
    async with chan.bootstrap() as runtime:
        assert list(runtime.virtual_sub_channels()) == ["push"]

        await runtime.refresh_metas()
        notice = runtime.self_meta().notice
        assert "push" in notice and address in notice


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


class _AcceptMesh(_StubMesh):
    """记录 accept/reject 收到的 raw address, 供契约断言."""

    def __init__(self):
        super().__init__()
        self._accepted: list[str] = []
        self._rejected: list[str] = []
        self._cells = {f"node/vision/{_UID_A}": object()}

    def view(self):
        return self._cells

    def channel_proxies(self):
        return {}

    async def accept(self, address, *, lookup=False):
        self._accepted.append(address)

    async def reject(self, address):
        self._rejected.append(address)


@pytest.mark.asyncio
async def test_accept_resolves_normalized_address_to_raw():
    stub = _AcceptMesh()
    chan = new_mesh_channel(_net(stub))
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        normalized = CellAddressCodec(f"node/vision/{_UID_A}").normalized
        await runtime.execute_command("accept", kwargs={"target": normalized})

        assert stub._accepted == [f"node/vision/{_UID_A}"], (
            "accept takes the display form (normalized address) and resolves it "
            "to the raw network identity"
        )


# ---- naming / alias ---- #


class _RunMatrix:
    def __init__(self):
        self.project = SimpleNamespace(nodes=_NodesCatalog())
        self._spawns = 0

    async def run_node(self, target, *, extra_args=None, alias=None):
        self._spawns += 1
        uid = f"01J8ZZZZZZAAAAAAAAAAAAAA{self._spawns}"
        runtime = SimpleNamespace(
            address=f"node/vision/{uid}",
            alias=alias or "vision",
            cell=SimpleNamespace(
                category="vision", name="vision", fullname="node/vision",
                providing=["channel"], home="/tmp",
                event_level=CellEventLevel.INFO,
            ),
        )
        meta = SimpleNamespace(pid=1000 + self._spawns, exit_code=None, cwd="/tmp")
        # NodeManager records the minted name at spawn; spawn() → spawned_nodes().
        self.project.nodes._spawned[runtime.address] = runtime.alias
        return SimpleNamespace(
            runtime=runtime, address=runtime.address,
            process=SimpleNamespace(meta=meta, output=None),
        )


@pytest.mark.asyncio
async def test_run_reports_alias_and_mount_path():
    matrix = _RunMatrix()
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command(
            "run", kwargs={"target": "nodes/visions/camera", "alias": "vision"},
        )
        assert "matrix.mesh.vision" in result, "the receipt shows the mount path"


@pytest.mark.asyncio
async def test_empty_content_event_is_pure_refetch(monkeypatch):
    stub = _StubMesh()
    chan = new_mesh_channel(_net(stub))
    sent = []
    monkeypatch.setattr(CommandUtil, "send_signal", lambda s: sent.append(s))

    async with chan.bootstrap() as runtime:
        stub.emit("")                       # empty content = refetch hint, no signal
        stub.emit("something else")
        await runtime.refresh_metas()

        assert len(sent) == 1, "empty content is a refetch hint, not a signal"
        assert sent[0].name == "cell_event"

        # the empty-content event never entered the ring — only one real event remains
        assert "recent events (1):" in runtime.self_meta().notice


@pytest.mark.asyncio
async def test_notice_rows_carry_name_identity_and_pid():
    """One shape for every row, so a handle in view is a handle that works:
    ``alias (address) pid=N`` while alive, ``address pid=N`` once it is not."""
    matrix = _StubMatrix()
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        live = _handle(_UID_A, pid=4242)
        dead = _handle(_UID_B, pid=4243, exit_code=1)
        matrix.handled[live.address] = live
        matrix.dead.append(dead)
        matrix.project.nodes = _NodesCatalog(spawned={live.address: "vision"})

        await runtime.refresh_metas()
        fragments = runtime.self_meta().named_notices

        assert f"vision ({live.address}) pid=4242" in fragments["running"]
        assert f"{dead.address} pid=4243" in fragments["exited"]
        assert f"vision ({dead.address})" not in fragments["exited"], (
            "a name does not outlive its cell"
        )
        assert "read_output(target=4243)" in fragments["exited"], (
            "the debug pointer is the pid"
        )


@pytest.mark.asyncio
async def test_branch_table_is_capped():
    """Mesh is a warm surface, and its branch table is a tail like any other:
    bounded, with the overspill pointing at the pull path."""
    stub = _StubMesh()
    matrix = _net(stub)
    addresses = [f"node/push/a{i:02d}bcd" for i in range(12)]
    spawned = {addr: f"push_{i}" for i, addr in enumerate(addresses)}
    matrix.project.nodes.spawned_nodes = lambda: dict(spawned)
    stub.channel_proxies = lambda: {addr: PyChannel(name="push_0") for addr in addresses}

    chan = new_mesh_channel(matrix)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        notice = runtime.self_meta().notice

        assert "branches (12):" in notice, "the count is the whole truth, not the window"
        assert "...+4 more, status() for full list" in notice
        assert "push_9" not in notice, "rows past the window are not replayed"


@pytest.mark.asyncio
async def test_a_live_cell_answers_to_the_pid_its_row_prints():
    """Every handle a row prints must resolve — otherwise the notice advertises a
    command that fails. pid is a debug handle for a live cell as much as for an
    exited one."""
    matrix = _StubMatrix()
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        handle = _handle(_UID_A, pid=4242)
        matrix.handled[handle.address] = handle
        matrix.project.nodes = _NodesCatalog(spawned={handle.address: "vision"})

        await runtime.refresh_metas()
        assert "pid=4242" in runtime.self_meta().named_notices["running"]

        by_pid = await runtime.execute_command("status", kwargs={"target": "4242"})
        assert "running" in by_pid, "the handle the row advertised must work"


@pytest.mark.asyncio
async def test_a_shorthand_never_resolves_a_live_cell_either():
    """What a row shows is what a call takes: address, pid, alias. A name segment or
    a uid prefix is not a form this face knows — dropping the loose matcher is what
    removes the "which shape does this command want" inference."""
    matrix = _StubMatrix()
    chan = new_nodes_channel(matrix)
    async with chan.bootstrap() as runtime:
        handle = _handle(_UID_A, pid=4242)
        matrix.handled[handle.address] = handle
        matrix.project.nodes = _NodesCatalog(spawned={})  # nothing was named

        for shorthand in ("vision", _UID_A, "node/vision"):
            with pytest.raises(CommandError, match="not found"):
                await runtime.execute_command(
                    "status", kwargs={"target": shorthand}
                )

        # and the three printed forms all work
        for printed in (handle.address, "4242"):
            out = await runtime.execute_command("status", kwargs={"target": printed})
            assert "running" in out


@pytest.mark.asyncio
async def test_accept_takes_only_the_forms_the_mesh_face_prints():
    """The mesh notice and its events print the address and the branch name — those are
    the forms. A shorthand cut from an address (name segment / uid prefix) is typing
    convenience, and is not a form this face knows."""
    stub = _AcceptMesh()
    chan = new_mesh_channel(_net(stub))
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()

        for shorthand in ("vision", _UID_A, "node/vision"):
            with pytest.raises(CommandError, match="does not match any network cell"):
                await runtime.execute_command("accept", kwargs={"target": shorthand})

        await runtime.execute_command(
            "accept", kwargs={"target": f"node/vision/{_UID_A}"}
        )
        assert stub._accepted == [f"node/vision/{_UID_A}"]
