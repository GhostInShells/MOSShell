"""DshLauncher 远程流摄取行为证据 — $events emit/waterfall/ready 分派 + workspace RPC 拉取."""

import json

import pytest

from ghoshell_moss.deepseek_harness.launcher import DshLauncher, DshLauncherConfig
from ghoshell_moss.deepseek_harness.types import domains
from ghoshell_moss.deepseek_harness.types.nouns import WorkspaceView


class _WsClient:
    """workspace.list 拉取面的哑元 client."""

    def __init__(self, value: domains.WorkspaceListValue):
        self._value = value
        self.calls: list[str] = []

    async def workspace_list(self) -> domains.WorkspaceListValue:
        self.calls.append("workspace.list")
        return self._value


class _RpcClient:
    """$events/result 回话面的哑元 client — 记录 rpc(method, payload)."""

    def __init__(self):
        self.rpc_calls: list[tuple[str, dict]] = []

    async def rpc(self, method, payload):
        self.rpc_calls.append((method, payload))
        return {"ok": True}


def _ws(workspace_id: str, path: str, title: str = "t") -> WorkspaceView:
    return WorkspaceView(workspaceId=workspace_id, path=path, title=title)


def _make_launcher() -> DshLauncher:
    return DshLauncher(DshLauncherConfig())


def _item(value: dict) -> str:
    return json.dumps({"type": "item", "streamId": "moss-events", "value": value})


@pytest.mark.asyncio
async def test_dispatch_emit_routes_to_handlers():
    launcher = _make_launcher()
    seen: list[tuple[str, list]] = []
    launcher.on_remote_emit(lambda event, args: seen.append((event, args)))
    await launcher._dispatch_raw_frame(
        _item({"type": "emit", "event": "api-session/status", "args": ["s1", True]})
    )
    assert seen == [("api-session/status", ["s1", True])]


@pytest.mark.asyncio
async def test_dispatch_ready_binds_client_id():
    launcher = _make_launcher()
    await launcher._dispatch_raw_frame(
        _item({"type": "ready", "clientId": "c1", "host": {"home": "/x"}})
    )
    assert launcher._remote_client_id == "c1"


@pytest.mark.asyncio
async def test_dispatch_waterfall_sends_result():
    launcher = _make_launcher()
    launcher.client = _RpcClient()
    # 先收 ready 绑定 clientId, 再收 waterfall.
    await launcher._dispatch_raw_frame(_item({"type": "ready", "clientId": "c1", "host": {}}))
    launcher.on_remote_waterfall(lambda event, request: {"kind": "result", "value": {"ok": True}})
    await launcher._dispatch_raw_frame(
        _item({"type": "waterfall", "event": "approval/request", "eventId": "e1", "agentId": "a1", "request": {}})
    )
    assert launcher.client.rpc_calls == [
        ("$events/result", {"args": {"clientId": "c1", "eventId": "e1", "outcome": {"kind": "result", "value": {"ok": True}}}})
    ]


@pytest.mark.asyncio
async def test_dispatch_waterfall_defaults_to_next():
    """无 handler 的 waterfall 默认回 next (放行), 不炸流."""
    launcher = _make_launcher()
    launcher.client = _RpcClient()
    await launcher._dispatch_raw_frame(_item({"type": "ready", "clientId": "c1", "host": {}}))
    await launcher._dispatch_raw_frame(
        _item({"type": "waterfall", "event": "approval/request", "eventId": "e2", "agentId": "a1", "request": {}})
    )
    assert launcher.client.rpc_calls[0][0] == "$events/result"
    assert launcher.client.rpc_calls[0][1]["args"]["outcome"] == {"kind": "next"}


@pytest.mark.asyncio
async def test_workspaces_force_pulls_baseline():
    launcher = _make_launcher()
    client = _WsClient(domains.WorkspaceListValue(items=[_ws("w1", "/tmp/a")]))
    launcher.client = client
    ws = await launcher.workspaces(force=True)
    assert [w.workspaceId for w in ws] == ["w1"]
    assert client.calls == ["workspace.list"]


@pytest.mark.asyncio
async def test_workspaces_caches_after_pull():
    """0.1.5 无 workspace 推帧 — 首次 RPC 拉取后缓存, 后续命中缓存不再拉."""
    launcher = _make_launcher()
    client = _WsClient(domains.WorkspaceListValue(items=[_ws("w1", "/tmp/a")]))
    launcher.client = client
    assert [w.workspaceId for w in await launcher.workspaces()] == ["w1"]
    assert [w.workspaceId for w in await launcher.workspaces()] == ["w1"]
    assert client.calls == ["workspace.list"]


@pytest.mark.asyncio
async def test_workspace_for_path_resolves():
    launcher = _make_launcher()
    client = _WsClient(domains.WorkspaceListValue(items=[_ws("w1", "/tmp/a", title="A")]))
    launcher.client = client
    found = await launcher.workspace_for_path("/tmp/a")
    assert found is not None and found.title == "A"
    assert await launcher.workspace_for_path("/tmp/nonexistent") is None
