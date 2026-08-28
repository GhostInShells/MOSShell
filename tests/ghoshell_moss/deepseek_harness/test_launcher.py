"""DshLauncher host 级 workspace 镜像行为证据 — changed/removed 采样 + 冷锚基线 + path 解析."""

import json

import pytest

from ghoshell_moss.deepseek_harness.launcher import DshLauncher, DshLauncherConfig
from ghoshell_moss.deepseek_harness.types import domains
from ghoshell_moss.deepseek_harness.types.events import HostFrame, MuxFrame
from ghoshell_moss.deepseek_harness.types.nouns import WorkspaceView


class _WsClient:
    """workspace.list 拉取面的哑元 client."""

    def __init__(self, value: domains.WorkspaceListValue):
        self._value = value
        self.calls: list[str] = []

    async def workspace_list(self) -> domains.WorkspaceListValue:
        self.calls.append("workspace.list")
        return self._value


def _ws(workspace_id: str, path: str, title: str = "t") -> WorkspaceView:
    return WorkspaceView(workspaceId=workspace_id, path=path, title=title)


def _make_launcher() -> DshLauncher:
    return DshLauncher(DshLauncherConfig())


def test_workspace_mirror_upserts_and_removes():
    launcher = _make_launcher()
    launcher._mirror_workspace(HostFrame(type="host/workspace-changed", workspace=_ws("w1", "/tmp/a")))
    launcher._mirror_workspace(HostFrame(type="host/workspace-changed", workspace=_ws("w2", "/tmp/b")))
    assert [w.workspaceId for w in launcher._workspaces.values()] == ["w1", "w2"]
    launcher._mirror_workspace(HostFrame(type="host/workspace-removed", workspaceId="w1"))
    assert [w.workspaceId for w in launcher._workspaces.values()] == ["w2"]


@pytest.mark.asyncio
async def test_workspaces_force_pulls_baseline():
    launcher = _make_launcher()
    client = _WsClient(domains.WorkspaceListValue(items=[_ws("w1", "/tmp/a")]))
    launcher.client = client
    ws = await launcher.workspaces(force=True)
    assert [w.workspaceId for w in ws] == ["w1"]
    assert client.calls == ["workspace.list"]


@pytest.mark.asyncio
async def test_workspaces_mirror_avoids_pull():
    launcher = _make_launcher()
    client = _WsClient(domains.WorkspaceListValue(items=[]))
    launcher.client = client
    launcher._mirror_workspace(HostFrame(type="host/workspace-changed", workspace=_ws("w1", "/tmp/a")))
    ws = await launcher.workspaces()
    assert [w.workspaceId for w in ws] == ["w1"]
    assert client.calls == []


@pytest.mark.asyncio
async def test_workspace_for_path_resolves():
    launcher = _make_launcher()
    launcher._mirror_workspace(HostFrame(type="host/workspace-changed", workspace=_ws("w1", "/tmp/a", title="A")))
    found = await launcher.workspace_for_path("/tmp/a")
    assert found is not None and found.title == "A"
    assert await launcher.workspace_for_path("/tmp/nonexistent") is None


@pytest.mark.asyncio
async def test_dispatch_frame_dedups_payload_type():
    """mux 下行帧判别符同时放 method 与 payload.type (dsh fullFrame) — 解析须去重不崩.

    `_dispatch_raw_frame` 是帧摄取原语 (WS loop 内部), 此处验证其协议承诺: 不去重会
    `MuxFrame(type=method, **payload)` 撞车 TypeError, 静默杀死整条事件流.
    """
    launcher = _make_launcher()
    received: list[MuxFrame] = []
    launcher.on_mux_frame(lambda frame: received.append(frame) or None)
    raw = json.dumps({
        "type": "server-request",
        "rpcId": "r1",
        "method": "session/subscribed",
        "payload": {"type": "session/subscribed", "sessionId": "s1", "lastSeq": 1},
    })
    await launcher._dispatch_raw_frame(raw)
    assert len(received) == 1
    frame = received[0]
    assert isinstance(frame, MuxFrame)
    assert frame.type == "session/subscribed"
    assert frame.sessionId == "s1"
    assert frame.lastSeq == 1
