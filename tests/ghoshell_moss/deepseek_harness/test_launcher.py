"""DshLauncher 远程流摄取行为证据 — $events emit/waterfall/ready 分派 + workspace RPC 拉取.

session/follow 摄取面:
- `event` 帧 → SessionEvent.from_dict → session.accept_session_event.
- `assistant-stream` start/chunk 帧 → 合成 assistant/chunk 事件 (turn/step 由 start 补帧).
- snapshot / end 帧静默忽略.
- `_open_follow_stream` 的 open 载荷钉住 wire 契约 ({args:{address, assistantStream:true}}).
"""

import asyncio
import json

import pytest

from ghoshell_moss.deepseek_harness.launcher import DshLauncher, DshLauncherConfig
from ghoshell_moss.deepseek_harness.session import DshSession
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


def _follow_item(session_id: str, value: dict) -> str:
    return json.dumps({"type": "item", "streamId": f"moss-follow-{session_id}", "value": value})


class _DummyClient:
    """create_session 构造 DshSession 不触碰 client — 哑元即可."""

    pass


class _FakeWs:
    """记录 send 载荷的哑元 WS, 用于钉住开流 wire 契约."""

    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, raw: str):
        self.sent.append(json.loads(raw))


async def _drain(session: DshSession) -> None:
    """等 session 消费 task 处理完已入队帧 (白盒: 看内部队列排空)."""
    for _ in range(1000):
        if not session._queue:
            return
        await asyncio.sleep(0)
    raise AssertionError("session consume queue did not drain")


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


# ---- session/follow 摄取面 ---- #


@pytest.mark.asyncio
async def test_open_follow_stream_sends_request_payload():
    """开流载荷钉住 wire 契约: follow(request, signal) 的命名 args 只有 request 字段."""
    launcher = _make_launcher()
    ws = _FakeWs()
    await launcher._open_follow_stream(ws, "s1")
    assert ws.sent == [{
        "type": "open",
        "streamId": "moss-follow-s1",
        "endpoint": "session/follow",
        "payload": {"args": {"request": {"address": {"kind": "session", "sessionId": "s1"}, "assistantStream": True}}},
    }]


@pytest.mark.asyncio
async def test_follow_event_frame_feeds_session():
    """`event` 帧 → SessionEvent.from_dict → session.accept_session_event, 事件名/载荷保真."""
    launcher = _make_launcher()
    launcher.client = _DummyClient()
    session = launcher.create_session("s1")
    seen: list[tuple[str, dict]] = []

    async def on_event(event) -> None:
        seen.append((event.meta.type, event.data))

    session.on_session_event("*", on_event)
    async with session:
        await launcher._dispatch_raw_frame(_follow_item("s1", {
            "type": "event",
            "event": {"type": "tool/call", "seq": 5, "time": 100,
                      "data": {"callId": "c1", "name": "moss_interleaved_ctml"}},
        }))
        await _drain(session)

    assert seen == [("tool/call", {"callId": "c1", "name": "moss_interleaved_ctml"})]


@pytest.mark.asyncio
async def test_follow_assistant_stream_synthesizes_chunk():
    """assistant-stream chunk 帧 → 合成 assistant/chunk (turn/step 由 start 补帧), 喂给 session."""
    launcher = _make_launcher()
    launcher.client = _DummyClient()
    session = launcher.create_session("s1")
    seen: list[dict] = []

    async def on_chunk(event) -> None:
        seen.append(event.data)

    session.on_session_event("assistant/chunk", on_chunk)
    async with session:
        await launcher._dispatch_raw_frame(_follow_item("s1", {
            "type": "assistant-stream",
            "frame": {"type": "start", "turn": 2, "step": 3},
        }))
        await launcher._dispatch_raw_frame(_follow_item("s1", {
            "type": "assistant-stream",
            "frame": {"type": "chunk", "time": 200,
                      "chunk": {"type": "text-delta", "index": 0, "text": "<say>"}},
        }))
        await _drain(session)

    assert seen == [{"turn": 2, "step": 3, "chunk": {"type": "text-delta", "index": 0, "text": "<say>"}}]


@pytest.mark.asyncio
async def test_follow_snapshot_and_end_are_ignored():
    """snapshot (历史页) 与 end (终止标记) 不喂 session — live 流程只消费 event/assistant-stream."""
    launcher = _make_launcher()
    launcher.client = _DummyClient()
    session = launcher.create_session("s1")
    seen: list[str] = []

    async def on_any(event) -> None:
        seen.append(event.meta.type)

    session.on_session_event("*", on_any)
    async with session:
        await launcher._dispatch_raw_frame(_follow_item("s1", {"type": "snapshot", "cursor": 0, "records": []}))
        await launcher._dispatch_raw_frame(_follow_item("s1", {
            "type": "assistant-stream",
            "frame": {"type": "end"},
        }))
        await _drain(session)

    assert seen == []
