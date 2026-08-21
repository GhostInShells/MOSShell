"""DshSession 状态面行为证据 — 运行态镜像事件 + on_session_event* 事件分派.

覆盖:
- 初始态: 新建 session 假设 idle, when_idle 立即返回, when_running 阻塞.
- host/session-status 帧翻转 running ⇄ idle 镜像事件.
- on_session_event_model(AssistantMessageEvent) 收每步事件, token_usage 属性累计会话量.
- on_session_event (raw) 收原始 SessionEvent 信封, 与强类型回调并存.
- disposer 解绑后不再收到后续事件.
- 不同事件名各走各的 handler, 未注册事件名静默忽略.
"""

import asyncio

import pytest

from ghoshell_moss.deepseek_harness.session import DshSession, WILDCARD_EVENT
from ghoshell_moss.deepseek_harness.types import sessions
from ghoshell_moss.deepseek_harness.types.events import HostFrame, MuxFrame
from ghoshell_moss.deepseek_harness.types.session_events import (
    AssistantMessageEvent,
    ContentBlock,
    EpochHeader,
    Message,
    RequestHeader,
    SessionEvent,
    TokenUsage,
    TurnStart,
)


class _DummyClient:
    """帧消费路径不触碰 client — 用哑元即可构造."""

    pass


class _RpcClient:
    """pull 路径的哑元 client — call() 返回预置 value, plugin_call() 返回预置 plugin JSON."""

    def __init__(
        self,
        models_value=None,
        history_value=None,
        session_list_value=None,
        plugin_values=None,
    ):
        self._models_value = models_value
        self._history_value = history_value
        self._session_list_value = session_list_value
        self._plugin_values = plugin_values or {}
        self.calls: list[str] = []
        self.plugin_calls: list[tuple[str, dict]] = []

    async def call(self, method, params, value_cls):
        self.calls.append(method)
        if method == "session.models":
            return self._models_value
        if method == "session.history":
            return self._history_value
        if method == "session.list":
            return self._session_list_value
        raise AssertionError(f"unexpected rpc {method}")

    async def plugin_call(self, path, payload=None):
        self.plugin_calls.append((path, payload or {}))
        return self._plugin_values.get(path, {})


def _usage_frame(input_tokens: int, output_tokens: int) -> MuxFrame:
    model = AssistantMessageEvent(
        turn=1,
        step=0,
        message=Message(content=[ContentBlock(type="text", text="hi")]),
        usage=TokenUsage(inputTokens=input_tokens, outputTokens=output_tokens),
    )
    # 真实 mux 流经 SessionEvent.from_dict 从扁平信封读到 meta.type;
    # 直接构造默认是空串, 这里补上判别符模拟真实信封.
    model.meta.type = model.event_type()
    return MuxFrame(type="session/event", sessionId="s1", event=model.to_session_event())


def _request_header_frame(system: str, session_id: str = "s1") -> MuxFrame:
    model = RequestHeader(header=EpochHeader(system=system), reason="initial")
    model.meta.type = model.event_type()
    return MuxFrame(type="session/event", sessionId=session_id, event=model.to_session_event())


def _session_added_frame(cwd: str, agent_preset: str, session_id: str = "s1") -> HostFrame:
    return HostFrame(
        type="host/session-added",
        sessionId=session_id,
        cwd=cwd,
        agentPreset=agent_preset,
    )


async def _drain(session: DshSession) -> None:
    """等 session 消费 task 处理完已入队帧 (白盒: 看内部队列排空)."""
    for _ in range(1000):
        if not session._queue:
            return
        await asyncio.sleep(0)
    raise AssertionError("session consume queue did not drain")


@pytest.mark.asyncio
async def test_initial_state_is_idle():
    session = DshSession(session_id="s1", client=_DummyClient())
    async with session:
        assert session.running is False
        # idle 镜像事件初始置位 → when_idle 立即返回.
        await session.when_idle()
        # running 事件未置位 → when_running 阻塞直到超时.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(session.when_running(), 0.05)


@pytest.mark.asyncio
async def test_status_frames_flip_running_idle():
    session = DshSession(session_id="s1", client=_DummyClient())
    async with session:
        session.accept_frame(
            HostFrame(type="host/session-status", sessionId="s1", running=True)
        )
        await asyncio.wait_for(session.when_running(), 1)
        assert session.running is True

        session.accept_frame(
            HostFrame(type="host/session-status", sessionId="s1", running=False)
        )
        await asyncio.wait_for(session.when_idle(), 1)
        assert session.running is False


@pytest.mark.asyncio
async def test_usage_accumulates_and_event_fires():
    """on_session_event_model(AssistantMessageEvent) 收每步事件, token_usage 属性累计会话量."""
    session = DshSession(session_id="s1", client=_DummyClient())
    got = asyncio.Event()
    seen: list[tuple[int, int]] = []

    async def on_assistant(event: AssistantMessageEvent) -> None:
        seen.append((event.usage.inputTokens, event.usage.outputTokens))
        got.set()

    session.on_session_event_model(AssistantMessageEvent, on_assistant)
    async with session:
        session.accept_frame(_usage_frame(10, 5))
        await asyncio.wait_for(got.wait(), 1)
        got.clear()
        session.accept_frame(_usage_frame(20, 3))
        await asyncio.wait_for(got.wait(), 1)

    # 回调收每步事件 (非累计); 会话累计量经 token_usage 属性读.
    assert seen == [(10, 5), (20, 3)]
    assert session.token_usage.inputTokens == 30
    assert session.token_usage.outputTokens == 8


@pytest.mark.asyncio
async def test_raw_event_handler_receives_envelope():
    """on_session_event (raw) 收原始 SessionEvent 信封, 与强类型回调并存."""
    session = DshSession(session_id="s1", client=_DummyClient())
    raw_seen: list[str] = []
    typed_seen: list[int] = []

    async def on_raw(event: SessionEvent) -> None:
        raw_seen.append(event.meta.type)

    async def on_typed(event: AssistantMessageEvent) -> None:
        typed_seen.append(event.usage.inputTokens)

    session.on_session_event("assistant/message", on_raw)
    session.on_session_event_model(AssistantMessageEvent, on_typed)
    async with session:
        session.accept_frame(_usage_frame(5, 0))
        await _drain(session)

    assert raw_seen == ["assistant/message"]
    assert typed_seen == [5]


@pytest.mark.asyncio
async def test_event_handler_disposer_removes():
    """disposer 解绑后不再收到同事件名的后续事件."""
    session = DshSession(session_id="s1", client=_DummyClient())
    got = asyncio.Event()
    seen: list[int] = []

    async def on_assistant(event: AssistantMessageEvent) -> None:
        seen.append(event.usage.inputTokens)
        got.set()

    remove = session.on_session_event_model(AssistantMessageEvent, on_assistant)
    async with session:
        session.accept_frame(_usage_frame(1, 0))
        await asyncio.wait_for(got.wait(), 1)

        remove()
        got.clear()
        session.accept_frame(_usage_frame(2, 0))
        # 解绑后不再收到 → 等待超时.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(got.wait(), 0.05)

    assert seen == [1]


@pytest.mark.asyncio
async def test_event_dispatch_routes_by_event_name():
    """不同事件名各走各的 handler, 事件互不串扰."""
    session = DshSession(session_id="s1", client=_DummyClient())
    assistant_seen: list[int] = []
    header_seen: list[str] = []

    async def on_assistant(event: AssistantMessageEvent) -> None:
        assistant_seen.append(event.usage.inputTokens)

    async def on_header(event: RequestHeader) -> None:
        header_seen.append(event.header.system)

    session.on_session_event_model(AssistantMessageEvent, on_assistant)
    session.on_session_event_model(RequestHeader, on_header)
    async with session:
        session.accept_frame(_usage_frame(9, 1))
        session.accept_frame(_request_header_frame("prompt: p"))
        session.accept_frame(_request_header_frame("prompt: q"))
        await _drain(session)

    assert assistant_seen == [9]
    assert header_seen == ["prompt: p", "prompt: q"]


@pytest.mark.asyncio
async def test_wildcard_event_handler_receives_all_events():
    """on_session_event(WILDCARD_EVENT) 注册 catch-all — 每个 session/event 帧都派发, 不挑事件名."""
    session = DshSession(session_id="s1", client=_DummyClient())
    seen: list[str] = []

    async def on_any(event: SessionEvent) -> None:
        seen.append(event.meta.type)

    session.on_session_event(WILDCARD_EVENT, on_any)
    async with session:
        session.accept_frame(_usage_frame(5, 0))
        session.accept_frame(_request_header_frame("prompt: p"))
        await _drain(session)

    assert seen == ["assistant/message", "request/header"]


@pytest.mark.asyncio
async def test_wildcard_coexists_with_exact_handlers():
    """catch-all 与精确名 handler 并存: 同一事件两条消费路径都收, 互不干扰."""
    session = DshSession(session_id="s1", client=_DummyClient())
    wildcard_seen: list[str] = []
    typed_seen: list[int] = []

    async def on_any(event: SessionEvent) -> None:
        wildcard_seen.append(event.meta.type)

    async def on_assistant(event: AssistantMessageEvent) -> None:
        typed_seen.append(event.usage.inputTokens)

    session.on_session_event(WILDCARD_EVENT, on_any)
    session.on_session_event_model(AssistantMessageEvent, on_assistant)
    async with session:
        session.accept_frame(_usage_frame(7, 0))
        await _drain(session)

    assert wildcard_seen == ["assistant/message"]
    assert typed_seen == [7]


@pytest.mark.asyncio
async def test_unknown_event_type_safely_ignored():
    """未注册事件名的 session/event 帧静默忽略, 不炸消费循环."""
    session = DshSession(session_id="s1", client=_DummyClient())
    got = asyncio.Event()
    seen: list[int] = []

    async def on_assistant(event: AssistantMessageEvent) -> None:
        seen.append(event.usage.inputTokens)
        got.set()

    session.on_session_event_model(AssistantMessageEvent, on_assistant)
    async with session:
        turn = TurnStart(turn=1)
        turn.meta.type = turn.event_type()
        session.accept_frame(
            MuxFrame(type="session/event", sessionId="s1", event=turn.to_session_event())
        )
        session.accept_frame(_usage_frame(5, 0))
        await asyncio.wait_for(got.wait(), 1)

    assert seen == [5]


@pytest.mark.asyncio
async def test_instruction_pulls_plugin_route():
    client = _RpcClient(
        plugin_values={
            "/moss-api/ghost/dolores/session/instruction": {"instruction": "prompt: full"},
        },
    )
    session = DshSession(session_id="s1", client=client)
    async with session:
        assert await session.instruction() == "prompt: full"
    assert client.plugin_calls == [
        ("/moss-api/ghost/dolores/session/instruction", {"sessionId": "s1"}),
    ]


@pytest.mark.asyncio
async def test_instruction_returns_none_when_plugin_has_no_instruction():
    client = _RpcClient(plugin_values={})
    session = DshSession(session_id="s1", client=client)
    async with session:
        assert await session.instruction() is None


@pytest.mark.asyncio
async def test_surface_messages_pulls_plugin_route():
    raw_message = {"id": "m1", "role": "user", "content": [{"type": "text", "text": "hi"}]}
    client = _RpcClient(
        plugin_values={
            "/moss-api/ghost/dolores/session/surface": {"messages": [raw_message]},
        },
    )
    session = DshSession(session_id="s1", client=client)
    async with session:
        messages = await session.surface_messages()
    assert len(messages) == 1
    assert messages[0].role == "user"
    assert messages[0].content[0].text == "hi"
    assert client.plugin_calls == [
        ("/moss-api/ghost/dolores/session/surface", {"sessionId": "s1"}),
    ]


@pytest.mark.asyncio
async def test_model_selection_pulls_and_caches():
    models_value = sessions.SessionModels(
        current=sessions.ModelSelection(provider="deepseek", model="v4", reasoningEffort="high"),
        routable=True,
    )
    client = _RpcClient(models_value=models_value)
    session = DshSession(session_id="s1", client=client)
    async with session:
        sel = await session.model_selection()
        assert (sel.provider, sel.model, sel.reasoningEffort) == ("deepseek", "v4", "high")
        assert await session.routable() is True
        # 缓存命中: 第二次不再发 RPC.
        await session.model_selection()
        await session.routable()
        assert client.calls.count("session.models") == 1


@pytest.mark.asyncio
async def test_model_selection_force_repulls():
    client = _RpcClient(
        models_value=sessions.SessionModels(
            current=sessions.ModelSelection(provider="a", model="m1"),
            routable=True,
        ),
    )
    session = DshSession(session_id="s1", client=client)
    async with session:
        await session.model_selection()
        client._models_value = sessions.SessionModels(
            current=sessions.ModelSelection(provider="b", model="m2"),
            routable=False,
        )
        sel = await session.model_selection(force=True)
        assert sel.model == "m2"
        assert client.calls.count("session.models") == 2


@pytest.mark.asyncio
async def test_cwd_and_preset_mirror_session_added():
    session = DshSession(session_id="s1", client=_DummyClient())
    async with session:
        session.accept_frame(_session_added_frame("/tmp/proj", "minimal"))
        await _drain(session)
        assert await session.cwd() == "/tmp/proj"
        assert await session.agent_preset() == "minimal"


@pytest.mark.asyncio
async def test_cwd_and_preset_force_pull_session_list():
    list_value = sessions.SessionListValue(
        items=[sessions.SessionSummary(sessionId="s1", cwd="/tmp/x", agentPreset="standard")],
    )
    client = _RpcClient(session_list_value=list_value)
    session = DshSession(session_id="s1", client=client)
    async with session:
        assert await session.cwd(force=True) == "/tmp/x"
        # force 拉一次同时填充 cwd + agent_preset, 后者命中缓存.
        assert await session.agent_preset() == "standard"
        assert client.calls.count("session.list") == 1
