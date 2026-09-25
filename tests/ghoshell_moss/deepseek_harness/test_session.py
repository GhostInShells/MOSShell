"""DshSession 状态面行为证据 — 运行态镜像事件 + on_session_event* 事件分派.

覆盖:
- 初始态: 新建 session 假设 idle, when_idle 立即返回, when_running 阻塞.
- api-session/status 事件翻转 running ⇄ idle 镜像事件.
- on_session_event_model(AssistantMessageEvent) 收每步事件, token_usage 属性累计会话量.
- on_session_event (raw) 收原始 SessionEvent 信封, 与强类型回调并存.
- disposer 解绑后不再收到后续事件.
- 不同事件名各走各的 handler, 未注册事件名静默忽略.
"""

import asyncio

import pytest

from ghoshell_moss.deepseek_harness.session import DshSession, WILDCARD_EVENT
from ghoshell_moss.deepseek_harness.types import sessions
from ghoshell_moss.deepseek_harness.types.session_events import (
    AssistantMessageEvent,
    ContentBlock,
    EpochHeader,
    Message,
    RequestHeader,
    SessionEvent,
    TokenUsage,
    TurnEnd,
    TurnEndReason,
    TurnStart,
)


class _DummyClient:
    """帧消费路径不触碰 client — 用哑元即可构造."""

    pass


class _RpcClient:
    """pull 路径的哑元 client — call() 返回预置 value, plugin_call() 返回预置 plugin JSON."""

    def __init__(
        self,
        model_catalog=None,
        session_list_value=None,
        page_value=None,
        plugin_values=None,
    ):
        self._model_catalog = model_catalog
        self._session_list_value = session_list_value
        self._page_value = page_value
        self._plugin_values = plugin_values or {}
        self.calls: list[str] = []
        self.prompt_params: list = []
        self.page_params: list = []
        self.plugin_calls: list[tuple[str, dict]] = []

    async def call(self, method, params, value_cls, *, args_key="request"):
        self.calls.append(method)
        if method == "session/modelCatalog":
            return self._model_catalog
        if method == "session/list":
            return self._session_list_value
        if method == "session/page":
            self.page_params.append(params)
            return self._page_value or sessions.SessionPageValue()
        if method == "session/prompt":
            self.prompt_params.append(params)
            return sessions.SessionPromptValue(accepted=True)
        if method == "session/cancel":
            return sessions.SessionCancelValue(accepted=True)
        raise AssertionError(f"unexpected rpc {method}")

    async def plugin_call(self, path, payload=None):
        self.plugin_calls.append((path, payload or {}))
        return self._plugin_values.get(path, {})


def _usage_event(input_tokens: int, output_tokens: int) -> SessionEvent:
    model = AssistantMessageEvent(
        turn=1,
        step=0,
        message=Message(content=[ContentBlock(type="text", text="hi")]),
        usage=TokenUsage(inputTokens=input_tokens, outputTokens=output_tokens),
    )
    model.meta.type = model.event_type()
    return model.to_session_event()


def _request_header_event(system: str) -> SessionEvent:
    model = RequestHeader(header=EpochHeader(system=system), reason="initial")
    model.meta.type = model.event_type()
    return model.to_session_event()


def _session_added_summary(cwd: str, agent_preset: str, session_id: str = "s1") -> dict:
    return {"sessionId": session_id, "cwd": cwd, "agentPreset": agent_preset}


def _turn_start_event(turn: int) -> SessionEvent:
    model = TurnStart(turn=turn)
    model.meta.type = model.event_type()
    return model.to_session_event()


def _turn_end_event(turn: int, kind: str) -> SessionEvent:
    model = TurnEnd(turn=turn, reason=TurnEndReason(kind=kind))
    model.meta.type = model.event_type()
    return model.to_session_event()


def _assistant_message_event(text: str, turn: int = 1) -> SessionEvent:
    model = AssistantMessageEvent(
        turn=turn,
        step=0,
        message=Message(content=[ContentBlock(type="text", text=text)]),
    )
    model.meta.type = model.event_type()
    return model.to_session_event()


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
        session.accept_host_event("api-session/status", ["s1", True])
        await asyncio.wait_for(session.when_running(), 1)
        assert session.running is True

        session.accept_host_event("api-session/status", ["s1", False])
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
        session.accept_session_event(_usage_event(10, 5))
        await asyncio.wait_for(got.wait(), 1)
        got.clear()
        session.accept_session_event(_usage_event(20, 3))
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
        session.accept_session_event(_usage_event(5, 0))
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
        session.accept_session_event(_usage_event(1, 0))
        await asyncio.wait_for(got.wait(), 1)

        remove()
        got.clear()
        session.accept_session_event(_usage_event(2, 0))
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
        session.accept_session_event(_usage_event(9, 1))
        session.accept_session_event(_request_header_event("prompt: p"))
        session.accept_session_event(_request_header_event("prompt: q"))
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
        session.accept_session_event(_usage_event(5, 0))
        session.accept_session_event(_request_header_event("prompt: p"))
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
        session.accept_session_event(_usage_event(7, 0))
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
        session.accept_session_event(turn.to_session_event())
        session.accept_session_event(_usage_event(5, 0))
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
    catalog = sessions.ModelCatalog(
        default=sessions.ModelSelection(provider="deepseek", model="v4", reasoningEffort="high"),
        routableProviders=["deepseek"],
    )
    client = _RpcClient(model_catalog=catalog)
    session = DshSession(session_id="s1", client=client)
    async with session:
        sel = await session.model_selection()
        assert (sel.provider, sel.model, sel.reasoningEffort) == ("deepseek", "v4", "high")
        assert await session.routable() is True
        # 缓存命中: 第二次不再发 RPC.
        await session.model_selection()
        await session.routable()
        assert client.calls.count("session/modelCatalog") == 1


@pytest.mark.asyncio
async def test_model_selection_force_repulls():
    client = _RpcClient(
        model_catalog=sessions.ModelCatalog(
            default=sessions.ModelSelection(provider="a", model="m1"),
            routableProviders=["a"],
        ),
    )
    session = DshSession(session_id="s1", client=client)
    async with session:
        await session.model_selection()
        client._model_catalog = sessions.ModelCatalog(
            default=sessions.ModelSelection(provider="b", model="m2"),
            routableProviders=["b"],
        )
        sel = await session.model_selection(force=True)
        assert sel.model == "m2"
        assert client.calls.count("session/modelCatalog") == 2


@pytest.mark.asyncio
async def test_cwd_and_preset_mirror_session_added():
    session = DshSession(session_id="s1", client=_DummyClient())
    async with session:
        session.accept_host_event("api-session/added", [_session_added_summary("/tmp/proj", "minimal")])
        await _drain(session)
        assert await session.cwd() == "/tmp/proj"
        assert await session.agent_preset() == "minimal"


@pytest.mark.asyncio
async def test_cwd_force_pull_session_list():
    list_value = sessions.SessionListValue(
        items=[sessions.SessionSummary(sessionId="s1", cwd="/tmp/x")],
    )
    client = _RpcClient(session_list_value=list_value)
    session = DshSession(session_id="s1", client=client)
    async with session:
        assert await session.cwd(force=True) == "/tmp/x"
        assert client.calls.count("session/list") == 1


@pytest.mark.asyncio
async def test_snapshot_sets_preset_cwd_and_cursor():
    """follow 快照 header 喂 cwd/agentPreset, cursor 供 history() 作 throughSeq."""
    client = _RpcClient()
    session = DshSession(session_id="s1", client=client)
    async with session:
        session.accept_follow_snapshot({"cwd": "/x", "agentPreset": "standard"}, 25)
        assert await session.agent_preset() == "standard"
        assert await session.cwd() == "/x"
        events = await session.history(max_messages=3)
        assert events == []
        assert client.page_params[0].throughSeq == 25
        assert client.page_params[0].address.sessionId == "s1"
        assert client.page_params[0].maxMessages == 3


@pytest.mark.asyncio
async def test_history_requires_cursor():
    client = _RpcClient()
    session = DshSession(session_id="s1", client=client)
    async with session:
        with pytest.raises(RuntimeError):
            await session.history()


# ---- 单轮对话 (run) 与标准中断 (cancel) ---- #


@pytest.mark.asyncio
async def test_run_returns_single_turn_result():
    """run() 阻塞到本轮 turn/end, 返回 final_response + finish_reason + 区间事件."""
    client = _RpcClient()
    session = DshSession(session_id="s1", client=client)
    async with session:
        task = asyncio.create_task(session.run("hi"))
        await asyncio.sleep(0)  # 让 run() 挂上收集器并进入等待

        session.accept_session_event(_turn_start_event(1))
        session.accept_session_event(_assistant_message_event("hello"))
        session.accept_session_event(_turn_end_event(1, "completed"))

        result = await asyncio.wait_for(task, 1)

    assert result.session_id == "s1"
    assert result.final_response == "hello"
    assert result.finish_reason == "completed"
    # 区间事件 = 本轮 turn/start..turn/end (含端点).
    assert [e.meta.type for e in result.events] == ["turn/start", "assistant/message", "turn/end"]
    # str 入参被包成单个 text 块发给 session.prompt.
    assert len(client.prompt_params) == 1
    assert client.prompt_params[0].content[0].text == "hi"


@pytest.mark.asyncio
async def test_run_second_turn_is_isolated():
    """连续两轮: 第二轮结果只反映第二轮, 收集器与在飞门控在 run 结束后已释放."""
    client = _RpcClient()
    session = DshSession(session_id="s1", client=client)
    async with session:
        first = asyncio.create_task(session.run("q1"))
        await asyncio.sleep(0)
        session.accept_session_event(_turn_start_event(1))
        session.accept_session_event(_assistant_message_event("a1", turn=1))
        session.accept_session_event(_turn_end_event(1, "completed"))
        assert (await asyncio.wait_for(first, 1)).final_response == "a1"

        second = asyncio.create_task(session.run("q2"))
        await asyncio.sleep(0)
        session.accept_session_event(_turn_start_event(2))
        session.accept_session_event(_assistant_message_event("a2", turn=2))
        session.accept_session_event(_turn_end_event(2, "completed"))
        result = await asyncio.wait_for(second, 1)

    assert result.final_response == "a2"
    assert [e.meta.type for e in result.events] == ["turn/start", "assistant/message", "turn/end"]


@pytest.mark.asyncio
async def test_run_settles_on_cancel():
    """cancel() 中断在跑的 turn → pending run 以 turn/end 结算, reason 反映中断."""
    client = _RpcClient()
    session = DshSession(session_id="s1", client=client)
    async with session:
        task = asyncio.create_task(session.run("hi"))
        await asyncio.sleep(0)
        session.accept_session_event(_turn_start_event(1))

        await session.cancel()
        session.accept_session_event(_turn_end_event(1, "interrupted"))

        result = await asyncio.wait_for(task, 1)

    assert result.finish_reason == "interrupted"
    assert "session/cancel" in client.calls


@pytest.mark.asyncio
async def test_run_rejects_concurrent_run():
    """同一 session 同时只允许一个 run 在飞 (对齐 ACP one in-flight request)."""
    client = _RpcClient()
    session = DshSession(session_id="s1", client=client)
    async with session:
        task = asyncio.create_task(session.run("hi"))
        await asyncio.sleep(0)
        with pytest.raises(RuntimeError):
            await session.run("again")

        # 收尾第一轮, 避免悬挂.
        session.accept_session_event(_turn_start_event(1))
        session.accept_session_event(_turn_end_event(1, "completed"))
        await asyncio.wait_for(task, 1)


@pytest.mark.asyncio
async def test_cancel_calls_rpc_and_returns_value():
    client = _RpcClient()
    session = DshSession(session_id="s1", client=client)
    async with session:
        value = await session.cancel()
    assert value.accepted is True
    assert client.calls == ["session/cancel"]
