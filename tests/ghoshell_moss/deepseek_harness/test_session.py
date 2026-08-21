"""DshSession 状态面行为证据 — 运行态镜像事件 + token 记账回调.

覆盖:
- 初始态: 新建 session 假设 idle, when_idle 立即返回, when_running 阻塞.
- host/session-status 帧翻转 running ⇄ idle 镜像事件.
- assistant/message usage 累进累计量, usage update 回调收到累计值.
- 回调支持同步/异步, 解绑后不再收到.
"""

import asyncio

import pytest

from ghoshell_moss.deepseek_harness.session import DshSession
from ghoshell_moss.deepseek_harness.types import sessions
from ghoshell_moss.deepseek_harness.types.events import HostFrame, MuxFrame
from ghoshell_moss.deepseek_harness.types.session_events import (
    AssistantMessageEvent,
    ContentBlock,
    EpochHeader,
    Message,
    RequestHeader,
    TokenUsage,
)


class _DummyClient:
    """帧消费路径不触碰 client — 用哑元即可构造."""

    pass


class _RpcClient:
    """pull 路径的哑元 client — call() 返回预置 value, 并记录被调用的 method."""

    def __init__(self, models_value=None, history_value=None, session_list_value=None):
        self._models_value = models_value
        self._history_value = history_value
        self._session_list_value = session_list_value
        self.calls: list[str] = []

    async def call(self, method, params, value_cls):
        self.calls.append(method)
        if method == "session.models":
            return self._models_value
        if method == "session.history":
            return self._history_value
        if method == "session.list":
            return self._session_list_value
        raise AssertionError(f"unexpected rpc {method}")


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
async def test_usage_accumulates_and_callback_fires():
    session = DshSession(session_id="s1", client=_DummyClient())
    got = asyncio.Event()
    seen: list[tuple[int, int]] = []

    def on_usage(total: TokenUsage) -> None:
        seen.append((total.inputTokens, total.outputTokens))
        got.set()

    session.on_usage_update(on_usage)
    async with session:
        session.accept_frame(_usage_frame(10, 5))
        await asyncio.wait_for(got.wait(), 1)
        got.clear()
        session.accept_frame(_usage_frame(20, 3))
        await asyncio.wait_for(got.wait(), 1)

    # 回调每次收到累计值, 非单步增量.
    assert seen == [(10, 5), (30, 8)]
    assert session.token_usage.inputTokens == 30
    assert session.token_usage.outputTokens == 8


@pytest.mark.asyncio
async def test_usage_async_callback_supported():
    session = DshSession(session_id="s1", client=_DummyClient())
    got = asyncio.Event()
    seen: list[int] = []

    async def on_usage(total: TokenUsage) -> None:
        seen.append(total.inputTokens)
        got.set()

    session.on_usage_update(on_usage)
    async with session:
        session.accept_frame(_usage_frame(7, 2))
        await asyncio.wait_for(got.wait(), 1)

    assert seen == [7]


@pytest.mark.asyncio
async def test_usage_callback_disposer_removes():
    session = DshSession(session_id="s1", client=_DummyClient())
    got = asyncio.Event()
    seen: list[int] = []

    def on_usage(total: TokenUsage) -> None:
        seen.append(total.inputTokens)
        got.set()

    remove = session.on_usage_update(on_usage)
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
async def test_instruction_mirrors_request_header_frame():
    session = DshSession(session_id="s1", client=_DummyClient())
    async with session:
        session.accept_frame(_request_header_frame("prompt: hello"))
        await _drain(session)
        assert await session.instruction() == "prompt: hello"


@pytest.mark.asyncio
async def test_instruction_force_pulls_history_fold():
    header = RequestHeader(header=EpochHeader(system="prompt: folded"), reason="change")
    header.meta.type = header.event_type()
    history_value = sessions.SessionHistoryValue(
        events=[sessions.HistoryEntry(event=header.to_session_event())],
    )
    client = _RpcClient(history_value=history_value)
    session = DshSession(session_id="s1", client=client)
    async with session:
        assert await session.instruction(force=True) == "prompt: folded"
        assert "session.history" in client.calls


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
