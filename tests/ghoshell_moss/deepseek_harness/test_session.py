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
from ghoshell_moss.deepseek_harness.types.events import HostFrame, MuxFrame
from ghoshell_moss.deepseek_harness.types.session_events import (
    AssistantMessageEvent,
    ContentBlock,
    Message,
    TokenUsage,
)


class _DummyClient:
    """帧消费路径不触碰 client — 用哑元即可构造."""

    pass


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
