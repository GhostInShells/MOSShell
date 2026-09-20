"""DshRuntime 行为证据 — flat 控制面 (name 寻址 + ledger 订阅 + watch + 回执).

network-free: 注入 fake connection/session, 只测协议承诺:
- open_session 按 name 接线; send/wait/interrupt/status/read 走 lifted 动词.
- ledger 订阅把 user/message·assistant/message·turn/end 折叠成尾句/未读.
- watch 默认 silent, notify/next 显式升档; 人类消息 (source.kind=user) 才触发, 模型自己的不算.
- run_bg 回执进 runs() 账本, 完成按 watch level 发 signal.
"""

import asyncio

import pytest

from ghoshell_moss.deepseek_harness.runtime import DshRuntime
from ghoshell_moss.deepseek_harness.types.session_events import (
    AssistantMessageEvent,
    ContentBlock,
    Message,
    MessageSource,
    TokenUsage,
    TurnEnd,
    TurnEndReason,
    UserMessageEvent,
)


class _Cfg:
    host = "127.0.0.1"
    port = 3080


class _FakeSession:
    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id
        self.running = False
        self.token_usage = TokenUsage()
        self.entered = False
        self.closed = False
        self.prompts: list = []
        self.cancelled = False
        self.surface: list[Message] = []
        self._handlers: dict[str, list] = {}

    async def __aenter__(self) -> "_FakeSession":
        self.entered = True
        return self

    async def __aexit__(self, *exc) -> None:
        self.closed = True

    async def close(self) -> None:
        self.closed = True

    def on_session_event_model(self, model_cls, callback):
        self._handlers.setdefault(model_cls.event_type(), []).append(callback)

        def _remove() -> None:
            self._handlers[model_cls.event_type()].remove(callback)

        return _remove

    async def prompt(self, *, content, mode="queue", client_timezone=None):
        self.prompts.append(content)
        return None

    async def cancel(self):
        self.cancelled = True
        return None

    async def surface_messages(self) -> list[Message]:
        return self.surface

    async def fire(self, model) -> None:
        for cb in list(self._handlers.get(model.event_type(), [])):
            await cb(model)


class _FakeConnection:
    def __init__(self) -> None:
        self.entered = False
        self.closed = False
        self.running = False
        self.config = _Cfg()
        self.sessions: dict[str, _FakeSession] = {}

    async def __aenter__(self) -> "_FakeConnection":
        self.entered = True
        self.running = True
        return self

    async def __aexit__(self, *exc) -> None:
        self.closed = True
        self.running = False

    def is_running(self) -> bool:
        return self.running

    def create_session(self, session_id: str) -> _FakeSession:
        session = _FakeSession(session_id)
        self.sessions[session_id] = session
        return session


def _runtime(send_signal=None) -> tuple[DshRuntime, _FakeConnection]:
    conn = _FakeConnection()
    return DshRuntime(conn, send_signal=send_signal), conn


def _assistant(text: str) -> AssistantMessageEvent:
    return AssistantMessageEvent(message=Message(role="assistant", content=[ContentBlock(type="text", text=text)]))


def _human(text: str, kind: str = "user") -> UserMessageEvent:
    return UserMessageEvent(content=[ContentBlock(type="text", text=text)], source=MessageSource(kind=kind))


def _turn_end(kind: str = "completed") -> TurnEnd:
    return TurnEnd(reason=TurnEndReason(kind=kind))


# -- 会话接线 -- #


@pytest.mark.asyncio
async def test_open_session_registers_by_name():
    rt, conn = _runtime()
    await rt.__aenter__()
    await rt.open_session("sid-1", "p1")
    assert rt.session_names() == ["p1"]
    assert conn.sessions["sid-1"].entered
    await rt.close()


@pytest.mark.asyncio
async def test_open_session_same_id_second_name_rejected():
    rt, conn = _runtime()
    await rt.__aenter__()
    await rt.open_session("sid-1", "p1")
    with pytest.raises(ValueError):
        await rt.open_session("sid-1", "p2")
    await rt.close()


# -- lifted 动词 -- #


@pytest.mark.asyncio
async def test_send_prompts_session():
    rt, conn = _runtime()
    await rt.__aenter__()
    await rt.open_session("sid-1", "p1")
    result = await rt.send("p1", "hello")
    assert "sent" in result
    assert conn.sessions["sid-1"].prompts
    await rt.close()


@pytest.mark.asyncio
async def test_wait_returns_preview_after_turn_end():
    rt, conn = _runtime()
    await rt.__aenter__()
    await rt.open_session("sid-1", "p1")
    session = conn.sessions["sid-1"]
    task = asyncio.create_task(rt.wait("p1"))
    await asyncio.sleep(0)  # 让 wait 挂上 turn/end 监听.
    await session.fire(_assistant("the answer"))
    await session.fire(_turn_end("completed"))
    result = await asyncio.wait_for(task, 1)
    assert "the answer" in result
    assert "completed" in result
    await rt.close()


@pytest.mark.asyncio
async def test_interrupt_cancels():
    rt, conn = _runtime()
    await rt.__aenter__()
    await rt.open_session("sid-1", "p1")
    await rt.interrupt("p1")
    assert conn.sessions["sid-1"].cancelled
    await rt.close()


@pytest.mark.asyncio
async def test_read_returns_surface_and_clears_unread():
    rt, conn = _runtime()
    await rt.__aenter__()
    await rt.open_session("sid-1", "p1")
    session = conn.sessions["sid-1"]
    await session.fire(_human("hi there"))
    assert "unread" in rt.describe("p1")
    session.surface = [Message(role="user", content=[ContentBlock(type="text", text="hi there")])]
    result = await rt.read("p1", n=10)
    assert "hi there" in result
    assert "idle" in rt.describe("p1")
    await rt.close()


@pytest.mark.asyncio
async def test_status_reports_running_and_tokens():
    rt, conn = _runtime()
    await rt.__aenter__()
    await rt.open_session("sid-1", "p1")
    conn.sessions["sid-1"].running = True
    conn.sessions["sid-1"].token_usage = TokenUsage(inputTokens=3, outputTokens=5)
    result = await rt.status("p1")
    assert "running=True" in result
    assert "tokens_in=3" in result and "tokens_out=5" in result
    await rt.close()


# -- watch / signal -- #


@pytest.mark.asyncio
async def test_watch_silent_by_default():
    sent = []
    rt, conn = _runtime(send_signal=sent.append)
    await rt.__aenter__()
    await rt.open_session("sid-1", "p1")
    rt.watch("p1", "row")  # 只更新行, 零 signal.
    await conn.sessions["sid-1"].fire(_human("human ping"))
    assert sent == []
    await rt.close()


@pytest.mark.asyncio
async def test_watch_notify_emits_signal_with_provenance():
    sent = []
    rt, conn = _runtime(send_signal=sent.append)
    await rt.__aenter__()
    await rt.open_session("sid-1", "p1")
    rt.watch("p1", "notify")
    await conn.sessions["sid-1"].fire(_human("human ping"))
    assert len(sent) == 1
    sig = sent[0]
    assert "p1" in sig.description
    assert sig.metadata.get("next") is not True
    await rt.close()


@pytest.mark.asyncio
async def test_watch_next_sets_next_flag():
    sent = []
    rt, conn = _runtime(send_signal=sent.append)
    await rt.__aenter__()
    await rt.open_session("sid-1", "p1")
    rt.watch("p1", "next")
    await conn.sessions["sid-1"].fire(_human("urgent"))
    assert len(sent) == 1
    assert sent[0].metadata.get("next") is True
    await rt.close()


@pytest.mark.asyncio
async def test_model_origin_user_message_not_signaled():
    sent = []
    rt, conn = _runtime(send_signal=sent.append)
    await rt.__aenter__()
    await rt.open_session("sid-1", "p1")
    rt.watch("p1", "notify")
    await conn.sessions["sid-1"].fire(_human("model echo", kind="plugin"))
    assert sent == []
    await rt.close()


@pytest.mark.asyncio
async def test_notice_rows_only_watched():
    rt, conn = _runtime()
    await rt.__aenter__()
    await rt.open_session("sid-1", "p1")
    await rt.open_session("sid-2", "p2")
    rt.watch("p1", "row")
    rows = rt.notice_rows()
    assert "p1" in rows and "p2" not in rows
    await rt.close()


# -- run_bg 回执 -- #


@pytest.mark.asyncio
async def test_run_bg_receipt_and_completion_signal():
    sent = []
    rt, conn = _runtime(send_signal=sent.append)
    await rt.__aenter__()
    await rt.open_session("sid-1", "p1")
    rt.watch("p1", "notify")
    receipt = rt.run_code_bg("p1", "async def run(session):\n    return 'ok'")
    assert "#1" in receipt
    # 等后台任务完成.
    for _ in range(50):
        if rt.runs().count("done") == 0:
            await asyncio.sleep(0.01)
    assert "#1" in rt.runs()
    assert len(sent) == 1 and "p1" in sent[0].description
    await rt.close()


# -- 生命周期 -- #


@pytest.mark.asyncio
async def test_close_is_idempotent_and_closes_connection():
    rt, conn = _runtime()
    await rt.__aenter__()
    assert conn.entered
    await rt.close()
    assert conn.closed
    await rt.close()


@pytest.mark.asyncio
async def test_close_cancels_background_run():
    rt, conn = _runtime()
    await rt.__aenter__()
    await rt.open_session("sid-1", "p1")
    rt.run_code_bg("p1", "async def run(session):\n    await asyncio.sleep(10)")
    await asyncio.sleep(0)
    await rt.close()
    assert rt.runs().count("done") == 0
