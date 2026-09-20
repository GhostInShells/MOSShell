"""deepseek_harness.channel 行为证据 — flat 单 channel 控制面.

network-free: 注入 fake connection/session, 只测命令注册、notice 单行、named_notices
只在 watch 后出现、instruction 编码注意力铁律. 编译+注入+await 纯逻辑在 test_runtime 覆盖.
"""

from __future__ import annotations

import pytest

from ghoshell_moss.deepseek_harness.channel import new_dsh_runtime_channel
from ghoshell_moss.deepseek_harness.runtime import DshRuntime


class _Cfg:
    host = "127.0.0.1"
    port = 3080


class _FakeSession:
    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id
        self.running = False
        self.token_usage = None
        self.entered = False
        self.closed = False
        self.prompts: list = []

    async def __aenter__(self) -> "_FakeSession":
        self.entered = True
        return self

    async def __aexit__(self, *exc) -> None:
        self.closed = True

    async def close(self) -> None:
        self.closed = True

    def on_session_event_model(self, model_cls, callback):
        def _remove() -> None:
            return None

        return _remove

    async def prompt(self, *, content, mode="queue", client_timezone=None):
        self.prompts.append(content)
        return None


class _FakeConnection:
    def __init__(self) -> None:
        self.config = _Cfg()
        self.running = False
        self.sessions: dict[str, _FakeSession] = {}

    async def __aenter__(self) -> "_FakeConnection":
        self.running = True
        return self

    async def __aexit__(self, *exc) -> None:
        self.running = False

    def is_running(self) -> bool:
        return self.running

    def create_session(self, session_id: str) -> _FakeSession:
        session = _FakeSession(session_id)
        self.sessions[session_id] = session
        return session


def _channel():
    return new_dsh_runtime_channel(DshRuntime(_FakeConnection()))


@pytest.mark.asyncio
async def test_flat_command_set_registered():
    chan = _channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {
            "sessions", "open", "new", "send", "wait", "interrupt", "status",
            "read", "watch", "unwatch", "runs", "run", "run_bg",
        }


@pytest.mark.asyncio
async def test_notice_is_single_line_summary():
    chan = _channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        notice = runtime.self_meta().notice
        assert "127.0.0.1" in notice
        assert "sessions" in notice


@pytest.mark.asyncio
async def test_named_notices_empty_until_watch():
    chan = _channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command("open", kwargs={"session_id": "sid-1", "name": "p1"})
        await runtime.refresh_metas()
        assert runtime.self_meta().named_notices == {}
        await runtime.execute_command("watch", kwargs={"name": "p1"})
        await runtime.refresh_metas()
        assert "p1" in runtime.self_meta().named_notices


@pytest.mark.asyncio
async def test_open_mints_name_and_returns_address():
    chan = _channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("open", kwargs={"session_id": "sid-1"})
        assert "s0" in result
        sessions = await runtime.execute_command("sessions", kwargs={})
        assert "s0" in sessions


@pytest.mark.asyncio
async def test_open_rejects_invalid_name():
    chan = _channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("open", kwargs={"session_id": "sid-1", "name": "bad-name"})
        assert "invalid name" in result


@pytest.mark.asyncio
async def test_instruction_encodes_attention_rules():
    chan = _channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        instruction = runtime.self_meta().instruction
        assert "silent" in instruction
        assert "watch" in instruction
        assert "next" in instruction


@pytest.mark.asyncio
async def test_send_via_command_prompts():
    chan = _channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command("open", kwargs={"session_id": "sid-1", "name": "p1"})
        result = await runtime.execute_command("send", kwargs={"name": "p1", "text": "hello"})
        assert "sent" in result
