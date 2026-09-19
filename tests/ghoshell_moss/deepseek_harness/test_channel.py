"""deepseek_harness.channel 行为证据 — 父 (connection) → 子 (session) 两层命令面.

network-free: connection/session 注入 fake, 只测命令注册、指令指路、run 注入 surface、
open 物化 session 子 channel. 编译+注入+await 的纯逻辑在 test_runtime 已覆盖.
"""

from __future__ import annotations

import pytest

from ghoshell_moss.deepseek_harness.channel import new_dsh_runtime_channel
from ghoshell_moss.deepseek_harness.runtime import DshRuntime


class _FakeSession:
    def __init__(self, session_id: str = "") -> None:
        self.session_id = session_id
        self.running = False

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    async def close(self) -> None:
        return None


class _FakeConnection:
    async def __aenter__(self) -> "_FakeConnection":
        return self

    async def __aexit__(self, *exc) -> None:
        return None

    def create_session(self, session_id: str) -> _FakeSession:
        return _FakeSession(session_id)


def _channel():
    return new_dsh_runtime_channel(DshRuntime(_FakeConnection()))


@pytest.mark.asyncio
async def test_parent_commands_registered():
    chan = _channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {"run", "run_bg", "open", "close_session", "sessions"}


@pytest.mark.asyncio
async def test_parent_instruction_points_to_surfaces():
    chan = _channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert "ghoshell_moss.deepseek_harness.surfaces" in runtime.self_meta().instruction


@pytest.mark.asyncio
async def test_parent_run_injects_connection_surface():
    chan = _channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "run",
            kwargs={"text__": "async def run(connection):\n    return 'ok'\n"},
        )
        assert "ok" in result


@pytest.mark.asyncio
async def test_open_generates_valid_alias_for_uuid_session_id():
    """open() 默认别名不能是 session_id (含 `-`/数字开头, 非法 channel 名) — 应生成 sN."""
    chan = _channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "open", kwargs={"session_id": "session-1ac28941-668c-4de5-869d-fdfed1d5957d"}
        )
        assert "dsh.s0" in result
        assert "s0" in chan.virtual_children()


@pytest.mark.asyncio
async def test_open_rejects_invalid_alias():
    chan = _channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "open", kwargs={"session_id": "s1", "alias": "bad-name"}
        )
        assert "invalid alias" in result
        assert chan.virtual_children() == {}


@pytest.mark.asyncio
async def test_session_child_run_injects_session():
    chan = _channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command("open", kwargs={"session_id": "s1", "alias": "s1"})
        # 子 channel 是独立的 runtime, 经 chan.virtual_children 拿到后再 bootstrap.
        child = chan.virtual_children()["s1"]
        async with child.bootstrap() as child_runtime:
            await child_runtime.refresh_metas()
            cmd_names = {c.name for c in child_runtime.self_meta().commands}
            assert cmd_names == {"run", "run_bg"}
            result = await child_runtime.execute_command(
                "run",
                kwargs={"text__": "async def run(session):\n    return 'session-ok'\n"},
            )
            assert "session-ok" in result


@pytest.mark.asyncio
async def test_named_notices_present_open_sessions():
    """open 后 named notice 呈现 alias → session_id + running, 模型下轮可反查."""
    chan = _channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command("open", kwargs={"session_id": "s1", "alias": "s1"})
        await runtime.refresh_metas()
        notices = runtime.self_meta().named_notices
        assert notices.get("s1") == "session_id=s1 running=False"
