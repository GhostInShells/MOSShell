"""Tests for dsh_channel — 父→connection→session 三层, 代码驱动注入现场对象.

全部 network-free: connection/session 依赖注入 fake, 只测编译+注入+渲染的纯逻辑
与父 channel 命令面。真实的 connect→materialize→exec 流走 live 验证, 不进单测。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ghoshell_moss.channels.dsh_channel import (
    _new_connection_child,
    _new_session_child,
    build_dsh_channel,
    new_dsh_channel,
)
from ghoshell_moss.deepseek_harness.session import DshSession


class _FakeConnection:
    """no-op async context manager — 让 startup 的 enter_async_context 不碰网络."""

    def __init__(self, base_url: str = "http://127.0.0.1:3080"):
        self.config = SimpleNamespace(base_url=base_url)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None


def _dummy_session() -> DshSession:
    return DshSession(session_id="s1", client=object())


def _fake_connection_child(name: str = "c0") -> object:
    return _new_connection_child(
        name=name,
        connection=_FakeConnection(),
        cwd="/tmp",
        agent_preset="standard",
    )


# ---- 父 channel ---- #


@pytest.mark.asyncio
async def test_parent_commands_registered():
    chan = new_dsh_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {"connect", "disconnect", "connections"}


@pytest.mark.asyncio
async def test_parent_instruction_points_to_modules():
    chan = new_dsh_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        instruction = runtime.self_meta().instruction
        assert "ghoshell_moss.deepseek_harness.session" in instruction
        assert "ghoshell_moss.deepseek_harness.launcher" in instruction


# ---- connection 子 channel exec ---- #


@pytest.mark.asyncio
async def test_connection_exec_injects_connection():
    chan = _fake_connection_child()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "exec",
            kwargs={"text__": "async def main(connection):\n    return connection.config.base_url\n"},
        )
        assert "http://127.0.0.1:3080" in result


@pytest.mark.asyncio
async def test_connection_exec_rejects_plain_def():
    chan = _fake_connection_child()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "exec",
            kwargs={"text__": "def main(connection):\n    return 1\n"},
        )
        assert "must be `async def`" in result


# ---- session 子 channel exec ---- #


@pytest.mark.asyncio
async def test_session_exec_injects_session():
    chan = _new_session_child(_dummy_session(), name="s0")
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "exec",
            kwargs={"text__": "async def main(session):\n    return session.running\n"},
        )
        assert "False" in result


@pytest.mark.asyncio
async def test_session_exec_reports_runtime_error():
    chan = _new_session_child(_dummy_session(), name="s0")
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "exec",
            kwargs={"text__": "async def main(session):\n    return 1 / 0\n"},
        )
        assert "RUN ERROR" in result
        assert "ZeroDivisionError" in result


# ---- factory ---- #


@pytest.mark.asyncio
async def test_build_factory_yields_channel_with_declared_name():
    factory = build_dsh_channel(name="dshtool")
    chan = factory(None)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert runtime.self_meta().name == "dshtool"
