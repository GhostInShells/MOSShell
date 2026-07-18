"""Tests for terminal_channel (bash) — Subprocesses rebase.

Covers:
- exec 回执格式 + stderr + timeout
- run spawn 即返 + 退出通知（signal notify）
- read_output 限长与 offset/limit
- stop 只停自己的 index
- 生命周期两态：传入 running 实例不托管；未启动实例托管
"""

from __future__ import annotations

import asyncio

import pytest

from ghoshell_moss.channels.terminal_channel import (
    ExitNotifyAddition,
    new_terminal_channel,
)
from ghoshell_moss.core.subprocesses._impl import SubprocessesImpl


# -- commands registered ----------------------------------------------------

@pytest.mark.asyncio
async def test_commands_registered():
    chan = new_terminal_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {"exec", "run", "read_output", "stop"}


@pytest.mark.asyncio
async def test_instruction_has_system_context():
    chan = new_terminal_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        instruction = runtime.self_meta().instruction
        assert "[System Context]" in instruction
        assert "OS:" in instruction
        assert "Default cwd:" in instruction


# -- bash:exec — 机制① 同步阻塞 ---------------------------------------------

@pytest.mark.asyncio
async def test_exec_returns_stdout_and_exit_marker():
    chan = new_terminal_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("exec", kwargs={"cmd": "echo hello"})
        assert "hello" in result
        assert "exit: 0" in result


@pytest.mark.asyncio
async def test_exec_captures_stderr():
    chan = new_terminal_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "exec", kwargs={"cmd": "echo err >&2"}
        )
        assert "[stderr]" in result
        assert "err" in result


@pytest.mark.asyncio
async def test_exec_nonzero_exit():
    chan = new_terminal_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "exec", kwargs={"cmd": "exit 7"}
        )
        assert "exit: 7" in result


@pytest.mark.asyncio
async def test_exec_timeout():
    chan = new_terminal_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "exec", kwargs={"cmd": "sleep 5", "timeout": 0.3}
        )
        assert "timeout" in result.lower()


# -- bash:run — 机制③ 全异步 -------------------------------------------------

@pytest.mark.asyncio
async def test_run_returns_receipt_immediately():
    chan = new_terminal_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "run", kwargs={"cmd": "echo bg", "name": "greet"}
        )
        assert "started" in result
        assert "greet" in result
        assert "pid=" in result


@pytest.mark.asyncio
async def test_run_binds_notify_priority_addition():
    sp = SubprocessesImpl()
    chan = new_terminal_channel(sp)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command(
            "run", kwargs={"cmd": "sleep 0.2", "notify": "notice"}
        )
        # 找到刚 spawn 的进程 meta，验证 Addition 已挂
        metas = list(sp.executing().values()) + sp.executed()
        assert metas, "expected at least one process"
        additions = [ExitNotifyAddition.read(m) for m in metas]
        additions = [a for a in additions if a is not None]
        assert additions, "ExitNotifyAddition should be attached"
        assert any(a.level == "notice" for a in additions)


@pytest.mark.asyncio
async def test_run_unknown_notify_falls_back_to_background():
    sp = SubprocessesImpl()
    chan = new_terminal_channel(sp)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command(
            "run", kwargs={"cmd": "true", "notify": "bogus"}
        )
        metas = list(sp.executing().values()) + sp.executed()
        additions = [ExitNotifyAddition.read(m) for m in metas if m is not None]
        additions = [a for a in additions if a is not None]
        # 未识别的 notify 应存为 background 默认
        assert any(a.level == "background" for a in additions)


# -- bash:read_output — 机制② nonblocking 快命令 ----------------------------

@pytest.mark.asyncio
async def test_read_output_returns_captured_stdout():
    chan = new_terminal_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        started = await runtime.execute_command(
            "run", kwargs={"cmd": "echo peek"}
        )
        # 从回执抽 index — "[#N started]..."
        import re
        m = re.search(r"#(\d+)", started)
        assert m
        index = int(m.group(1))
        # 让退出 + drain 完成
        await asyncio.sleep(0.3)
        result = await runtime.execute_command(
            "read_output", kwargs={"index": index}
        )
        assert "peek" in result
        assert "exit: 0" in result


@pytest.mark.asyncio
async def test_read_output_unknown_index_raises_observe():
    chan = new_terminal_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        with pytest.raises(Exception):
            await runtime.execute_command(
                "read_output", kwargs={"index": 9999}
            )


@pytest.mark.asyncio
async def test_read_output_truncates_long_output():
    chan = new_terminal_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        # 生成远超 12000 字符的输出
        started = await runtime.execute_command(
            "run",
            kwargs={"cmd": "python -c 'print(\"x\"*100)' " * 200},
        )
        import re
        m = re.search(r"#(\d+)", started)
        assert m
        index = int(m.group(1))
        await asyncio.sleep(0.5)
        result = await runtime.execute_command(
            "read_output", kwargs={"index": index}
        )
        # 结果的整体长度受 _RESULT_CHAR_CAP 约束
        assert len(result) < 15_000


# -- bash:stop — 机制② nonblocking 快命令 -----------------------------------

@pytest.mark.asyncio
async def test_stop_stops_running_process():
    chan = new_terminal_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        started = await runtime.execute_command(
            "run", kwargs={"cmd": "sleep 30"}
        )
        import re
        index = int(re.search(r"#(\d+)", started).group(1))
        result = await runtime.execute_command(
            "stop", kwargs={"index": index, "timeout": 1.0}
        )
        assert "stopped" in result


@pytest.mark.asyncio
async def test_stop_unknown_index_raises_observe():
    chan = new_terminal_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        with pytest.raises(Exception):
            await runtime.execute_command("stop", kwargs={"index": 9999})


@pytest.mark.asyncio
async def test_stop_refuses_foreign_process():
    """所有权隔离: 共享的 Subprocesses 里, 别处 spawn 的进程 stop 不到."""
    sp = SubprocessesImpl()
    async with sp:
        foreign = await sp.shell("sleep 30")
        try:
            chan = new_terminal_channel(sp)
            async with chan.bootstrap() as runtime:
                await runtime.refresh_metas()
                with pytest.raises(Exception):
                    await runtime.execute_command(
                        "stop", kwargs={"index": foreign.meta.index}
                    )
        finally:
            await foreign.stop(timeout=0.5)


# -- 生命周期两态 -----------------------------------------------------------

@pytest.mark.asyncio
async def test_channel_owns_lifecycle_when_processes_not_started():
    sp = SubprocessesImpl()
    assert not sp.is_running()
    chan = new_terminal_channel(sp)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert sp.is_running(), "channel should have started processes"
    # channel 关闭后, 应该 __aexit__ 回来
    assert not sp.is_running(), "channel should have stopped processes"


@pytest.mark.asyncio
async def test_channel_does_not_own_lifecycle_when_already_running():
    sp = SubprocessesImpl()
    async with sp:
        assert sp.is_running()
        chan = new_terminal_channel(sp)
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            assert sp.is_running()
        # channel 关闭后, processes 仍应 running（归 outer owner）
        assert sp.is_running(), "outer-owned processes should keep running"


# -- context_messages 后台任务简表 ------------------------------------------

@pytest.mark.asyncio
async def test_context_messages_show_own_processes():
    chan = new_terminal_channel(SubprocessesImpl())
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        # 无进程时, context 为空
        meta = runtime.self_meta()
        assert not meta.context or all(
            "processes" not in "".join(c.get("text", "") for c in ctx.contents
                                       if c.get("type") == "text")
            for ctx in meta.context
        )
        # 起一个后台进程
        await runtime.execute_command("run", kwargs={"cmd": "sleep 5", "name": "watchdog"})
        await runtime.refresh_metas()
        meta = runtime.self_meta()
        context_text = "".join(
            c.get("text", "")
            for ctx in meta.context
            for c in ctx.contents
            if c.get("type") == "text"
        )
        assert "watchdog" in context_text
        assert "running" in context_text
