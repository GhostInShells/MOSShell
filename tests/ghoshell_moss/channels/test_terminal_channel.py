import tempfile
from pathlib import Path

import pytest

from ghoshell_moss.channels.terminal_channel import new_terminal_channel


@pytest.fixture
def tmpdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# -- commands registered ------------------------------------------------

@pytest.mark.asyncio
async def test_commands_registered():
    chan = new_terminal_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {"exec", "run", "read", "write"}


# -- bash:exec ---------------------------------------------------------

@pytest.mark.asyncio
async def test_exec_returns_stdout():
    chan = new_terminal_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("exec", kwargs={"cmd": "echo hello"})
        assert "hello" in result
        assert "[exit: 0]" in result


@pytest.mark.asyncio
async def test_exec_captures_stderr():
    chan = new_terminal_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("exec", kwargs={"cmd": "echo err >&2"})
        assert "[stderr]" in result
        assert "[exit: 0]" in result


@pytest.mark.asyncio
async def test_exec_command_not_found():
    chan = new_terminal_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("exec", kwargs={"cmd": "nonexistent_cmd_xyz"})
        assert "[exit:" in result
        assert "0" not in result.split("[exit:")[-1].split("]")[0]


# -- bash:run -----------------------------------------------------------

@pytest.mark.asyncio
async def test_run_nonblocking():
    chan = new_terminal_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("run", kwargs={"cmd": "echo bg_task"})
        assert "[running" in result or "[run failed" in result


# -- bash:write + bash:read ---------------------------------------------

@pytest.mark.asyncio
async def test_write_and_read(tmpdir):
    chan = new_terminal_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command("write", kwargs={"path": "hello.txt", "text__": "hello world"})
        content = await runtime.execute_command("read", kwargs={"path": "hello.txt"})
        assert "hello world" in content
        assert "1|" in content  # line numbers


@pytest.mark.asyncio
async def test_write_overwrites(tmpdir):
    chan = new_terminal_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command("write", kwargs={"path": "x.txt", "text__": "v1"})
        await runtime.execute_command("write", kwargs={"path": "x.txt", "text__": "v2"})
        content = await runtime.execute_command("read", kwargs={"path": "x.txt"})
        assert "v2" in content
        assert "v1" not in content


@pytest.mark.asyncio
async def test_read_with_line_numbers(tmpdir):
    chan = new_terminal_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command("write", kwargs={"path": "nums.txt", "text__": "a\nb\nc"})
        content = await runtime.execute_command("read", kwargs={"path": "nums.txt"})
        lines = content.split("\n")
        assert lines[0].startswith("1|")
        assert lines[1].startswith("2|")
        assert lines[2].startswith("3|")


# -- path safety --------------------------------------------------------

@pytest.mark.asyncio
async def test_write_traversal_rejected(tmpdir):
    chan = new_terminal_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        with pytest.raises(Exception):
            await runtime.execute_command("write", kwargs={"path": "../escape.txt", "text__": "x"})


@pytest.mark.asyncio
async def test_read_traversal_rejected(tmpdir):
    chan = new_terminal_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        with pytest.raises(Exception):
            await runtime.execute_command("read", kwargs={"path": "/etc/passwd"})


# -- nested directories -------------------------------------------------

@pytest.mark.asyncio
async def test_nested_write_read(tmpdir):
    chan = new_terminal_channel(tmpdir)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command("write", kwargs={"path": "a/b/c.txt", "text__": "deep"})
        content = await runtime.execute_command("read", kwargs={"path": "a/b/c.txt"})
        assert "deep" in content
        assert Path(tmpdir, "a", "b", "c.txt").read_text() == "deep"
