import pytest

from ghoshell_moss.channels.shell_channel import new_shell_channel


# -- commands registered -------------------------------------------------

@pytest.mark.asyncio
async def test_commands_registered():
    chan = new_shell_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        cmd_names = {c.name for c in runtime.self_meta().commands}
        assert cmd_names == {"sendline", "read_output", "sendcontrol", "close"}


# -- instruction ---------------------------------------------------------

@pytest.mark.asyncio
async def test_instruction_set():
    chan = new_shell_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        meta = runtime.self_meta()
        assert meta.instruction
        assert "sendline" in meta.instruction


# -- context messages ----------------------------------------------------

@pytest.mark.asyncio
async def test_context_messages_before_spawn():
    chan = new_shell_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        meta = runtime.self_meta()
        assert len(meta.context) > 0
        context_text = "".join(
            c.get("text", "") for c in meta.context[0].contents
            if c.get("type") == "text"
        )
        assert "not started" in context_text


@pytest.mark.asyncio
async def test_context_messages_after_sendline():
    chan = new_shell_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command("sendline", kwargs={"text__": "echo hello"})

        await runtime.refresh_metas()
        meta = runtime.self_meta()
        context_text = "".join(
            c.get("text", "") for c in meta.context[0].contents
            if c.get("type") == "text"
        )
        assert "bash" in context_text.lower()
        assert "cursor:1" in context_text
        assert "segments:" in context_text


# -- sendline executes shell command --------------------------------------

@pytest.mark.asyncio
async def test_sendline_returns_output():
    chan = new_shell_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "sendline", kwargs={"text__": "echo hello world"}
        )
        assert "hello world" in result
        assert "segment #1" in result


@pytest.mark.asyncio
async def test_sendline_creates_segments():
    chan = new_shell_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command("sendline", kwargs={"text__": "echo first"})
        result = await runtime.execute_command(
            "sendline", kwargs={"text__": "echo second"}
        )
        assert "segment #2" in result

        # read back previous segment
        content = await runtime.execute_command(
            "read_output", kwargs={"id": 1}
        )
        assert "first" in content


@pytest.mark.asyncio
async def test_sendline_command_not_found():
    chan = new_shell_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "sendline", kwargs={"text__": "nonexistent_cmd_xyz"}
        )
        # shell returns non-zero, output still captured
        assert "segment #1" in result


# -- read_output ---------------------------------------------------------

@pytest.mark.asyncio
async def test_read_output_nonexistent():
    chan = new_shell_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command(
            "read_output", kwargs={"id": 99}
        )
        assert "not found" in result


# -- sendcontrol ----------------------------------------------------------

@pytest.mark.asyncio
async def test_sendcontrol():
    chan = new_shell_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        # sendcontrol triggers auto-spawn, then sends ^C
        result = await runtime.execute_command(
            "sendcontrol", kwargs={"char": "c"}
        )
        assert "sent" in result


# -- close ----------------------------------------------------------------

@pytest.mark.asyncio
async def test_close_before_spawn():
    chan = new_shell_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("close", kwargs={})
        assert "not started" in result


@pytest.mark.asyncio
async def test_close_after_use():
    chan = new_shell_channel()
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command("sendline", kwargs={"text__": "echo x"})
        result = await runtime.execute_command("close", kwargs={})
        assert "closed" in result
