import asyncio
import tempfile
from pathlib import Path

import pytest

from ghoshell_moss import new_shell_main_channel
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.concepts.errors import CommandError
from ghoshell_moss.core.ctml import ctml_shell_test, new_ctml_shell
from ghoshell_moss.channels.macro_store import MacroStoreModule


# -- 基本查询: list / not-found -- #

@pytest.mark.asyncio
async def test_macro_list_empty():
    """Empty store returns placeholder."""
    main = new_shell_main_channel()
    main.with_module(MacroStoreModule())

    async with main.bootstrap() as runtime:
        listing = await runtime.execute_command("macro_list")
        assert "no macros" in listing


@pytest.mark.asyncio
async def test_macro_not_found():
    """Invoking a non-existent macro raises CommandError."""
    main = new_shell_main_channel()
    main.with_module(MacroStoreModule())

    async with main.bootstrap() as runtime:
        with pytest.raises(CommandError, match="not found"):
            await runtime.execute_command("macro", kwargs={"label": "nonexistent"})
        with pytest.raises(CommandError, match="not found"):
            await runtime.execute_command("macro_read", kwargs={"label": "nonexistent"})


# -- 宏展开: CTML 层端到端 -- #

@pytest.mark.asyncio
async def test_macro_expansion():
    """Save via CTML, invoke via macro, verify expansion at call site."""
    a_chan = new_channel(name="a")
    calls = []

    @a_chan.build.command()
    async def say(text: str = ""):
        calls.append(text)

    def builder(shell):
        shell.main_channel.with_module(MacroStoreModule())

    tasks = await ctml_shell_test(
        a_chan,
        builder=builder,
        ctml=(
            '<macro_save label="greet" description="say hello">'
            '<![CDATA[<a:say text="hello from macro"/>]]>'
            '</macro_save>\n'
            '<macro label="greet"/>\n'
        ),
    )
    assert calls == ["hello from macro"]

    # macro task (model-emitted) has no macro_id; expanded say task does.
    macro_task = [t for t in tasks if t.caller_name() == "macro"][0]
    expanded = [t for t in tasks if t.caller_name() == "a:say"][0]
    assert macro_task.macro_id is None
    assert expanded.macro_id is not None


@pytest.mark.asyncio
async def test_macro_nested_expansion():
    """Macro expanding to another macro — recursion within depth cap."""
    a_chan = new_channel(name="a")
    calls = []

    @a_chan.build.command()
    async def mark():
        calls.append("marked")

    def builder(shell):
        shell.main_channel.with_module(MacroStoreModule())

    tasks = await ctml_shell_test(
        a_chan,
        builder=builder,
        ctml=(
            '<macro_save label="inner"><![CDATA[<a:mark/>]]></macro_save>\n'
            '<macro_save label="outer"><![CDATA[<macro label="inner"/>]]></macro_save>\n'
            '<macro label="outer"/>\n'
        ),
    )
    assert calls == ["marked"]

    outer = [t for t in tasks if t.caller_name() == "macro" and t.macro_id is None][0]
    inner = [t for t in tasks if t.caller_name() == "macro" and t.macro_id is not None][0]
    mark = [t for t in tasks if t.caller_name() == "a:mark"][0]
    assert outer.macro_id is None
    assert inner.macro_id is not None
    assert mark.macro_id is not None


@pytest.mark.asyncio
async def test_macro_overwrite():
    """Save same label twice — last write wins."""
    a_chan = new_channel(name="a")
    calls = []

    @a_chan.build.command()
    async def say(text: str = ""):
        calls.append(text)

    def builder(shell):
        shell.main_channel.with_module(MacroStoreModule())

    await ctml_shell_test(
        a_chan,
        builder=builder,
        ctml=(
            '<macro_save label="x"><![CDATA[<a:say text="first"/>]]></macro_save>\n'
            '<macro_save label="x"><![CDATA[<a:say text="second"/>]]></macro_save>\n'
            '<macro label="x"/>\n'
        ),
    )
    assert calls == ["second"]


# -- 磁盘持久化 -- #

@pytest.mark.asyncio
async def test_macro_dir_persistence():
    """With dir param, macros persist to disk and survive across instances."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)

        # Save via CTML in first shell.
        def builder(shell):
            shell.main_channel.with_module(MacroStoreModule(dir=dir_path))

        shell = new_ctml_shell()
        builder(shell)
        async with shell:
            interpreter = await shell.interpreter(clear_after_exit=True)
            async with interpreter:
                interpreter.feed(
                    '<macro_save label="persist"><![CDATA[<a:say/>]]></macro_save>'
                )
                interpreter.commit()
                await interpreter.wait_tasks(throw=True)
        # 写盘在 executor 线程中异步进行, 稍等一下.
        await asyncio.sleep(0.1)

        # Second instance loads from disk.
        main2 = new_shell_main_channel()
        main2.with_module(MacroStoreModule(dir=dir_path))
        async with main2.bootstrap() as runtime:
            ctml = await runtime.execute_command(
                "macro_read", kwargs={"label": "persist"}
            )
            assert ctml == "<a:say/>"
