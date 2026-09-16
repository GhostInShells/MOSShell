import pytest

from ghoshell_moss import new_shell_main_channel
from ghoshell_moss.channels.macro_store import (
    CDATA_END,
    CDATA_START,
    MICRO_SUFFIX,
    MacroStoreModule,
)
from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.core.concepts.errors import CommandError
from ghoshell_moss.core.ctml import ctml_shell_test


def _recorder():
    """A channel with a single `say(text)` command recording its calls."""
    chan = new_channel(name="a")
    calls = []

    @chan.build.command()
    async def say(text: str = ""):
        calls.append(text)

    return chan, calls


def _text_recorder():
    """A channel with a `say(text__)` command capturing its text body (CDATA-friendly)."""
    chan = new_channel(name="a")
    received = []

    @chan.build.command()
    async def say(text__: str = ""):
        received.append(text__)

    return chan, received


def _macro_builder(root=None):
    def builder(shell):
        shell.main_channel.with_module(MacroStoreModule(root=root))

    return builder


# -- 会话态: label -- #

@pytest.mark.asyncio
async def test_macro_list_empty():
    main = new_shell_main_channel()
    main.with_module(MacroStoreModule())

    async with main.bootstrap() as runtime:
        assert "no macros" in await runtime.execute_command("macro_list")


@pytest.mark.asyncio
async def test_macro_not_found():
    main = new_shell_main_channel()
    main.with_module(MacroStoreModule())

    async with main.bootstrap() as runtime:
        for command, kwargs in (
                ("macro", {"ref": "nonexistent"}),
                ("macro_read", {"label": "nonexistent"}),
                ("macro_forget", {"label": "nonexistent"}),
        ):
            with pytest.raises(CommandError, match="not found"):
                await runtime.execute_command(command, kwargs=kwargs)


@pytest.mark.asyncio
async def test_macro_expansion():
    """Save via CTML, invoke via macro, verify expansion at call site."""
    chan, calls = _recorder()
    tasks = await ctml_shell_test(
        chan,
        builder=_macro_builder(),
        ctml=(
            '<macro_save label="greet" description="say hello">'
            '<![CDATA[<a:say text="hello from macro"/>]]>'
            '</macro_save>\n'
            '<macro ref="greet"/>\n'
        ),
    )
    assert calls == ["hello from macro"]

    macro_task = [t for t in tasks if t.caller_name() == "macro"][0]
    expanded = [t for t in tasks if t.caller_name() == "a:say"][0]
    assert macro_task.macro_id is None
    assert expanded.macro_id is not None


@pytest.mark.asyncio
async def test_macro_overwrite_last_write_wins():
    chan, calls = _recorder()
    await ctml_shell_test(
        chan,
        builder=_macro_builder(),
        ctml=(
            '<macro_save label="x"><![CDATA[<a:say text="first"/>]]></macro_save>\n'
            '<macro_save label="x"><![CDATA[<a:say text="second"/>]]></macro_save>\n'
            '<macro ref="x"/>\n'
        ),
    )
    assert calls == ["second"]


@pytest.mark.asyncio
async def test_macro_forget():
    main = new_shell_main_channel()
    main.with_module(MacroStoreModule())

    async with main.bootstrap() as runtime:
        # forget on a missing label is an error, and the label is gone after forgetting.
        with pytest.raises(CommandError, match="not found"):
            await runtime.execute_command("macro_forget", kwargs={"label": "x"})


# -- 占位符: 嵌套 CDATA -- #

@pytest.mark.asyncio
async def test_nested_save_rejected():
    """A body that itself saves a macro is rejected (store write inside a stored body)."""
    chan, _ = _recorder()
    tasks = await ctml_shell_test(
        chan,
        builder=_macro_builder(),
        ctml='<macro_save label="outer"><![CDATA[<macro_save label="inner">x</macro_save>]]></macro_save>\n',
    )
    assert "must not nest" in str(_failed(tasks, "macro_save").exception())


@pytest.mark.asyncio
async def test_placeholder_roundtrip():
    """A body wrapping text in CDATA stores placeholders; expansion restores real CDATA."""
    chan, received = _text_recorder()
    await ctml_shell_test(
        chan,
        builder=_macro_builder(),
        ctml=(
            '<macro_save label="m">'
            f'<![CDATA[<a:say>{CDATA_START}<b>hi</b>{CDATA_END}</a:say>]]>'
            '</macro_save>\n'
            '<macro ref="m"/>\n'
        ),
    )
    assert received == ["<b>hi</b>"]


@pytest.mark.asyncio
async def test_macro_read_returns_placeholder_form():
    """read hands the body back in storage form, so it can be pasted into macro_save again."""
    chan, _ = _text_recorder()
    tasks = await ctml_shell_test(
        chan,
        builder=_macro_builder(),
        ctml=(
            '<macro_save label="m">'
            f'<![CDATA[<a:say>{CDATA_START}<b>hi</b>{CDATA_END}</a:say>]]>'
            '</macro_save>\n'
            '<macro_read label="m"/>\n'
        ),
    )
    read_task = [t for t in tasks if t.caller_name() == "macro_read"][0]
    stored = str(read_task.result())
    assert CDATA_START in stored and CDATA_END in stored
    assert "<![CDATA[" not in stored


# -- 校验 -- #

def _failed(tasks, name):
    """The failed task named `name` — command errors land in task results, not raised out."""
    match = [t for t in tasks if t.caller_name() == name and not t.success()]
    assert match, f"no failed task named {name} in {[t.caller_name() for t in tasks]}"
    return match[0]


@pytest.mark.asyncio
async def test_validate_rejects_unknown_command():
    """A body referencing a command that does not exist fails at save time."""
    chan, _ = _recorder()
    tasks = await ctml_shell_test(
        chan,
        builder=_macro_builder(),
        ctml='<macro_save label="bad"><![CDATA[<nope:missing/>]]></macro_save>\n',
    )
    assert "nope" in str(_failed(tasks, "macro_save").exception())


@pytest.mark.asyncio
async def test_validate_rejects_empty_body():
    chan, _ = _recorder()
    tasks = await ctml_shell_test(
        chan,
        builder=_macro_builder(),
        ctml='<macro_save label="bad"><![CDATA[]]></macro_save>\n',
    )
    assert "empty" in str(_failed(tasks, "macro_save").exception())


# -- 文件态: micro -- #

def _write_micro(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.asyncio
async def test_macro_save_to_file_and_load(tmp_path):
    chan, calls = _recorder()
    await ctml_shell_test(
        chan,
        builder=_macro_builder(root=tmp_path),
        ctml=(
            '<macro_save label="greet" description="say hello" file="greet.ctml_micro.md">'
            '<![CDATA[<a:say text="from file"/>]]>'
            '</macro_save>\n'
            '<macro_load path="greet.ctml_micro.md"/>\n'
            '<macro ref="greet"/>\n'
        ),
    )
    assert calls == ["from file"]

    written = (tmp_path / "greet.ctml_micro.md").read_text(encoding="utf-8")
    assert "name: greet" in written
    assert "description: say hello" in written
    assert "<a:say text=\"from file\"/>" in written


@pytest.mark.asyncio
async def test_macro_invoke_by_file_path(tmp_path):
    """A micro file can be invoked directly, without loading it as a label."""
    _write_micro(
        tmp_path,
        f"direct{MICRO_SUFFIX}",
        "---\nname: direct\ndescription: invoke by path\n---\n<a:say text=\"direct\"/>\n",
    )
    chan, calls = _recorder()
    await ctml_shell_test(
        chan,
        builder=_macro_builder(root=tmp_path),
        ctml=f'<macro ref="direct{MICRO_SUFFIX}" is_file="true"/>\n',
    )
    assert calls == ["direct"]


@pytest.mark.asyncio
async def test_micro_lists_paths_with_descriptions(tmp_path):
    _write_micro(
        tmp_path,
        f"greet{MICRO_SUFFIX}",
        "---\nname: greet\ndescription: say hello\n---\n<a:say/>\n",
    )
    _write_micro(tmp_path, f"broken{MICRO_SUFFIX}", "<a:say/>\n")

    main = new_shell_main_channel()
    main.with_module(MacroStoreModule(root=tmp_path))
    async with main.bootstrap() as runtime:
        listing = await runtime.execute_command("micro", kwargs={"file": "."})
    assert f"greet{MICRO_SUFFIX}: say hello" in listing
    assert f"broken{MICRO_SUFFIX} (invalid" in listing


@pytest.mark.asyncio
async def test_file_path_cannot_escape_root(tmp_path):
    outside = tmp_path.parent / "outside.ctml_micro.md"
    outside.write_text("---\nname: x\n---\n<a:say/>\n", encoding="utf-8")

    main = new_shell_main_channel()
    main.with_module(MacroStoreModule(root=tmp_path))
    async with main.bootstrap() as runtime:
        for path in ("../outside.ctml_micro.md", str(outside)):
            with pytest.raises(CommandError, match="escapes the macro root"):
                await runtime.execute_command("macro_load", kwargs={"path": path})


@pytest.mark.asyncio
async def test_file_commands_need_root():
    """Without a root, file-space commands report unavailability instead of guessing."""
    main = new_shell_main_channel()
    main.with_module(MacroStoreModule())
    async with main.bootstrap() as runtime:
        with pytest.raises(CommandError):
            await runtime.execute_command("micro", kwargs={"file": "."})
