"""cli decorator 的行为契约.

验证: 反射穿透 (code-as-prompt), 单独运行, -h/--help 拦截, channel 机制运行,
以及参数面 — facade 注入 / cwd / timeout / input_filter / output_processor.
"""

import inspect
from pathlib import Path

import pytest

from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.core.subprocesses._impl import SubprocessesImpl
from ghoshell_moss.decorators import cli

_TMP = str(Path("/tmp").resolve())


@cli(["echo"])
async def echo(arguments: str = "") -> tuple[int, str, str]:
    """echo the arguments back to stdout."""
    ...


@cli(["echo"], help=lambda: "usage: echo <text...>")
async def echo_help(arguments: str = "") -> tuple[int, str, str]:
    """echo with -h/--help intercepted."""
    ...


@cli(["echo"], name="echoer")
async def echo_named(arguments: str = "") -> tuple[int, str, str]:
    """echo under an explicit tool name."""
    ...


def _strip_say(argv: list[str]) -> list[str]:
    return argv[1:] if argv and argv[0] == "say" else argv


@cli(["echo"], input_filter=_strip_say)
async def echo_say(arguments: str = "") -> tuple[int, str, str]:
    """echo, stripping a leading 'say' token."""
    ...


def _reject_junk(argv: list[str]) -> list[str]:
    if any("junk" in a for a in argv):
        raise ValueError("junk is not allowed")
    return argv


@cli(["echo"], input_filter=_reject_junk)
async def echo_strict(arguments: str = "") -> tuple[int, str, str]:
    """echo, rejecting arguments containing 'junk'."""
    ...


def _prepend_note(result: tuple[int, str, str]) -> tuple[int, str, str]:
    code, stdout, stderr = result
    return (code, "[note] " + stdout, stderr)


@cli(["echo"], output_processor=_prepend_note)
async def echo_annotated(arguments: str = "") -> tuple[int, str, str]:
    """echo with an output annotation prepended."""
    ...


@cli(["pwd"], cwd="/tmp")
async def pwd_in_tmp(arguments: str = "") -> tuple[int, str, str]:
    """pwd with a fixed cwd."""
    ...


@cli(["pwd"], facade=SubprocessesImpl(cwd="/tmp"))
async def pwd_injected(arguments: str = "") -> tuple[int, str, str]:
    """pwd via an injected facade rooted at /tmp."""
    ...


@cli(["sleep"], timeout=0.05)
async def sleepy(arguments: str = "") -> tuple[int, str, str]:
    """sleep with a 50ms timeout."""
    ...


def test_signature_reflection():
    # 函数签名即工具签名: inspect / parse_function_interface 沿 __wrapped__ 穿透.
    sig = inspect.signature(echo)
    assert str(sig) == "(arguments: str = '') -> tuple[int, str, str]"
    assert echo._cli_name == "echo"


def test_name_override():
    assert echo_named._cli_name == "echoer"


@pytest.mark.asyncio
async def test_standalone_run():
    code, stdout, stderr = await echo("hello world")
    assert code == 0
    assert stdout.strip() == "hello world"
    assert stderr == ""


@pytest.mark.asyncio
async def test_help_intercepted_without_spawn():
    code, stdout, stderr = await echo_help("--help")
    assert code == 0
    assert stdout.strip() == "usage: echo <text...>"
    assert stderr == ""


@pytest.mark.asyncio
async def test_py_command_reflects_tool_signature():
    # PyCommand 反射装饰后的工具: 界面就是工具的签名 + docstring.
    chan = PyChannel(name="cli")
    cmd = chan.build.command(return_command=True)(echo)
    meta = cmd.meta()
    assert "arguments" in meta.interface
    assert "-> tuple[int, str, str]" in meta.interface


@pytest.mark.asyncio
async def test_runs_through_channel():
    chan = PyChannel(name="cli")
    chan.build.command()(echo)

    async with chan.bootstrap() as runtime:
        cmd = runtime.get_command("echo")
        assert cmd is not None
        code, stdout, stderr = await cmd("channel echo")
        assert code == 0
        assert stdout.strip() == "channel echo"
        assert stderr == ""


@pytest.mark.asyncio
async def test_input_filter_strips_token():
    # input_filter 改写 argv: 'say hello' → 'hello'.
    code, stdout, stderr = await echo_say("say hello")
    assert code == 0
    assert stdout.strip() == "hello"
    code, stdout, stderr = await echo_say("hello")
    assert stdout.strip() == "hello"


@pytest.mark.asyncio
async def test_input_filter_rejects():
    with pytest.raises(ValueError):
        await echo_strict("junk")


@pytest.mark.asyncio
async def test_output_processor_normalizes():
    code, stdout, stderr = await echo_annotated("hi")
    assert code == 0
    assert stdout.startswith("[note] ")
    assert "hi" in stdout


@pytest.mark.asyncio
async def test_cwd_param():
    code, stdout, stderr = await pwd_in_tmp("")
    assert code == 0
    assert stdout.strip() == _TMP


@pytest.mark.asyncio
async def test_facade_injection_used():
    # 注入的 facade (默认 cwd=/tmp) 生效 — 若落到 project IoC 会返回项目根.
    code, stdout, stderr = await pwd_injected("")
    assert code == 0
    assert stdout.strip() == _TMP


@pytest.mark.asyncio
async def test_timeout_returns_124_and_stops():
    import time

    start = time.monotonic()
    code, stdout, stderr = await sleepy("1")
    elapsed = time.monotonic() - start
    assert code == 124
    assert "timeout" in stderr
    assert elapsed < 0.5, f"timeout took {elapsed:.2f}s, expected graceful stop"

