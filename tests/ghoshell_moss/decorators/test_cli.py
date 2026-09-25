"""cli decorator 的行为契约.

验证: 反射穿透 (code-as-prompt), -h/--help 拦截, input_filter 拒绝, facade 注入.

不覆盖直接执行子进程的用例: 它们走 cli 的懒加载 facade (Project.discover()),
会 seal 一个指向真实 workspace 的 Environment, 把 MOSS_* 写入 os.environ 并注册
全局单例, 污染后续测试 (test_environment_design / test_local_project 全量跑必挂).
直接执行子进程的语义留给集成测试.
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


def _reject_junk(argv: list[str]) -> list[str]:
    if any("junk" in a for a in argv):
        raise ValueError("junk is not allowed")
    return argv


@cli(["echo"], input_filter=_reject_junk)
async def echo_strict(arguments: str = "") -> tuple[int, str, str]:
    """echo, rejecting arguments containing 'junk'."""
    ...


@cli(["pwd"], facade=SubprocessesImpl(cwd="/tmp"))
async def pwd_injected(arguments: str = "") -> tuple[int, str, str]:
    """pwd via an injected facade rooted at /tmp."""
    ...


def test_signature_reflection():
    # 函数签名即工具签名: inspect / parse_function_interface 沿 __wrapped__ 穿透.
    sig = inspect.signature(echo)
    assert str(sig) == "(arguments: str = '') -> tuple[int, str, str]"
    assert echo._cli_name == "echo"


def test_name_override():
    assert echo_named._cli_name == "echoer"


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
async def test_input_filter_rejects():
    with pytest.raises(ValueError):
        await echo_strict("junk")


@pytest.mark.asyncio
async def test_facade_injection_used():
    # 注入的 facade (默认 cwd=/tmp) 生效 — 若落到 project IoC 会返回项目根.
    code, stdout, stderr = await pwd_injected("")
    assert code == 0
    assert stdout.strip() == _TMP
