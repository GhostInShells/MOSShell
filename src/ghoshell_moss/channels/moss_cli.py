"""Moss CLI 自举 channel — 去授权暴露 moss 自身 CLI | 集成 | beta

exec 命令用 @cli decorator 局部糖形式: 执行机器交给 decorator, channel 保留注入的
Subprocesses (facade=processes) 与生命周期, 展示格式化 (exit tail / friendly-empty)
留在 channel. 剥 moss/--ai 前缀由 input_filter 承担, 输出截断由 output_processor 承担.

which 是 channel 本地命令 (不 spawn 子进程): 报告 exec 真实使用的解释器与等价调用式,
让模型能把同一条 moss 命令带到别的环境 (bash / 脚本) 里复现, 并看出两处 env 是否分叉.

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.moss_cli import build_moss_cli_channel

    main = new_shell_main_channel()
    main.import_channels(build_moss_cli_channel(name="moss_cli"))
"""

import platform
import shutil
import sys
from pathlib import Path

from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.channel_builder import (
    MutableChannel,
    ChannelFactory,
    new_channel,
)
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.contracts.subprocesses import Subprocesses
from ghoshell_moss.decorators import cli

__all__ = ["new_moss_cli_channel", "build_moss_cli_channel"]

_RESULT_CHAR_CAP = 12_000
_DEFAULT_TIMEOUT = 120.0


def _strip_moss_prefix(argv: list[str]) -> list[str]:
    """入参过滤: 剥掉误带的 moss / --ai 前缀 (模型反射性输入)."""
    if argv and argv[0] == "moss":
        argv = argv[1:]
    if argv and argv[0] == "--ai":
        argv = argv[1:]
    return argv


def _cap(text: str) -> str:
    if len(text) <= _RESULT_CHAR_CAP:
        return text
    dropped = len(text) - _RESULT_CHAR_CAP
    return f"...[{dropped} chars truncated]\n" + text[-_RESULT_CHAR_CAP:]


def _cap_result(result: tuple[int, str, str]) -> tuple[int, str, str]:
    """出参加工: 截断 stdout, 形状不变 (三元组 → 三元组)."""
    code, stdout, stderr = result
    if len(stdout) > _RESULT_CHAR_CAP:
        dropped = len(stdout) - _RESULT_CHAR_CAP
        stdout = f"...[{dropped} chars truncated]\n" + stdout[-_RESULT_CHAR_CAP:]
    return (code, stdout, stderr)


def _moss_version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("ghoshell-moss")
    except PackageNotFoundError:
        return "unknown"


def new_moss_cli_channel(
    processes: Subprocesses,
    *,
    cwd: str = "",
    name: str = "moss_cli",
    description: str | None = None,
) -> MutableChannel:
    """纯组合原语: 在 Subprocesses 契约上组装 moss CLI 自举 channel.

    :param processes: Subprocesses 契约实例.
    :param cwd: moss CLI 子进程工作目录 (项目根). 空 = 进程 cwd.
    :param name: CTML 标签名.
    :param description: 覆盖默认描述.
    """
    default_cwd = Path(cwd).resolve() if cwd else Path.cwd()
    if description is None:
        description = (
            "Run moss CLI commands as a de-authorized channel — no bash, no shell "
            "escaping. Use it in place of invoking the moss CLI yourself when the "
            "environment offers no other shell capability."
        )

    chan = new_channel(name=name, description=description)
    owns_lifecycle: list[bool] = [False]

    @chan.build.startup
    async def _startup() -> None:
        if not processes.is_running():
            await processes.__aenter__()
            owns_lifecycle[0] = True

    @chan.build.close
    async def _close() -> None:
        if owns_lifecycle[0]:
            owns_lifecycle[0] = False
            await processes.__aexit__(None, None, None)

    @chan.build.instruction
    def instruction() -> str:
        return (
            "## moss CLI\n"
            "Run moss CLI commands through `exec` — pass ONLY the subcommand + args; "
            "the moss CLI and its --ai flag are supplied for you, and there is no shell "
            "to escape.\n"
            "When this environment gives you no other way to run shell commands, use "
            "this channel in place of invoking the moss CLI yourself.\n"
            "Before driving the moss CLI, load an entry point: `start` is the cognitive "
            "map (what MOSS is, what it can do, where to go next); `all-commands` is the "
            "full command tree. Append --help to a single command for its arguments.\n"
        )

    # 局部糖形式: decorator 做边界 (exec 模式 / 超时 / 过滤 / 加工), closure 绑定注入的
    # processes / cwd / timeout. 工具本体纯声明 (签名 + docstring), 运行时不被调用.
    @cli(
        [sys.executable, "-m", "ghoshell_moss.cli", "--ai"],
        name="moss-cli",
        facade=processes,
        cwd=default_cwd,
        timeout=_DEFAULT_TIMEOUT,
        input_filter=_strip_moss_prefix,
        output_processor=_cap_result,
    )
    async def exec_command(arguments: str = "") -> tuple[int, str, str]:
        """Run a moss CLI command via `python -m ghoshell_moss.cli --ai`.

        Pass ONLY the subcommand + args — never include 'moss' or '--ai'.
        """
        ...

    @chan.build.command(name="exec", blocking=True, always_observe=True)
    async def exec_cmd(text__: str = "") -> str:
        """Run a moss CLI command and wait for its result.

        :param text__: subcommand + arguments. e.g. 'codex get-interface ghoshell_moss.channels.moss_cli'.
                       NEVER include 'moss' or '--ai'.
        """
        if not text__.strip():
            return (
                "[exec] empty command — put the moss CLI subcommand inside the exec "
                "tag body, e.g. `codex blueprint`"
            )
        code, stdout, stderr = await exec_command(text__)
        parts: list[str] = []
        if stdout:
            parts.append(stdout.rstrip())
        if stderr:
            parts.append(f"[stderr]\n{stderr.rstrip()}")
        body = "\n".join(parts)
        return _cap(f"{body}\n[exit: {code}]".lstrip("\n"))

    @chan.build.command(name="which", blocking=True, always_observe=True)
    async def which() -> str:
        """Report how this channel actually invokes the moss CLI.

        Channel-local command — NOT a moss subcommand, so never route it through `exec`.
        Read it before copying a moss command into another environment (shell, script,
        MCP call), or to check whether the `moss` on PATH is this same install.
        """
        interpreter = sys.executable
        lines = [
            "How `exec` runs the moss CLI in this environment:",
            f"interpreter: {interpreter}  "
            f"({platform.python_implementation()} {platform.python_version()})",
            f"exec argv:   {interpreter} -m ghoshell_moss.cli --ai <subcommand> [args]",
        ]
        on_path = shutil.which("moss")
        if on_path is None:
            lines.append("equivalent:  no `moss` on PATH — use the exec argv form")
        elif Path(on_path).parent == Path(interpreter).parent:
            lines.append(
                f"equivalent:  {on_path} --ai <subcommand> [args]  (same environment)"
            )
        else:
            lines.append(
                f"equivalent:  {on_path} --ai <subcommand> [args]  "
                f"(DIFFERENT environment: {Path(on_path).parent})"
            )
        lines.append(f"moss:        {_moss_version()}")
        lines.append(f"cwd:         {default_cwd}")
        return "\n".join(lines)

    return chan


def build_moss_cli_channel(
    *,
    cwd: str = "",
    name: str = "moss_cli",
    description: str | None = None,
) -> ChannelFactory:
    """IoC 集成工厂: 从容器解析 Subprocesses 与项目根, 返回 ChannelFactory."""
    def factory(container: IoCContainer) -> Channel:
        resolved_cwd = cwd
        if not resolved_cwd:
            from ghoshell_moss.core.blueprint.project import Project
            project = Project.discover()
            resolved_cwd = str(project.root)
        processes = container.get(Subprocesses)
        if processes is None:
            from ghoshell_moss.core.subprocesses import SubprocessesImpl
            processes = SubprocessesImpl(cwd=resolved_cwd or None)
        return new_moss_cli_channel(
            processes,
            cwd=resolved_cwd,
            name=name,
            description=description,
        )
    return factory
