"""Moss CLI 自举 channel — 去授权暴露 moss 自身 CLI | 集成 | beta

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.moss_cli import build_moss_cli_channel

    main = new_shell_main_channel()
    main.import_channels(build_moss_cli_channel(name="moss_cli"))
"""

import asyncio
import shlex
import sys
from pathlib import Path

from ghoshell_container import IoCContainer

from ghoshell_moss.core.blueprint.channel_builder import (
    MutableChannel,
    ChannelFactory,
    new_channel,
)
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.contracts.subprocesses import CaptureSpec, ManagedProcess, Subprocesses

__all__ = ["new_moss_cli_channel", "build_moss_cli_channel"]

_EXEC_BUFFER_LINES = 200
_RESULT_CHAR_CAP = 12_000
_DEFAULT_TIMEOUT = 120.0

def _parse_command(text: str) -> list[str]:
    """解析命令串为 argv. 剥掉误带的 moss / --ai 前缀."""
    args = shlex.split(text)
    if args and args[0] == "moss":
        args.pop(0)
    if args and args[0] == "--ai":
        args.pop(0)
    return args


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
            "Moss CLI self-control. Run moss commands (python -m ghoshell_moss.cli) "
            "as a de-authorized channel — no bash, no shell escaping."
        )

    chan = new_channel(name=name, description=description)
    owns_lifecycle: list[bool] = [False]

    def _output_text(managed: ManagedProcess) -> str:
        parts: list[str] = []
        if managed.output is not None:
            stdout = managed.output.stdout()
            if stdout:
                parts.append(stdout.rstrip())
            stderr = managed.output.stderr()
            if stderr:
                parts.append(f"[stderr]\n{stderr.rstrip()}")
        return "\n".join(parts)

    def _cap(text: str) -> str:
        if len(text) <= _RESULT_CHAR_CAP:
            return text
        dropped = len(text) - _RESULT_CHAR_CAP
        return f"...[{dropped} chars truncated]\n" + text[-_RESULT_CHAR_CAP:]

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
            "## moss_cli channel\n"
            "Run any moss command via `exec` — pass ONLY the subcommand + args.\n"
            "'python -m ghoshell_moss.cli --ai' is prepended automatically.\n"
            "First frame: <moss_cli:exec>moss start</moss_cli:exec> to load the cognitive map.\n"
            "Example: <moss_cli:exec>codex get-interface ghoshell_moss.channels.moss_cli</moss_cli:exec>\n"
        )

    @chan.build.command(name="exec", blocking=True, always_observe=True)
    async def exec_command(text__: str = "", timeout: float = _DEFAULT_TIMEOUT) -> str:
        """Run a moss CLI command and wait for its result.

        :param text__: subcommand + arguments. e.g. 'codex get-interface ghoshell_moss.channels.moss_cli'.
                       NEVER include 'moss' or '--ai'.
        :param timeout: seconds to wait before force-stopping the subprocess.
        """
        args = _parse_command(text__)
        if not args:
            return (
                "[exec] empty command — put the moss subcommand inside the tag body, "
                "e.g. <moss_cli:exec>codex blueprint</moss_cli:exec>"
            )

        managed = await processes.execute(
            sys.executable, "-m", "ghoshell_moss.cli", "--ai", *args,
            name="moss-cli",
            cwd=str(default_cwd),
            capture=CaptureSpec(buffer_lines=_EXEC_BUFFER_LINES),
        )
        try:
            await asyncio.wait_for(managed.process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            await managed.stop()
            body = _output_text(managed)
            return _cap(f"{body}\n[timeout after {timeout}s, stopped]".lstrip("\n"))
        if managed.output is not None:
            await managed.output.wait_drained()
        body = _output_text(managed)
        tail = f"[exit: {managed.process.returncode}]"
        return _cap(f"{body}\n{tail}".lstrip("\n"))

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
