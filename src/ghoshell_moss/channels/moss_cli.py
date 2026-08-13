"""Moss CLI 自举 channel — 去授权暴露 moss 自身 CLI | 集成 | beta

exec 命令用 @cli decorator 局部糖形式: 执行机器交给 decorator, channel 保留注入的
Subprocesses (facade=processes) 与生命周期, 展示格式化 (exit tail / friendly-empty)
留在 channel. 剥 moss/--ai 前缀由 input_filter 承担, 输出截断由 output_processor 承担.

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.moss_cli import build_moss_cli_channel

    main = new_shell_main_channel()
    main.import_channels(build_moss_cli_channel(name="moss_cli"))
"""

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
                "[exec] empty command — put the moss subcommand inside the tag body, "
                "e.g. <moss_cli:exec>codex blueprint</moss_cli:exec>"
            )
        code, stdout, stderr = await exec_command(text__)
        parts: list[str] = []
        if stdout:
            parts.append(stdout.rstrip())
        if stderr:
            parts.append(f"[stderr]\n{stderr.rstrip()}")
        body = "\n".join(parts)
        return _cap(f"{body}\n[exit: {code}]".lstrip("\n"))

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
