"""进程控制: bash.exec / bash.run / bash.read_output / bash.stop | 系统控制 | alpha

Example:
    # workspace-integrated: registered via manifests providers
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.terminal_channel import build_terminal_channel
    main = new_shell_main_channel()
    main.import_channels(build_terminal_channel())
    # rename / retag:
    main.import_channels(build_terminal_channel(name="sh"))

    # direct composition (tests / hand-wired scripts)
    from ghoshell_moss.channels.terminal_channel import new_terminal_channel
    from ghoshell_moss.core.subprocesses._impl import SubprocessesImpl
    main.import_channels(new_terminal_channel(SubprocessesImpl()))
"""

from __future__ import annotations

import getpass
import locale
import platform
from datetime import datetime
from pathlib import Path

from ghoshell_container import IoCContainer

from ghoshell_moss.contracts.subprocesses import (
    CaptureSpec,
    ManagedProcess,
    Subprocesses,
)
from ghoshell_moss.core.blueprint.channel_builder import (
    ChannelFactory,
    CommandUtil,
    MutableChannel,
    new_channel,
)
from ghoshell_moss.core.blueprint.mindflow import Priority
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.message.message import Addition

__all__ = [
    "new_terminal_channel",
    "build_terminal_channel",
    "ExitNotifyAddition",
]

_NOTIFY_LEVELS: dict[str, Priority] = {
    "background": Priority.BACKGROUND,
    "info": Priority.INFO,
    "notice": Priority.NOTICE,
    "warning": Priority.WARNING,
}

_EXEC_BUFFER_LINES = 200
_RUN_BUFFER_LINES = 100
_RESULT_CHAR_CAP = 12_000
_DEAD_KEEP = 20


class ExitNotifyAddition(Addition):
    """后台进程退出通知的显著程度, 绑在 ProcessMeta.additional 上随 meta 流转."""

    level: str = "background"

    @classmethod
    def keyword(cls) -> str:
        return "terminal_exit_notify"


# -- factory (IoC integration, core API) -------------------------------------


def build_terminal_channel(
    *,
    cwd: str = "",
    name: str = "bash",
    description: str | None = None,
) -> ChannelFactory:
    """High-order factory: configure name/description/cwd, return a ChannelFactory.

    Configuration (name/description/cwd) is decoupled from IoC — because
    ``ChannelFactory`` is ``(IoCContainer) -> Channel``, config has no place
    in that signature. Call ``build_terminal_channel(...)`` at declaration
    time to get a factory, hand the factory to ``import_channels``.

    Resolves Subprocesses from the container (project environments register a
    per-Project singleton via ProjectSubprocessesProvider); falls back to a
    private SubprocessesImpl whose lifecycle the channel then manages itself.

    Default cwd resolution (highest → lowest priority):

    1. Explicit ``cwd`` argument here.
    2. ``Matrix.project_home`` from container (the moss project root, NOT
       the workspace ``.moss/`` subdir — MCP scenarios spend most of their
       time on repo-root paths).
    3. Process cwd (falls through in ``new_terminal_channel``).

    :param cwd: default working directory for commands. Empty = matrix
        project_home if available, else process cwd.
    :param name: CTML tag name (default ``bash``).
    :param description: Override the built-in description; ``None`` = default.
    """

    def factory(container: IoCContainer) -> Channel:
        resolved_cwd = cwd
        if not resolved_cwd:
            from ghoshell_moss.core.blueprint.matrix import Matrix

            matrix = container.get(Matrix)
            if matrix is not None:
                resolved_cwd = str(matrix.project_home)
        processes = container.get(Subprocesses)
        if processes is None:
            from ghoshell_moss.core.subprocesses._impl import SubprocessesImpl
            processes = SubprocessesImpl(cwd=resolved_cwd or None)
        return new_terminal_channel(
            processes,
            cwd=resolved_cwd,
            name=name,
            description=description,
        )

    return factory


# -- composition primitive (contract consumer, no IoC knowledge) ------------


def new_terminal_channel(
    processes: Subprocesses,
    *,
    cwd: str = "",
    name: str = "bash",
    description: str | None = None,
) -> MutableChannel:
    """Compose a process-control channel over the Subprocesses contract.

    Pure composition primitive — knows nothing about IoC. Lifecycle rule:
    if ``processes.is_running()`` at channel startup, the instance belongs
    to an outer owner (e.g. matrix.processes shared singleton) and is used
    without lifecycle management; otherwise the channel enters/exits it
    within its own startup/close.

    Ownership isolation: the channel only shows and only stops processes
    it spawned itself (a shared Subprocesses may also carry cell processes
    governed elsewhere).

    :param processes: Subprocesses contract instance.
    :param cwd: default working directory for spawned commands
        (empty = current process cwd). Relative ``cwd`` command args
        resolve against it.
    :param name: CTML tag name.
    :param description: Override the built-in description; ``None`` = default.
    """
    default_cwd = Path(cwd).resolve() if cwd else Path.cwd()
    if description is None:
        description = (
            "Shell command execution. No session state — pass cwd explicitly."
        )

    chan = new_channel(
        name=name,
        description=description,
    )

    spawned: dict[int, ManagedProcess] = {}
    owns_lifecycle: list[bool] = [False]

    def _resolve_cwd(arg: str) -> str:
        if not arg:
            return str(default_cwd)
        path = Path(arg)
        if not path.is_absolute():
            path = default_cwd / path
        return str(path)

    def _prune_dead() -> None:
        dead = [i for i, m in spawned.items() if m.process.returncode is not None]
        for index in dead[:-_DEAD_KEEP]:
            spawned.pop(index, None)

    def _cap(text: str, index: int) -> str:
        if len(text) <= _RESULT_CHAR_CAP:
            return text
        dropped = len(text) - _RESULT_CHAR_CAP
        return (
            f"...[{dropped} chars truncated — read_output({index}) "
            f"or see log files]\n" + text[-_RESULT_CHAR_CAP:]
        )

    def _output_text(managed: ManagedProcess, *, offset: int = 0, limit: int = 0) -> str:
        parts: list[str] = []
        if managed.output is not None:
            stdout = managed.output.stdout(offset=offset, limit=limit)
            if stdout:
                parts.append(stdout.rstrip())
            stderr = managed.output.stderr()
            if stderr:
                parts.append(f"[stderr]\n{stderr.rstrip()}")
        return "\n".join(parts)

    # -- lifecycle ---------------------------------------------------------

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

    # -- exec (机制①: 同步阻塞, 占据 channel FIFO) --------------------------

    @chan.build.command(name="exec", blocking=True, always_observe=True)
    async def exec_cmd(text__: str = "", *, cwd: str = "", timeout: float = 60.0) -> str:
        """Run a shell command and wait for its result. Occupies this channel until done."""
        import asyncio

        cmd = text__.strip()
        if not cmd:
            return CommandUtil.observe(
                "[exec] empty command — put the shell line inside the tag body, "
                "e.g. <bash:exec><![CDATA[ls foo]]></bash:exec>"
            )
        managed = await processes.shell(
            cmd,
            cwd=_resolve_cwd(cwd),
            capture=CaptureSpec(buffer_lines=_EXEC_BUFFER_LINES),
        )
        index = managed.meta.index
        # exec 是一次性同步命令, 结果已完整返回 — 不进 spawned dict.
        # read_output / stop 只作用于 run 起的后台进程.
        try:
            await asyncio.wait_for(managed.process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            await managed.stop()
            body = _output_text(managed)
            return _cap(
                f"{body}\n[#{index} timeout after {timeout}s, stopped]".lstrip("\n"),
                index,
            )
        if managed.output is not None:
            await managed.output.wait_drained()
        body = _output_text(managed)
        tail = f"[#{index} exit: {managed.process.returncode}]"
        return _cap(f"{body}\n{tail}".lstrip("\n"), index)

    # -- run (机制③: 全异步, spawn 即返, 结束异步通知) -----------------------

    @chan.build.command(name="run", blocking=False, always_observe=True)
    async def run_cmd(
        text__: str = "",
        *,
        name: str = "",
        cwd: str = "",
        notify: str = "background",
    ) -> str:
        """Start a background process. Returns immediately; use read_output(index) to inspect."""
        cmd = text__.strip()
        if not cmd:
            return CommandUtil.observe(
                "[run] empty command — put the shell line inside the tag body, "
                "e.g. <bash:run><![CDATA[python worker.py]]></bash:run>"
            )
        level = notify if notify in _NOTIFY_LEVELS else "background"
        managed = await processes.shell(
            cmd,
            name=name or None,
            cwd=_resolve_cwd(cwd),
            capture=CaptureSpec(buffer_lines=_RUN_BUFFER_LINES),
        )
        ExitNotifyAddition(level=level).set(managed.meta)
        index = managed.meta.index
        spawned[index] = managed
        _prune_dead()

        async def _notify_on_exit() -> None:
            await managed.process.wait()
            if managed.output is not None:
                await managed.output.wait_drained()
            from ghoshell_moss.core.blueprint.session import Session

            session = CommandUtil.get_contract(Session)
            if session is None:
                return
            addition = ExitNotifyAddition.read(managed.meta)
            priority = _NOTIFY_LEVELS[addition.level if addition else "background"]
            label = managed.meta.name
            code = managed.process.returncode
            content = f"[{chan.name()} #{index}] '{label}' ended, exit: {code}"
            if code not in (0, None) and managed.output is not None:
                stderr_tail = managed.output.stderr(limit=10)
                if stderr_tail:
                    content += f"\n[stderr tail]\n{stderr_tail.rstrip()}"
                content += f"\nread_output({index}) for full output"
            session.add_input_signal(
                content,
                description=f"background process '{label}' ended",
                priority=priority.value,
            )

        CommandUtil.create_task(_notify_on_exit())
        return f"[#{index} started] '{managed.meta.name}' pid={managed.meta.pid}"

    # -- read_output (机制②: nonblocking 快命令) ----------------------------

    @chan.build.command(name="read_output", blocking=False, always_observe=True)
    async def read_output(index: int, *, offset: int = 0, limit: int = 0) -> str:
        """Read stdout/stderr from a process (memory tail window). Works for running and finished."""
        managed = spawned.get(index)
        if managed is None:
            return CommandUtil.observe(
                f"[#{index}] unknown process index (yours: {sorted(spawned)})"
            )
        body = _output_text(managed, offset=offset, limit=limit)
        code = managed.process.returncode
        state = "running" if code is None else f"exit: {code}"
        return _cap(f"{body}\n[#{index} {state}]".lstrip("\n"), index)

    # -- stop (机制②: nonblocking 快命令) -----------------------------------

    @chan.build.command(name="stop", blocking=False, always_observe=False)
    async def stop(index: int, *, timeout: float = 5.0) -> str:
        """Stop a background process (SIGTERM → grace → killpg)."""
        managed = spawned.get(index)
        if managed is None:
            return CommandUtil.observe(
                f"[#{index}] unknown process index (yours: {sorted(spawned)})"
            )
        await managed.stop(timeout=timeout)
        return f"[#{index} stopped, exit: {managed.process.returncode}]"

    # -- context messages (own-only 后台任务简表, run 起的进程) ------------

    @chan.build.context_messages
    def terminal_context() -> list[str]:
        running: list[str] = []
        exited: list[str] = []
        now = datetime.now().timestamp()
        for index, managed in spawned.items():
            meta = managed.meta
            cmd_preview = meta.command if len(meta.command) <= 60 else meta.command[:57] + "..."
            if managed.process.returncode is None:
                uptime = int(now - meta.created)
                running.append(
                    f"  #{index} '{meta.name}' pid={meta.pid} uptime={uptime}s "
                    f"cmd={cmd_preview!r}"
                )
            else:
                exited.append(
                    f"  #{index} '{meta.name}' exit={managed.process.returncode} "
                    f"cmd={cmd_preview!r}"
                )
        if not running and not exited:
            return []
        lines = [f"[{chan.name()}] background processes (read_output(index) to inspect):"]
        if running:
            lines.append("running:")
            lines.extend(running)
        if exited:
            lines.append("recently exited:")
            lines.extend(exited[-5:])
        return ["\n".join(lines)]

    # -- instruction -------------------------------------------------------

    lang, encoding = locale.getlocale()
    system_context = (
        "[System Context]\n"
        f"OS: {platform.platform(terse=True)}\n"
        f"User: {getpass.getuser()}\n"
        f"Default cwd: {default_cwd}\n"
        f"TimeZone: {datetime.now().astimezone().tzinfo}\n"
        f"Lang: {lang} / Encoding: {encoding}"
    )

    @chan.build.instruction
    def terminal_instruction() -> str:
        return (
            "Shell command execution. No session state — each command gets "
            "an explicit cwd. Background processes appear in context.\n\n"
            + system_context
        )

    return chan
