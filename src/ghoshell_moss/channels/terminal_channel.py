"""Ghost 操作系统命令：bash.exec / bash.run / bash.read / bash.write | 系统控制 | alpha

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.terminal_channel import new_terminal_channel
    main = new_shell_main_channel()
    main.import_channels(new_terminal_channel("/path/to/workspace"))
"""

from ghoshell_moss.core.blueprint.channel_builder import new_channel, MutableChannel
from ghoshell_moss.core.terminal import SubprocessTerminal

__all__ = ["new_terminal_channel"]


def new_terminal_channel(
    workspace_root: str = "",
    *,
    name: str = "bash",
) -> MutableChannel:
    """Create a terminal channel: bash.exec, bash.run, bash.read, bash.write.

    ``exec`` blocks the channel until completion — use when the next step
    depends on the result.  ``run`` is non-blocking — use for background
    tasks.  The model is responsible for topological ordering: chain
    dependent commands via ``exec``, fire independent ones via ``run``.

    :param workspace_root: file I/O boundary root (empty = process cwd)
    :param name: channel name (CTML tag name)
    """
    terminal = SubprocessTerminal(workspace_root) if workspace_root else SubprocessTerminal()

    chan = new_channel(
        name=name,
        description="OS tools: run shell commands, read/write files within workspace",
    )

    # -- bash.exec (blocking) ------------------------------------------

    @chan.build.command(name="exec", always_observe=True, blocking=True)
    async def exec_cmd(cmd: str, *, cwd: str = "", timeout: float = 60.0) -> str:
        """Execute a shell command. Blocks until completion — use when you need
        the result before the next step.

        :param cmd: shell command to execute
        :param cwd: working directory (relative to workspace root, empty = root)
        :param timeout: max execution time in seconds (default 60)
        """
        r = terminal.exec(cmd, cwd=cwd, timeout=timeout)
        parts = []
        if r.stdout:
            parts.append(r.stdout.rstrip())
        if r.stderr:
            parts.append(f"[stderr]\n{r.stderr.rstrip()}")
        parts.append(f"[exit: {r.exit_code}]")
        return "\n".join(parts)

    # -- bash.run (non-blocking) ---------------------------------------

    @chan.build.command(name="run", always_observe=False, blocking=False)
    async def run_cmd(cmd: str, *, cwd: str = "", timeout: float = 60.0) -> str:
        """Fire-and-forget shell command. Does NOT block — use for background
        tasks when you don't need the result immediately.  Check back with
        ``exec`` if you need to verify completion.

        :param cmd: shell command to execute
        :param cwd: working directory (relative to workspace root, empty = root)
        :param timeout: max execution time in seconds (default 60)
        """
        r = terminal.exec(cmd, cwd=cwd, timeout=timeout)
        if not r.ok:
            return f"[run failed, exit: {r.exit_code}] {r.stderr.rstrip()}"
        return f"[running, exit: {r.exit_code}]"

    # -- bash.read -----------------------------------------------------

    @chan.build.command(name="read", always_observe=True)
    async def read_file(path: str) -> str:
        """Read a file within workspace. Returns content with line numbers.

        :param path: file path relative to workspace root
        """
        return terminal.read_file(path)

    # -- bash.write ----------------------------------------------------

    @chan.build.command(name="write")
    async def write_file(path: str, *, text__: str = "") -> str:
        """Write content to a file within workspace. Overwrites existing content.

        :param path: file path relative to workspace root
        :param text__: file content (via CTML open-close tag)
        """
        terminal.write_file(path, text__)
        return f"wrote {path} ({len(text__)} chars)"

    return chan
