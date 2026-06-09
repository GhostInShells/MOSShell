"""Persistent interactive terminal session | 系统控制 | alpha

Example:
    from ghoshell_moss import new_shell_main_channel
    from ghoshell_moss.channels.shell_channel import new_shell_channel
    main = new_shell_main_channel()
    main.import_channels(new_shell_channel())
"""

from ghoshell_moss.core.blueprint.channel_builder import new_channel, MutableChannel
from ghoshell_moss.core.terminal.pexpect_session import PexpectSession

__all__ = ["new_shell_channel"]

_INSTRUCTION = """\
Persistent terminal session — your shell environment survives across commands.
Auto-starts bash on first sendline. venv, cwd, and env vars are preserved.

Commands:
  sendline(text, wait=5.0) — send text + Enter. Blocks until shell prompt
    reappears (or wait seconds elapse). Returns tail -200 lines of output.
    Each call creates a numbered segment.
  read_output(id, offset=0, limit=0) — read full or partial output of a
    segment by ID. limit=0 means no limit.
  sendcontrol(char) — send control character. 'c' for Ctrl-C, 'd' for
    Ctrl-D, 'z' for Ctrl-Z.
  close() — kill the terminal session.

Context messages show: cursor position, segment ID list, live terminal tail
(last 20 lines). When output is folded, use read_output(id) to get the full
content."""


def new_shell_channel(
    cwd: str = "",
    *,
    name: str = "shell",
    raw_mode: bool = False,
    session: PexpectSession | None = None,
) -> MutableChannel:
    """Create an interactive shell channel backed by pexpect.

    :param cwd: working directory (empty = process cwd)
    :param name: channel name (CTML tag name)
    :param raw_mode: if True, keep ANSI escape sequences in output
    :param session: pre-configured PexpectSession (or subclass instance).
                    If None, creates a default bash session.
    """
    if session is None:
        session = PexpectSession(cwd=cwd, raw_mode=raw_mode)

    chan = new_channel(
        name=name,
        description=(
            "Persistent terminal session. Shell environment survives across "
            "commands — venv, cwd, env vars preserved. Auto-starts on first "
            "sendline. Each command creates a numbered segment; use "
            "read_output(id) to retrieve full output when lines are folded."
        ),
    )

    # -- commands (sync methods, executed via asyncio.to_thread) ----------

    chan.build.command(
        name="sendline", always_observe=True, blocking=True
    )(session.sendline)

    chan.build.command(
        name="read_output", always_observe=True, blocking=True
    )(session.read_output)

    chan.build.command(
        name="sendcontrol", always_observe=False, blocking=True
    )(session.sendcontrol)

    chan.build.command(
        name="close", always_observe=False, blocking=True
    )(session.close)

    # -- context messages -------------------------------------------------

    @chan.build.context_messages
    def shell_context() -> list[str]:
        return [session.context_string()]

    # -- instruction ------------------------------------------------------

    @chan.build.instruction
    def shell_instruction() -> str:
        return _INSTRUCTION

    return chan
