"""Terminal implementations — OS-level tools for Ghost (bash + file I/O)."""

from .subprocess_terminal import CommandResult, SubprocessTerminal

__all__ = ["CommandResult", "SubprocessTerminal"]
