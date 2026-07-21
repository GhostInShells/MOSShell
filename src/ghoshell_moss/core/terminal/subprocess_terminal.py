"""Subprocess-based Terminal — command execution + file I/O.

Zero MOSS dependencies. Clean interface — Phase 2 extracts protocol to contracts.
"""

import subprocess
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["CommandResult", "SubprocessTerminal"]


@dataclass
class CommandResult:
    exit_code: int = -1
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.exit_code == 0


class SubprocessTerminal:
    """Operating system tools for Ghost: bash execution + file read/write.

    Phase 1 concrete class. Method signatures are the implicit protocol —
    Phase 2 extracts ABC when pexpect backend is added.
    """

    def __init__(self, root: str | Path = ""):
        self._root = Path(root).resolve() if root else Path.cwd()

    # -- bash ---------------------------------------------------------

    def exec(
        self, cmd: str, *, cwd: str = "", timeout: float = 60.0
    ) -> CommandResult:
        """Execute a shell command, block until completion.

        :param cmd: shell command to execute
        :param cwd: working directory (empty = process cwd)
        :param timeout: max execution time in seconds
        """
        work_dir = str(self._resolve_cwd(cwd))
        try:
            r = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
            )
            return CommandResult(
                exit_code=r.returncode,
                stdout=r.stdout,
                stderr=r.stderr,
            )
        except subprocess.TimeoutExpired as e:
            return CommandResult(
                exit_code=-1,
                stdout=e.stdout.decode() if e.stdout else "",
                stderr=f"timed out after {timeout}s\n" + (
                    e.stderr.decode() if e.stderr else ""
                ),
            )

    # -- file ---------------------------------------------------------

    def read_file(self, path: str) -> str:
        """Read file content with line numbers.

        :param path: file path (relative to root or absolute within root)
        :return: content prefixed with line numbers like "     1|import os"
        """
        target = self._safe_path(path)
        if not target.is_file():
            raise FileNotFoundError(f"not found: {path}")
        lines = target.read_text(encoding="utf-8").splitlines()
        width = len(str(len(lines)))
        return "\n".join(
            f"{i + 1:>{width}}|{line}" for i, line in enumerate(lines)
        )

    def write_file(self, path: str, content: str) -> None:
        """Write file content (overwrite mode).

        :param path: file path (relative to root or absolute within root)
        :param content: file content
        """
        target = self._safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    # -- internal -----------------------------------------------------

    def _resolve_cwd(self, cwd: str) -> Path:
        if not cwd:
            return self._root
        p = Path(cwd)
        if p.is_absolute():
            return p
        return self._root / p

    def _safe_path(self, path: str) -> Path:
        """Resolve path within root. Reject traversal attempts."""
        p = Path(path)
        if p.is_absolute():
            target = p.resolve()
        else:
            target = (self._root / p).resolve()
        root_resolved = self._root.resolve()
        if not str(target).startswith(str(root_resolved)):
            raise ValueError(f"path escape denied: {path!r}")
        return target
