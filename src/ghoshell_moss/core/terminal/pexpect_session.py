"""Pexpect-based persistent interactive terminal session.

Zero MOSS dependencies. Cursor-based output segmentation for channel integration.

Extensible via subclass hooks: override ``_shell_args``, ``_shell_env``,
``_prompt_marker`` to support different shells (bash, zsh, etc.).
"""

import os
import re
import threading
from pathlib import Path

from ghoshell_moss.depends import depend_host

depend_host()
import pexpect

__all__ = ["PexpectSession"]

_ANSI_RE = re.compile(
    r"\x1b\[[0-9;?]*[a-zA-Z]"               # CSI (including DEC private modes)
    r"|\x1b[>=]"                               # two-char escapes
    r"|\x1b\][^\x07\x1b]*(\x07|\x1b\\)"      # OSC
)
_BACKSPACE_RE = re.compile(r".[\b]")  # char + backspace = erased char
_LINE_EDIT_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")  # control chars except \t\n
_SEGMENT_TAIL = 200
_LIVE_TAIL = 20
_PROMPT_END = "__MOSS_EOP__"


def _clean_output(text: str) -> str:
    """Strip ANSI, backspace-edits, control chars, and normalize whitespace."""
    text = _ANSI_RE.sub("", text)
    # remove backspace-erased characters (shell line editing artifacts)
    text = _BACKSPACE_RE.sub("", text)
    # strip remaining low control chars except tab and newline
    text = _LINE_EDIT_RE.sub("", text)
    # normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse >2 consecutive blank lines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip("\n")


class PexpectSession:
    """Persistent interactive terminal session backed by pexpect.

    Each ``sendline`` creates a numbered segment. Output is stored in memory
    and retrieved via ``read_output(id)``. Context state exposed via
    ``context_string()`` for channel context_messages.

    Subclass hooks for shell variants::

        class ZshSession(PexpectSession):
            def _shell_args(self):
                return ["zsh", "-f"]

            def _shell_env(self):
                return {"PROMPT": f"{self._prompt_marker()} "}

    Usage::

        session = PexpectSession()
        result = session.sendline("pytest tests/ -x")
        print(result)  # tail -200 + fold marker
        full = session.read_output(1)  # full segment #1
        session.close()
    """

    def __init__(
        self,
        cwd: str | Path = "",
        *,
        raw_mode: bool = False,
        startup: list[str] | None = None,
    ):
        self._cwd = str(Path(cwd).resolve()) if cwd else os.getcwd()
        self._raw_mode = raw_mode
        self._startup = startup or []
        self._child: pexpect.spawn | None = None
        self._lock = threading.Lock()

        self._segments: dict[int, str] = {}
        self._cursor: int = 0

    # -- subclass hooks ---------------------------------------------------

    def _prompt_marker(self) -> str:
        return _PROMPT_END

    def _shell_args(self) -> list[str]:
        """Spawn args — override for different shells."""
        return ["bash", "--norc", "--noprofile"]

    def _shell_env(self) -> dict[str, str]:
        """Extra env vars — override to set shell-specific prompt."""
        return {"PS1": f"{self._prompt_marker()} "}

    # -- public properties ------------------------------------------------

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def is_alive(self) -> bool:
        return self._child is not None and self._child.isalive()

    @property
    def segment_ids(self) -> list[int]:
        return sorted(self._segments.keys())

    # -- lifecycle -------------------------------------------------------

    def _ensure_spawned(self) -> None:
        if self._child is not None:
            return
        with self._lock:
            if self._child is not None:
                return
            args = self._shell_args()
            env = {**os.environ, **self._shell_env()}
            self._child = pexpect.spawn(
                args[0],
                args=args[1:],
                cwd=self._cwd,
                encoding="utf-8",
                dimensions=(24, 160),
                env=env,
            )
            try:
                self._child.expect(self._prompt_marker(), timeout=5)
            except pexpect.TIMEOUT:
                pass

            for cmd in self._startup:
                self._child.sendline(cmd)
                try:
                    self._child.expect(self._prompt_marker(), timeout=10)
                except pexpect.TIMEOUT:
                    pass

    def close(self) -> str:
        if self._child is None:
            return "[session not started]"
        with self._lock:
            if self._child is not None:
                self._child.close()
                self._child = None
        return "[session closed]"

    # -- commands --------------------------------------------------------

    def sendline(self, text: str, *, wait: float = 5.0) -> str:
        """Send text + newline. Wait for prompt, create segment, return tail.

        :param text: command text to send (newline appended automatically)
        :param wait: seconds to wait for prompt.
        """
        self._ensure_spawned()

        with self._lock:
            self._cursor += 1
            seg_id = self._cursor

            self._child.sendline(text)

            try:
                self._child.expect(self._prompt_marker(), timeout=wait)
            except pexpect.TIMEOUT:
                pass

            raw = self._child.before or ""
            if not self._raw_mode:
                raw = _clean_output(raw)

            self._segments[seg_id] = raw
            return self._format_result(seg_id, raw)

    def read_output(
        self, id: int, *, offset: int = 0, limit: int = 0
    ) -> str:
        """Read full or partial output of a segment by ID.

        :param id: segment ID
        :param offset: start from this line
        :param limit: max lines (0 = no limit)
        """
        seg = self._segments.get(id)
        if seg is None:
            available = self.segment_ids
            return (
                f"[segment #{id} not found. "
                f"Available: {available if available else 'none'}]"
            )

        lines = seg.splitlines()
        if offset:
            lines = lines[offset:]
        if limit > 0:
            lines = lines[:limit]

        return "\n".join(lines)

    def sendcontrol(self, char: str) -> str:
        """Send control character.

        :param char: 'c' for Ctrl-C, 'd' for Ctrl-D, 'z' for Ctrl-Z
        """
        self._ensure_spawned()
        with self._lock:
            self._child.sendcontrol(char)
        return f"[sent ^{char}]"

    # -- context ---------------------------------------------------------

    def context_string(self) -> str:
        """Generate context text for channel context_messages."""
        if self._child is None:
            return "[shell] not started"

        alive = "alive" if self.is_alive else "dead"
        segments = self.segment_ids
        lines = [
            f"[shell] bash | cwd:{self._cwd} | "
            f"cursor:{self._cursor} | {alive}",
        ]

        if segments:
            seg_range = (
                f"{segments[0]}..{segments[-1]}"
                if len(segments) > 1
                else str(segments[0])
            )
            lines.append(
                f"  segments: [{seg_range}] "
                f"(read_output(id) for full content)"
            )

        # live tail from last segment
        if segments:
            last = self._segments[segments[-1]]
            tail = last.splitlines()[-_LIVE_TAIL:]
            lines.append("  --- live (tail -20) ---")
            lines.extend(f"  {line}" for line in tail)
            lines.append("  -----------------------")

        return "\n".join(lines)

    # -- internal --------------------------------------------------------

    def _format_result(self, seg_id: int, output: str) -> str:
        lines = output.splitlines()
        total = len(lines)

        header = f"[segment #{seg_id}, {total} lines"
        if total > _SEGMENT_TAIL:
            header += f", tail -{_SEGMENT_TAIL} shown"
        header += "]"

        if total <= _SEGMENT_TAIL:
            return f"{header}\n{output}"

        shown = "\n".join(lines[-_SEGMENT_TAIL:])
        folded = total - _SEGMENT_TAIL
        return (
            f"{header}\n{shown}\n"
            f"... {folded} lines folded. "
            f"read_output(id={seg_id}) for full content."
        )
