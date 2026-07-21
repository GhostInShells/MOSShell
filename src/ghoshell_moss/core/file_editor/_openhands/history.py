"""In-memory history for file edits — supports ``undo_edit``.

MOSS vendor patch (see UPSTREAM.md):

Upstream used a disk-backed ``FileCache`` (JSON files under a temp dir) to
persist history across restarts. We drop that: undo history is per-process
session state, and a session that dies loses its editor context anyway.
Backing to disk added file IO on every edit for a feature the model never
crosses process boundaries to use.

API kept identical (``add_history`` / ``pop_last_history`` / ``clear_history``
/ ``get_all_history``) so nothing in the vendor tree needs to know the storage
changed.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Deque


class FileHistoryManager:
    """Manages file edit history in memory (per-process, LIFO per file)."""

    def __init__(
        self,
        max_history_per_file: int = 5,
        history_dir: Path | None = None,  # kept for signature compatibility, ignored
    ):
        """Initialize the history manager.

        Args:
            max_history_per_file: Maximum history entries retained per file.
                Older entries drop off FIFO once the cap is hit.
            history_dir: Ignored (upstream used it to seed a disk cache dir).
                Kept in the signature so upstream callers do not break.
        """
        self.max_history_per_file = max_history_per_file
        self._history: dict[str, Deque[str]] = defaultdict(
            lambda: deque(maxlen=self.max_history_per_file)
        )

    def add_history(self, file_path: Path, content: str) -> None:
        """Push a snapshot of the pre-edit content onto the file's history."""
        self._history[str(file_path)].append(content)

    def pop_last_history(self, file_path: Path) -> str | None:
        """Pop and return the most recent history entry, or None if empty."""
        stack = self._history.get(str(file_path))
        if not stack:
            return None
        return stack.pop()

    def get_metadata(self, file_path: Path) -> dict:
        """Return metadata shape mirroring the upstream API (for tests)."""
        stack = self._history.get(str(file_path))
        entries = list(range(len(stack))) if stack else []
        counter = len(stack) if stack else 0
        return {'entries': entries, 'counter': counter}

    def clear_history(self, file_path: Path) -> None:
        """Drop all history for the given file."""
        self._history.pop(str(file_path), None)

    def get_all_history(self, file_path: Path) -> list[str]:
        """Return all history entries oldest-first (deque iteration order)."""
        stack = self._history.get(str(file_path))
        return list(stack) if stack else []
