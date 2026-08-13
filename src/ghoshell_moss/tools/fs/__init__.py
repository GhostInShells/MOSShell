"""fs read-only tools — read / list / glob within the project root.

Read-only filesystem surface for agents: read a file, list a directory, glob
paths. All paths resolve against the project root (= repo root) and are
rejected when they escape it (self-owned root boundary — file editor only
hints workspace_root, the boundary is ours). File operations reuse the
FileEditor contract (view / file_list / glob) — no parallel read path.

Synchronous callables: the agent sandbox is a sync exec context. Async
wrapping is the consumer's concern, not this layer's.
"""

from __future__ import annotations

import functools
from pathlib import Path

__all__ = ["read_file", "list_files", "glob_files"]


@functools.lru_cache(maxsize=1)
def _root() -> Path:
    from ghoshell_moss.core.blueprint.project import Project
    return Project.discover().root


def _resolve(path: str, root: Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError:
        raise ValueError(f"{path!r} is outside the project root {root}")
    return target


def _editor(root: Path):
    from ghoshell_moss.core.file_editor import DefaultFileEditor
    return DefaultFileEditor(workspace_root=root)


def read_file(path: str, view_range: list[int] | None = None) -> str:
    """Read a file within the project root, optionally a line range [start, end].

    Returns numbered lines (same format as `moss file-editor view`).
    """
    root = _root()
    target = _resolve(path, root)
    result = _editor(root).view(target, view_range=view_range)
    return result.output or "(empty file)"


def list_files(path: str = ".") -> str:
    """List a directory within the project root.

    One entry per line: name, size (human-readable), kind (file/dir/symlink).
    """
    root = _root()
    target = _resolve(path, root)
    result = _editor(root).file_list(target)
    return result.output


def glob_files(pattern: str) -> str:
    """Glob paths within the project root matching *pattern*.

    Pattern is relative to the project root (e.g. 'src/**/*.py'). Returns
    matched relative paths, one per line.
    """
    root = _root()
    matches = sorted(p.relative_to(root) for p in root.glob(pattern))
    if not matches:
        return f"Glob: {pattern}\n(no matches)"
    return "Glob: " + pattern + "\n  " + "\n  ".join(str(m) for m in matches)
