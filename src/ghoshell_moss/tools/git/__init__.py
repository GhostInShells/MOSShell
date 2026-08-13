"""git read-only tools — @cli-wrapped safe git subset for agent use.

Safe subset: only non-mutating git commands are exposed. Each tool is a
@cli-wrapped callable: prefix is fixed (git + subcommand), so write commands
(add/commit/push/reset/...) cannot be reached structurally — the whitelist is
the import surface itself. Execution runs via SubprocessFacade in the repo
root (cwd bound to project root, resolved lazily per call).

Declaration and wrapping are split: each function below is the pure
declaration (signature + docstring, body never invoked); the @cli wrap is a
separate assignment on the same name.
"""

from __future__ import annotations

import functools
from pathlib import Path

from ghoshell_moss.decorators import cli

__all__ = ["git_status", "git_diff"]

_STATUS_TIMEOUT = 30.0
_DIFF_TIMEOUT = 60.0
_DIFF_CHAR_CAP = 20_000


@functools.lru_cache(maxsize=1)
def _repo_root() -> Path:
    """Project root (= git repo root), resolved lazily and cached per process."""
    from ghoshell_moss.core.blueprint.project import Project
    return Project.discover().root


def _strip_git_prefix(argv: list[str]) -> list[str]:
    """Strip reflexive 'git' / subcommand prefixes from arguments (model reflex)."""
    for word in ("git", "status", "diff"):
        if argv and argv[0] == word:
            argv = argv[1:]
    return argv


def _cap_diff(result: tuple[int, str, str]) -> tuple[int, str, str]:
    code, stdout, stderr = result
    if len(stdout) > _DIFF_CHAR_CAP:
        dropped = len(stdout) - _DIFF_CHAR_CAP
        stdout = f"...[{dropped} chars truncated]\n" + stdout[-_DIFF_CHAR_CAP:]
    return (code, stdout, stderr)


async def git_status(arguments: str = "") -> tuple[int, str, str]:
    """Show the working tree status as short per-file output (git status --short).

    Read-only, bound to the repo root. Use to see which files changed / are
    staged. Pass extra flags or a path via arguments if needed.
    """
    ...


git_status = cli(
    ["git", "status", "--short"],
    name="git_status",
    cwd=_repo_root,
    timeout=_STATUS_TIMEOUT,
    input_filter=_strip_git_prefix,
)(git_status)


async def git_diff(arguments: str = "") -> tuple[int, str, str]:
    """Show unstaged working-tree changes (git diff).

    Read-only, bound to the repo root. Pass '--stat' for a file-by-file
    summary, or '-- <path>' to narrow to one file.
    """
    ...


git_diff = cli(
    ["git", "diff"],
    name="git_diff",
    cwd=_repo_root,
    timeout=_DIFF_TIMEOUT,
    input_filter=_strip_git_prefix,
    output_processor=_cap_diff,
)(git_diff)
