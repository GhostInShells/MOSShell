"""moss self CLI read-only tools — a safe subset of `moss` commands for agents.

Each tool is a @cli-wrapped ``python -m ghoshell_moss.cli --ai <subcommand>``
call. Only read-only commands are exposed — documentation, feature, and codex
reflection. Mutating commands (features create/set-status/init, skills recall,
…) are deliberately absent: the import surface is the authorization whitelist,
and this module is the whitelist of moss commands an agent may reach.

Synchronous callables (async @cli — the agent sandbox wraps them in
``asyncio.run``). cwd is lazily bound to the project root; output is capped
(last 12000 chars) like the moss_cli channel.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

from ghoshell_moss.decorators import cli

__all__ = [
    "moss_codex_get_interface",
    "moss_codex_get_source",
    "moss_codex_list",
    "moss_codex_where",
    "moss_codex_architecture",
    "moss_features_list",
    "moss_features_specification",
    "moss_docs_list",
    "moss_docs_read",
]

_PY = sys.executable
_TIMEOUT = 120.0
_RESULT_CHAR_CAP = 12_000


@functools.lru_cache(maxsize=1)
def _root() -> Path:
    from ghoshell_moss.core.blueprint.project import Project
    return Project.discover().root


def _cap_result(result: tuple[int, str, str]) -> tuple[int, str, str]:
    """出参加工: 截断 stdout, 形状不变 (三元组 → 三元组)."""
    code, stdout, stderr = result
    if len(stdout) > _RESULT_CHAR_CAP:
        dropped = len(stdout) - _RESULT_CHAR_CAP
        stdout = f"...[{dropped} chars truncated]\n" + stdout[-_RESULT_CHAR_CAP:]
    return (code, stdout, stderr)


# ── codex — module / architecture reflection ────────────────────────────────


@cli([_PY, "-m", "ghoshell_moss.cli", "--ai", "codex", "get-interface"], name="moss_codex_get_interface", timeout=_TIMEOUT, cwd=_root, output_processor=_cap_result)
async def moss_codex_get_interface(arguments: str = "") -> tuple[int, str, str]:
    """Reflect a module or attribute interface (moss codex get-interface).

    Pass 'module:attr' (or just 'module') in arguments to get the structured
    interface contract — signatures, fields, type annotations, Field descriptions.
    """
    ...


@cli([_PY, "-m", "ghoshell_moss.cli", "--ai", "codex", "get-source"], name="moss_codex_get_source", timeout=_TIMEOUT, cwd=_root, output_processor=_cap_result)
async def moss_codex_get_source(arguments: str = "") -> tuple[int, str, str]:
    """Read the full source of a module or attribute (moss codex get-source).

    Pass 'module:attr' (or just 'module') in arguments. Minimal, un-reflected
    source view.
    """
    ...


@cli([_PY, "-m", "ghoshell_moss.cli", "--ai", "codex", "list"], name="moss_codex_list", timeout=_TIMEOUT, cwd=_root, output_processor=_cap_result)
async def moss_codex_list(arguments: str = "") -> tuple[int, str, str]:
    """List modules in a package, or members of a module (moss codex list).

    Pass a package path (e.g. 'ghoshell_moss.tools') in arguments.
    """
    ...


@cli([_PY, "-m", "ghoshell_moss.cli", "--ai", "codex", "where"], name="moss_codex_where", timeout=_TIMEOUT, cwd=_root, output_processor=_cap_result)
async def moss_codex_where(arguments: str = "") -> tuple[int, str, str]:
    """Resolve a module/attribute to its canonical definition path (moss codex where).

    Pass 'module' or 'module:attr' in arguments.
    """
    ...


@cli([_PY, "-m", "ghoshell_moss.cli", "--ai", "codex", "architecture"], name="moss_codex_architecture", timeout=_TIMEOUT, cwd=_root, output_processor=_cap_result)
async def moss_codex_architecture(arguments: str = "") -> tuple[int, str, str]:
    """Show the curated module map — key packages and their roles (moss codex architecture).

    Use for navigation: find where an abstraction lives instead of grepping.
    """
    ...


# ── features — workstream tracking (read-only) ──────────────────────────────


@cli([_PY, "-m", "ghoshell_moss.cli", "--ai", "features", "list"], name="moss_features_list", timeout=_TIMEOUT, cwd=_root, output_processor=_cap_result)
async def moss_features_list(arguments: str = "") -> tuple[int, str, str]:
    """List active feature workstreams with status and priority (moss features list)."""
    ...


@cli([_PY, "-m", "ghoshell_moss.cli", "--ai", "features", "specification"], name="moss_features_specification", timeout=_TIMEOUT, cwd=_root, output_processor=_cap_result)
async def moss_features_specification(arguments: str = "") -> tuple[int, str, str]:
    """Show the features convention — FEATURE.md format and rules (moss features specification)."""
    ...


# ── docs — project reference documents (read-only) ──────────────────────────


@cli([_PY, "-m", "ghoshell_moss.cli", "--ai", "docs", "list"], name="moss_docs_list", timeout=_TIMEOUT, cwd=_root, output_processor=_cap_result)
async def moss_docs_list(arguments: str = "") -> tuple[int, str, str]:
    """List AI reference docs with titles and descriptions (moss docs list)."""
    ...


@cli([_PY, "-m", "ghoshell_moss.cli", "--ai", "docs", "read"], name="moss_docs_read", timeout=_TIMEOUT, cwd=_root, output_processor=_cap_result)
async def moss_docs_read(arguments: str = "") -> tuple[int, str, str]:
    """Read an AI reference document by path (moss docs read). Pass the doc path in arguments."""
    ...
