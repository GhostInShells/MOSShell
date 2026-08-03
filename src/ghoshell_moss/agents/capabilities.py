"""
Agent capabilities — importable stubs for agent .py files.

When an agent imports a name from this module, the factory detects it at
compile time and injects the real implementation (bound to cwd) into the
agent sandbox. The stub here is a documentation artefact — it never
executes in the sandbox; the factory overrides it with the real thing.

Usage in an agent .py:

    from ghoshell_moss.agents.capabilities import look_at

    # look_at is a stub here; at runtime the sandbox has the real one,
    # which you call from sandbox_exec.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "look_at",
    "codex_where",
    "codex_list",
    "codex_source",
    "CAPABILITY_FACTORIES",
]

# ── Capability stubs ─────────────────────────────────────────────────────────


def look_at(path: str) -> str:
    """Stub — replaced by the factory at sandbox creation time.

    Real behaviour: list a directory or read a file within the cwd.
    """
    raise NotImplementedError(
        "look_at is a capability stub — the factory injects the real "
        "implementation at sandbox creation time"
    )


def codex_where(module_name: str) -> str:
    """Stub. Real: resolve an import path to its filesystem location."""
    raise NotImplementedError(
        "codex_where is a capability stub — inject the real one via factory"
    )


def codex_list(package_name: str) -> str:
    """Stub. Real: list submodules of a package (importlib, no execution)."""
    raise NotImplementedError(
        "codex_list is a capability stub — inject the real one via factory"
    )


def codex_source(module_name: str) -> str:
    """Stub. Real: read the source code of a module."""
    raise NotImplementedError(
        "codex_source is a capability stub — inject the real one via factory"
    )


# ── Real implementations (factory side) ──────────────────────────────────────


def _real_look_at(cwd: Path):
    """Factory for the real look_at bound to a cwd."""

    def _look_at(path: str) -> str:
        target = Path(path)
        if not target.is_absolute():
            target = cwd / target
        target = target.resolve()
        try:
            target.relative_to(cwd)
        except ValueError:
            return f"Error: {path!r} is outside the working directory {cwd}"
        if not target.exists():
            return f"Error: {path} does not exist"
        if target.is_dir():
            entries = sorted(target.iterdir())
            if not entries:
                return f"Directory: {path}\n(empty)"
            lines: list[str] = []
            for e in entries:
                suffix = "/" if e.is_dir() else ""
                lines.append(f"  {e.name}{suffix}")
            return f"Directory: {path}\n" + "\n".join(lines)
        content = target.read_text(encoding="utf-8", errors="replace")
        max_len = 4000
        if len(content) > max_len:
            content = content[:max_len] + f"\n... (truncated, {len(content)} total chars)"
        return content

    return _look_at


# ── Real implementations: codex reflection ───────────────────────────────────


def _real_codex_where(_cwd: Path):
    """Resolve an import path to its filesystem location."""

    def _where(module_name: str) -> str:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return f"Error: module {module_name!r} not found"
        if spec.origin is None:
            return f"built-in: {module_name}"
        return spec.origin

    return _where


def _real_codex_list(_cwd: Path):
    """List submodules of a package (importlib, no module execution)."""

    def _list(package_name: str) -> str:
        spec = importlib.util.find_spec(package_name)
        if spec is None:
            return f"Error: package {package_name!r} not found"
        if spec.origin is None:
            return f"Error: {package_name!r} is a built-in or namespace package with no listing"
        loader = spec.loader
        if loader is None or not hasattr(loader, "get_resource_reader"):
            return f"Error: {package_name!r} does not support directory listing"
        path = Path(spec.origin).parent
        if not path.is_dir():
            return f"Error: {package_name!r} origin {path} is not a directory"
        submods: list[str] = []
        for info in pkgutil.iter_modules([str(path)]):
            submods.append(info.name)
        if not submods:
            return f"Package: {package_name}\n(no submodules)"
        return f"Package: {package_name}\n  " + "\n  ".join(sorted(submods))

    return _list


def _real_codex_source(_cwd: Path):
    """Read the source code of a module (inspect, no execution)."""

    def _source(module_name: str) -> str:
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            return f"Error: cannot import {module_name!r}: {e}"
        try:
            return inspect.getsource(module)
        except Exception as e:
            return f"Error: cannot read source of {module_name!r}: {e}"

    return _source


# ── Registry ─────────────────────────────────────────────────────────────────
# Key = capability name (matching the stub function name).
# Value = factory that takes cwd: Path and returns the real implementation.
# The factory module imports this to auto-detect and inject capabilities.


CAPABILITY_FACTORIES: dict[str, Callable[[Path], Any]] = {
    "look_at": _real_look_at,
    "codex_where": _real_codex_where,
    "codex_list": _real_codex_list,
    "codex_source": _real_codex_source,
}
