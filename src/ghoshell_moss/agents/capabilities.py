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

from pathlib import Path
from typing import Any, Callable

__all__ = [
    "look_at",
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


# ── Registry ─────────────────────────────────────────────────────────────────
# Key = capability name (matching the stub function name).
# Value = factory that takes cwd: Path and returns the real implementation.
# The factory module imports this to auto-detect and inject capabilities.


CAPABILITY_FACTORIES: dict[str, Callable[[Path], Any]] = {
    "look_at": _real_look_at,
}
