"""
Import recording & replay — the runtime half of "definition is authorization".

The agent .py file's compile is the ONLY moment new modules may load: we wrap
`__import__` with a recorder and remember every name the file pulled in. The
model's exec sandbox then gets a REPLAY `__import__`: names the file already
imported resolve idempotently (from sys.modules — no new code runs); anything
else raises ImportError whose message teaches the rule.

Why replay instead of just blocking (SANDBOX_BUILTINS drops __import__
entirely): models reflexively write `import math` in exec code when the
instruction shows `import math` in the source. Blocking that produces a
baffling failure for something the file explicitly authorized. Replay keeps
the source honest — what you see importable is importable, nothing more.

This module is backstage: it never enters an agent's instruction.
"""

from __future__ import annotations

import builtins as _builtins
import sys
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

__all__ = [
    "recording_builtins",
    "replay_import",
]

_ImportFn = Callable[..., ModuleType]


def recording_builtins() -> tuple[dict[str, Any], set[str]]:
    """
    Build a full-builtins dict whose `__import__` records requested names.

    Use as compile-time builtins for the agent .py (via Compiler
    local_injections). Returns (builtins_dict, recorded_names) — the set is
    filled in-place as the compile executes.

    Recorded per import statement:
    - the requested dotted name and all its ancestor prefixes
      (`import a.b.c` authorizes `a`, `a.b`, `a.b.c`)
    - fromlist submodules that resolved to real modules
      (`from a import b` authorizes `a.b` when b is a submodule)
    """
    recorded: set[str] = set()
    real_import: _ImportFn = _builtins.__import__

    def _record(name: str) -> None:
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            recorded.add(".".join(parts[:i]))

    def recording_import(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        module = real_import(name, globals, locals, fromlist, level)
        if level == 0:
            _record(name)
        else:
            # Relative import: resolve against the returned module's package.
            # (FEATURE §10.10 #1 leaves relative-import support open; record
            # whatever actually loaded so replay stays consistent with compile.)
            _record(module.__name__)
        for sub in fromlist or ():
            candidate = f"{name}.{sub}"
            if candidate in sys.modules:
                _record(candidate)
        return module

    return {**_builtins.__dict__, "__import__": recording_import}, recorded


def replay_import(allowed: frozenset[str]) -> _ImportFn:
    """
    Build the exec-time `__import__`: idempotent replay of compile-time imports.

    Allowed names delegate to the real import machinery — safe because the
    compile already loaded them (sys.modules hit, no new top-level code).
    Anything else raises ImportError with the authorization rule spelled out.
    """
    real_import: _ImportFn = _builtins.__import__

    def _replay(
        name: str,
        globals: Mapping[str, Any] | None = None,
        locals: Mapping[str, Any] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> ModuleType:
        if level != 0:
            raise ImportError(
                "relative imports are not available in the agent sandbox; "
                "use the names your definition file already imported."
            )
        if name not in allowed:
            raise ImportError(
                f"module {name!r} is not in your capability surface. "
                f"Your imports are your authorization: only modules imported "
                f"by your definition file are available. "
                f"Authorized: {', '.join(sorted(allowed)) or '(none)'}."
            )
        for sub in fromlist or ():
            candidate = f"{name}.{sub}"
            # Submodule pulls must have been resolved at compile time too;
            # attribute (non-module) fromlist entries pass through untouched.
            if candidate in sys.modules and candidate not in allowed:
                raise ImportError(
                    f"submodule {candidate!r} is not in your capability surface "
                    f"(your definition file imported {name!r} but not {candidate!r})."
                )
        return real_import(name, globals, locals, fromlist, level)

    return _replay
