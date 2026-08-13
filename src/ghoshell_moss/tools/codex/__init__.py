"""codex read-only tools — Python module reflection for agents.

Resolve an import path to its filesystem location, list a package's
submodules, read a module's source. Read-only introspection via importlib /
inspect / pkgutil; only ``codex_source`` triggers a module import (to obtain
its source object) — the other two are import-free.

Synchronous callables: the agent sandbox is a sync exec context.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
from pathlib import Path

__all__ = ["codex_where", "codex_list", "codex_source"]


def codex_where(module_name: str) -> str:
    """Resolve a Python import path to its filesystem location."""
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return f"Error: module {module_name!r} not found"
    if spec.origin is None:
        return f"built-in: {module_name}"
    return spec.origin


def codex_list(package_name: str) -> str:
    """List submodules of a package (importlib, no module execution)."""
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
    submods = [info.name for info in pkgutil.iter_modules([str(path)])]
    if not submods:
        return f"Package: {package_name}\n(no submodules)"
    return f"Package: {package_name}\n  " + "\n  ".join(sorted(submods))


def codex_source(module_name: str) -> str:
    """Read the source code of a module (inspect; imports the module)."""
    try:
        module = importlib.import_module(module_name)
    except Exception as e:
        return f"Error: cannot import {module_name!r}: {e}"
    try:
        return inspect.getsource(module)
    except Exception as e:
        return f"Error: cannot read source of {module_name!r}: {e}"
