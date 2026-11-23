"""Generic debug/reflection helpers for agentic classes.

This module stays light: the blueprint imports it lazily, and any codex-backed
rendering must import codex *inside* the function that uses it.
"""

from __future__ import annotations

import inspect
import sys
from typing import Any

__all__ = [
    "interface_item",
    "reflect_blocks",
    "reflect_interface",
    "reflect_source",
    "render_interface",
]


def reflect_blocks(ctx: Any) -> str:
    blocks: list[str] = []
    if ctx.reflect_source:
        blocks.append(reflect_source(ctx.reflect_source))
    if ctx.reflect_interface:
        blocks.append(reflect_interface(ctx.reflect_interface))
    return "\n".join(block for block in blocks if block)


def reflect_source(items: list[Any]) -> str:
    blocks: list[str] = []
    for item in items:
        name = getattr(item, "__qualname__", getattr(item, "__name__", repr(item)))
        try:
            source = inspect.getsource(item)
        except Exception:
            source = f"# source unavailable for {name}"
        blocks.append(
            f'<reflected-source name="{name}">\n```python\n{source}\n```\n</reflected-source>'
        )
    return "\n".join(blocks)


def reflect_interface(items: list[Any]) -> str:
    blocks: list[str] = []
    for item in items:
        name = getattr(item, "__qualname__", getattr(item, "__name__", repr(item)))
        blocks.append(
            f'<reflected-interface name="{name}">\n{interface_item(item)}\n</reflected-interface>'
        )
    return "\n".join(blocks)


def interface_item(item: Any) -> str:
    if inspect.ismodule(item):
        names = getattr(item, "__all__", None) or [
            name for name in dir(item) if not name.startswith("_")
        ]
        lines = [f"module {item.__name__}"]
        for name in names:
            value = getattr(item, name, None)
            if inspect.isclass(value) or inspect.isfunction(value):
                lines.append(interface_item(value))
        return "\n".join(lines)
    if inspect.isclass(item):
        lines = [f"class {item.__qualname__}: {inspect.getdoc(item) or ''}"]
        try:
            lines.append(f"  __init__{inspect.signature(item.__init__)}")
        except (TypeError, ValueError):
            pass
        for name, value in vars(item).items():
            if name.startswith("_"):
                continue
            if isinstance(value, (staticmethod, classmethod)):
                value = value.__func__
            if inspect.isfunction(value):
                try:
                    signature = inspect.signature(value)
                except (TypeError, ValueError):
                    signature = "(...)"
                lines.append(f"  {name}{signature}: {inspect.getdoc(value) or ''}")
            elif isinstance(value, property):
                lines.append(f"  {name}: property")
        return "\n".join(lines)
    if inspect.isfunction(item) or inspect.ismethod(item):
        try:
            signature = inspect.signature(item)
        except (TypeError, ValueError):
            signature = "(...)"
        return f"def {item.__qualname__}{signature}: {inspect.getdoc(item) or ''}"
    return repr(item)


def render_interface(ctx: Any, target: object) -> str:
    module = sys.modules.get(ctx.modulename) or inspect.getmodule(target)
    if module is None:
        raise RuntimeError(f"agentic class: cannot find the module of {type(target).__name__}")
    cls = type(target)
    lines = [f"# agentic class: {module.__name__}.{cls.__name__}"]
    doc = inspect.getdoc(cls) or ""
    if doc:
        lines.append(doc)
    context = getattr(target, "__context__", None)
    if context is not None:
        try:
            signature = inspect.signature(context)
        except (TypeError, ValueError):
            signature = "(...)"
        lines.append(f"context: __context__{signature}: {inspect.getdoc(context) or ''}")
    lines.append("ops:")
    for name, value in inspect.getmembers(cls):
        if not getattr(value, "__agent_method__", False):
            continue
        try:
            signature = inspect.signature(value)
        except (TypeError, ValueError):
            signature = "(...)"
        lines.append(f"- {name}{signature}: {inspect.getdoc(value) or ''}")
    lines.append(f"source: {ctx.file}")
    reflected = reflect_blocks(ctx)
    if reflected:
        lines.append(reflected)
    return "\n".join(lines)
