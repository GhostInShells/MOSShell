"""
Sandbox injections — capability protocols and unbound stubs.

Design principle: the AGENT.py file's imports from this module ARE its
authorization surface. SANDBOX_BUILTINS blocks __import__, so agent can only
touch what its own compile-time imports pulled in. This module is the
canonical entry point for MOSS-provided capabilities to agent sandboxes.

Runtime resolution: the runner replaces each unbound stub in the sandbox
namespace with a real implementation before the agent's exec loop starts.
Compile-time `file_editor = get_file_editor()` binds `file_editor` to an
_UnboundInjection proxy that raises helpfully if touched outside a sandbox.
Runner then does `sandbox.set("file_editor", RealFileEditor(...))` which
supersedes the proxy.

Future direction (post-v1): `get_*` functions are the uniform DI pattern for
MOSS capabilities. `get_contract(SomeContract)` discoverable via environment
manifests will let AGENT.py authors declare "I need this capability" without
knowing which concrete implementation the environment binds.
"""

from __future__ import annotations

from typing import Any, Iterator, Protocol, runtime_checkable

from ghoshell_moss.contracts.file_editor import FileEditor

__all__ = [
    "FileEditor",
    "AgentContext",
    "get_file_editor",
    "get_ctx",
]


@runtime_checkable
class AgentContext(Protocol):
    """
    Task-persistent working memory. Survives across invocations via memento.

    Values must be serializable through ghoshell_common.entity.EntityMeta:
    scalars (int / str / float / bool / None), lists and dicts of the above,
    pydantic BaseModel subclasses, and any object implementing the Entity
    protocol (__to_entity_meta__ / __from_entity_meta__). Non-serializable
    values raise TypeError at set time — model sees the error in the tool
    loop output and can retry with a serializable form.

    The interface intentionally mirrors dict. Model writes normal Python;
    persistence happens transparently at invoke boundaries (runner reads
    the snapshot and records a moment of type "agent.context/v1" if the
    ctx changed during the invoke).
    """

    def __setitem__(self, key: str, value: Any) -> None:
        """Set a persistent value. Raises TypeError if not serializable."""
        ...

    def __getitem__(self, key: str) -> Any:
        """Get a value. Raises KeyError if not present."""
        ...

    def __delitem__(self, key: str) -> None:
        """Remove a key. Raises KeyError if not present."""
        ...

    def __contains__(self, key: str) -> bool:
        """Check if a key is set."""
        ...

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        ...

    def __len__(self) -> int:
        """Number of keys."""
        ...

    def keys(self) -> list[str]:
        """List all keys."""
        ...

    def items(self) -> list[tuple[str, Any]]:
        """List (key, value) pairs — for `for k, v in ctx.items():` usage."""
        ...

    def values(self) -> list[Any]:
        """List values."""
        ...

    def get(self, key: str, default: Any = None) -> Any:
        """Safe get with default. Never raises KeyError."""
        ...


class _UnboundInjection:
    """
    Placeholder returned by get_* at AGENT.py compile time.

    Runner overrides the sandbox namespace binding before the exec loop
    starts. If this proxy is ever touched (i.e. runner failed to override),
    any attribute access raises a helpful error explaining what should have
    happened.
    """

    def __init__(self, name: str):
        self._name = name

    def __getattr__(self, attr: str) -> Any:
        if attr.startswith("__") and attr.endswith("__"):
            # Dunder probes (hasattr, inspect, reflection) must fail softly:
            # hasattr() only swallows AttributeError, so raising RuntimeError
            # here would blow up any reflection pass over a pre-swap namespace
            # (e.g. rendering the instruction of an agent file without running it).
            raise AttributeError(attr)
        raise RuntimeError(
            f"{self._name!r} is an unbound sandbox injection — the runner "
            f"should have replaced it before the exec loop started. "
            f"If you are seeing this, either the runner did not do the swap, "
            f"or this agent .py is being imported outside a MementoAgent context."
        )

    def __repr__(self) -> str:
        return f"<Unbound {self._name}>"


def get_file_editor() -> FileEditor:
    """
    Sandbox injection stub — returns a `FileEditor` bound to the AGENT.py's
    cwd (the directory containing the AGENT.py file, unless CLI --cwd
    overrides).

    At AGENT.py compile time this returns an _UnboundInjection proxy; the
    runner replaces the sandbox namespace binding with the real editor
    before the agent's exec loop starts.

    Usage in AGENT.py::

        from ghoshell_moss.agents.injections import FileEditor, get_file_editor
        file_editor: FileEditor = get_file_editor()
    """
    return _UnboundInjection("file_editor")  # type: ignore[return-value]


def get_ctx() -> AgentContext:
    """
    Sandbox injection stub — returns the `AgentContext` for the current
    line (branch). Loaded from the most recent "agent.context/v1" moment in
    the branch's history; empty if none exists.

    Changes made during an invoke are serialized back to memento at invoke
    boundary (as a new moment of the same type) — model does not need to
    explicitly save.

    Usage in AGENT.py::

        from ghoshell_moss.agents.injections import AgentContext, get_ctx
        ctx: AgentContext = get_ctx()

        # then in exec:
        ctx["current_section"] = "channel.md"
        if "iteration" in ctx:
            ctx["iteration"] += 1
    """
    return _UnboundInjection("ctx")  # type: ignore[return-value]
