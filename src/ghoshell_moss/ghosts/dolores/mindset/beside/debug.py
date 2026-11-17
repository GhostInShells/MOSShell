"""ghost.beside.debug — the runtime self-test surface for emanation tasks.

A virtual child of beside, mounted only when its dependencies exist. It exists so the
mechanism can be exercised by the one who runs it, without a caller in the code: point it at
any commit of this trajectory, a prompt, and a model reference, and it returns the validated
answer.

The model reference has two forms:

- ``package.module:Attr`` — an importable class, resolved with the process's own permissions.
- ``path/to/file.py:Attr`` with ``source=True`` — the file is compiled in a sandbox whose only
  allowed imports are ``pydantic`` / ``enum`` / ``typing`` / ``datetime`` / ``uuid``. That is
  enough to define a test model and not enough to touch the machine.

The coordinate resolves through the trajectory, so any commit this ghost ever wrote is a valid
target — including commits whose session has long been archived (the read is a replay).
"""

from __future__ import annotations

import importlib
from pathlib import Path

from pydantic import BaseModel

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel
from ghoshell_moss.core.codex.sandbox import SANDBOX_BUILTINS, Sandbox
from ghoshell_moss.deepseek_harness.v0_2_0 import DshSessionAnchor, DshSessionRef
from ghoshell_moss.memento import Memento

from ghoshell_moss.ghosts.dolores.mindset.emanation_task import EmanationTasks

__all__ = ["build_debug_channel", "load_debug_model"]

#: Imports a source-defined test model may use. Everything else is refused by the sandbox's
#: ``__import__``, so a debug model cannot reach the filesystem, the network, or subprocesses.
_ALLOWED_IMPORTS = frozenset({"pydantic", "enum", "typing", "datetime", "uuid"})

#: The metadata key a commit uses to carry its session reference (``_ego_memento._REF_KEY``).
_SESSION_REF_KEY = "ref"


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if level != 0 or root not in _ALLOWED_IMPORTS:
        raise ImportError(
            f"emanation debug sandbox allows only {sorted(_ALLOWED_IMPORTS)}; refused {name!r}"
        )
    return __import__(name, globals, locals, fromlist, level)


def load_debug_model(ref: str, *, source: bool) -> tuple[type[BaseModel], Sandbox | None]:
    """Resolve ``module:Attr`` (or ``file.py:Attr`` under a sandbox) into a model class."""
    module_ref, _, attr = ref.partition(":")
    if not attr:
        raise ValueError("model must be `module:Attr` or `path.py:Attr`")

    box: Sandbox | None = None
    if source:
        path = Path(module_ref).expanduser()
        if not path.is_file():
            raise ValueError(f"no such file: {path}")
        builtins = dict(SANDBOX_BUILTINS)
        builtins["__import__"] = _guarded_import
        box = Sandbox("emanation_debug", builtins=builtins)
        result = box.exec(path.read_text(encoding="utf-8"))
        if result.exception:
            box.close()
            raise ValueError(f"{path} failed to compile: {result.exception}")
        try:
            cls = box.get(attr)
        except AttributeError:
            box.close()
            raise ValueError(f"{path} defines no `{attr}`") from None
    else:
        cls = getattr(importlib.import_module(module_ref), attr)

    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        if box is not None:
            box.close()
        raise TypeError(f"{ref} is not a pydantic BaseModel subclass")
    return cls, box


def _anchor_of(raw: dict) -> DshSessionAnchor:
    """Fold a commit's stored session ref into an anchor.

    Ego commits store turns only — ``start_seq``/``end_seq`` stay null — and that is the shape
    the note path reads every day. ``DshSessionRef.to_anchor`` refuses a turn-only ref on
    purpose (``seq=None`` in an anchor means "session start"), so the fold goes through
    ``DshSessionAnchor.from_ref``: its ``turn`` field is the fallback for a caller that only
    ever recorded turns.
    """
    return DshSessionAnchor.from_ref(DshSessionRef.model_validate(raw), at="end")


def build_debug_channel(
        tasks: EmanationTasks,
        memento: Memento,
        *,
        name: str = "debug",
) -> MutableChannel:
    """Build the debug child channel around the live task runner and trajectory."""
    chan = new_channel(
        name=name,
        description="Runtime self-test tools — exercise a mechanism without writing a caller.",
    )

    @chan.build.instruction
    async def _instruction() -> str:
        return (
            "## Beside / debug (self-test)\n"
            "These commands exist to prove a mechanism works at runtime, not to get work done.\n"
            "`emanation_task` asks one structured question to this trajectory at any historical\n"
            "commit and returns the validated answer. Nothing here writes to the record."
        )

    @chan.build.command(name="emanation_task")
    async def emanation_task(
            coord: str,
            prompt: str,
            model: str,
            source: bool = False,
            timeout: float = 60.0,
    ) -> str:
        """Run one emanation task against any commit of this trajectory.

        The commit must carry a session reference (every ego commit does). The answer is
        validated against the model and returned as JSON.

        :param coord: branch coordinate, e.g. `1-764`
        :param prompt: what to ask at that position
        :param model: `package.module:Attr`; with source=True, `path.py:Attr`
        :param source: compile the file in a pydantic-only sandbox instead of importing it
        :param timeout: per-attempt timeout in seconds
        """
        view = memento.resolve_commit(coord)
        if view is None:
            return f"error: no commit at `{coord}`"
        raw = view.ref.metadata.get(_SESSION_REF_KEY)
        if not isinstance(raw, dict):
            return f"error: commit `{coord}` carries no session reference in metadata"
        anchor = _anchor_of(raw)

        box: Sandbox | None = None
        try:
            cls, box = load_debug_model(model, source=source)
            value = await tasks.request(prompt, cls, anchor=anchor, timeout=timeout)
            return value.model_dump_json(indent=2)
        except Exception as error:  # noqa: BLE001 — a debug tool reports, it never breaks the channel
            return f"error: {type(error).__name__}: {error}"
        finally:
            if box is not None:
                box.close()

    return chan
