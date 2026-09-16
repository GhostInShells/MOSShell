"""
MOSS signal map — the curated entry point to every SignalMeta.

This module only re-exports; it does not define implementations. Developers,
ghosts, and channels learn the full signal surface from one entry; a new
SignalMeta is registered here and added to the cognitive map via architecture.py.

**Discipline**: a SignalMeta's implementation lives with its nucleus
(`core/mindflow/xxx_nucleus.py`). This file only imports + __all__; do not define
a class body here — that would tear the two abstractions apart.

Directory:
  InputSignalMeta      — user message (NOTICE, default: turn toward the user)
  NotifySignalMeta     — must-not-lose message (NOTICE, notify: buffer on loss)
  InterruptSignalMeta  — stop now (FATAL, interrupt: take attention then drop)
  CommandSignalMeta    — execute logos directly (NOTICE, command_only)
  AsideSignalMeta      — notice without interrupting (NOTICE, aside: buffer without attention)
  CellEventSignalMeta  — cell lifecycle event (BACKGROUND, background_notice)
"""
from ghoshell_moss.core.blueprint.mindflow import (
    InputSignalMeta,
)
from ghoshell_moss.core.mindflow import (
    NotifySignalMeta,
    InterruptSignalMeta,
    CommandSignalMeta,
    AsideSignalMeta,
)
from ghoshell_moss.core.mindflow.cell_event_nucleus import (
    CellEventSignalMeta,
    CellTransition,
)

__all__ = [
    'InputSignalMeta',
    'NotifySignalMeta',
    'InterruptSignalMeta',
    'CommandSignalMeta',
    'AsideSignalMeta',
    'CellEventSignalMeta',
    'CellTransition',
]
