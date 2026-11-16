"""EgoLifecycle — the strongly typed listener registry for the ego's lifecycle.

This is the **registration surface** modules see. It deliberately exposes no ``fire``:
the ego is the only lifecycle driver, and a second trigger source would be a lie about
causality. The concrete bus (``EgoLifecycleBus``) owns firing and the task set; modules
only register and discard listeners.

Blocking is a property of the **phase**, not of a listener. Each ``on_*`` docstring
states whether the ego waits for that phase. A listener that needs ordering or a
lifecycle side effect is not a listener — that is a pipeline stage.

Event types are plain frozen dataclasses: strong types are the contract, and the
callback signature names the event it receives.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeAlias

__all__ = [
    "CommitNoteEvent",
    "Discard",
    "EgoLifecycle",
    "ListenerResult",
    "StartupEvent",
    "StepEndEvent",
    "ThinkingExitEvent",
    "ThinkingStartEvent",
]

Discard: TypeAlias = Callable[[], None]
"""Remove a previously registered listener. Idempotent."""

ListenerResult: TypeAlias = Awaitable[None] | None
"""A listener may be async or sync; a sync listener returns None and runs in a thread."""


@dataclass(frozen=True, slots=True)
class StartupEvent:
    """The ghost is starting: modules may initialize before the first turn."""

    mode: str = ""


@dataclass(frozen=True, slots=True)
class ThinkingStartEvent:
    """A thinking frame is about to enter its model call."""

    seq: int
    turn: int = 0
    effort: str = ""


@dataclass(frozen=True, slots=True)
class ThinkingExitEvent:
    """A thinking frame has left its model call (or failed to)."""

    seq: int
    ok: bool = True


@dataclass(frozen=True, slots=True)
class CommitNoteEvent:
    """A commit's note is ready — the span is now readable as a note."""

    commit_id: str
    note: str
    branch: str = ""


@dataclass(frozen=True, slots=True)
class StepEndEvent:
    """One dsh step has ended: its tool result is in the log, the next step has not begun.

    ``seq`` is the log position of the ``step/end`` event — the anchor a per-step watcher
    reads the session at. ``step`` is the step's own number inside ``turn``; a ticket issued
    at step N must not start before step N ends, which is exactly the edge this event marks.
    """

    turn: int = 0
    step: int = 0
    seq: int = 0


class EgoLifecycle(ABC):
    """Register lifecycle listeners. The ego fires; this surface only observes.

    Registration returns a :data:`Discard`; call it once to unregister. A listener
    must be order-independent against other listeners of the same phase: the bus may
    run them in parallel, and no registration order is promised.
    """

    @abstractmethod
    def on_startup(self, callback: Callable[[StartupEvent], ListenerResult]) -> Discard:
        """Register a startup listener.

        **Blocking phase**: the ghost waits for every startup listener before the
        first turn. Keep listeners fast; do not wait on external I/O here.
        """

    @abstractmethod
    def on_thinking_start(self, callback: Callable[[ThinkingStartEvent], ListenerResult]) -> Discard:
        """Register a pre-thinking listener.

        **Blocking phase**: fired before the thinking frame's enter RPC; the ghost
        waits for every listener. Keep listeners fast.
        """

    @abstractmethod
    def on_thinking_exit(self, callback: Callable[[ThinkingExitEvent], ListenerResult]) -> Discard:
        """Register a post-thinking listener.

        **Non-blocking phase**: fired after the exit RPC; listeners run as tasks.
        Exceptions are logged and never propagated to the ego.
        """

    @abstractmethod
    def on_commit_note(self, callback: Callable[[CommitNoteEvent], ListenerResult]) -> Discard:
        """Register a commit-note listener.

        **Non-blocking phase**: fired when a commit's note becomes ready; listeners
        run as tasks. Exceptions are logged and never propagated to the ego.
        """

    @abstractmethod
    def on_step_end(self, callback: Callable[[StepEndEvent], ListenerResult]) -> Discard:
        """Register a step-end listener.

        **Non-blocking phase**: the dsh session event stream is dispatched serially and
        awaits each handler, so a slow listener would stall every later event. The bus
        schedules listeners as tasks; they must not block the dispatch (spawn, do not
        await model calls). Exceptions are logged and never propagated to the ego.
        """
