"""EgoLifecycleBus — the concrete ego lifecycle dispatcher.

The ego owns one bus. It enters the bus on startup and exits it on shutdown; the bus
holds the task set for non-blocking phases, isolates listener failures, and exposes
``wait_all_done`` for a graceful exit.

Firing is split by the phase's contract (see :mod:`ego_lifecycle`):

- ``fire_startup`` / ``fire_thinking_start`` are **async and blocking**: the ego
  awaits all listeners of the phase before continuing.
- ``fire_thinking_exit`` / ``fire_commit_note`` are **sync and non-blocking**: each
  listener runs as its own task; the bus tracks, isolates, and logs failures.

No listener can opt into blocking; blocking is a phase property. If a module needs a
lifecycle side effect before the next step, it is a pipeline stage, not a listener.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Callable
from typing import Any

from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger

from .ego_lifecycle import (
    CommitNoteEvent,
    Discard,
    EgoLifecycle,
    ListenerResult,
    StartupEvent,
    StepEndEvent,
    ThinkingExitEvent,
    ThinkingStartEvent,
)

__all__ = ["EgoLifecycleBus"]

Listener = Callable[[Any], ListenerResult]


class EgoLifecycleBus(EgoLifecycle):
    """Concrete lifecycle bus: registration, dispatch, task isolation, graceful drain."""

    def __init__(self, *, logger: LoggerItf | None = None) -> None:
        self._logger = logger or get_moss_logger()
        self._startup: list[Listener] = []
        self._thinking_start: list[Listener] = []
        self._thinking_exit: list[Listener] = []
        self._commit_note: list[Listener] = []
        self._step_end: list[Listener] = []
        self._tasks: set[asyncio.Task[None]] = set()
        self._closing = False

    # ── EgoLifecycle (registration) ────────────────────────────────

    def on_startup(self, callback: Callable[[StartupEvent], ListenerResult]) -> Discard:
        return self._register(self._startup, callback)

    def on_thinking_start(
            self, callback: Callable[[ThinkingStartEvent], ListenerResult],
    ) -> Discard:
        return self._register(self._thinking_start, callback)

    def on_thinking_exit(
            self, callback: Callable[[ThinkingExitEvent], ListenerResult],
    ) -> Discard:
        return self._register(self._thinking_exit, callback)

    def on_commit_note(
            self, callback: Callable[[CommitNoteEvent], ListenerResult],
    ) -> Discard:
        return self._register(self._commit_note, callback)

    def on_step_end(
            self, callback: Callable[[StepEndEvent], ListenerResult],
    ) -> Discard:
        return self._register(self._step_end, callback)

    # ── firing ──────────────────────────────────────────────────────

    async def fire_startup(self, event: StartupEvent | None = None) -> None:
        """Blocking phase: await every startup listener before returning."""
        await self._run("startup", self._startup, event or StartupEvent())

    async def fire_thinking_start(self, event: ThinkingStartEvent) -> None:
        """Blocking phase: await every pre-thinking listener before the enter RPC."""
        await self._run("thinking_start", self._thinking_start, event)

    def fire_thinking_exit(self, event: ThinkingExitEvent) -> None:
        """Non-blocking phase: schedule every post-thinking listener as a task."""
        self._spawn("thinking_exit", self._thinking_exit, event)

    def fire_commit_note(self, event: CommitNoteEvent) -> None:
        """Non-blocking phase: schedule every commit-note listener as a task."""
        self._spawn("commit_note", self._commit_note, event)

    def fire_step_end(self, event: StepEndEvent) -> None:
        """Non-blocking phase: schedule every step-end listener as a task.

        The session event dispatcher awaits handlers serially; this method only schedules,
        so a listener that starts a model call cannot stall the event stream.
        """
        self._spawn("step_end", self._step_end, event)

    # ── lifecycle ──────────────────────────────────────────────────

    async def __aenter__(self) -> Self:
        self._closing = False
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._closing = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    async def wait_all_done(self, timeout: float | None = None) -> None:
        """Wait for the non-blocking tasks to drain.

        Called before closing the matrix on a graceful exit. On timeout the remaining
        tasks are cancelled and logged; this method never raises their exceptions.
        """
        if not self._tasks:
            return
        pending = set(self._tasks)
        if timeout is None:
            await asyncio.gather(*pending, return_exceptions=True)
            return
        _, pending = await asyncio.wait(pending, timeout=timeout)
        if pending:
            self._logger.warning(
                "ego lifecycle: %d task(s) still running after %.1fs; cancelling",
                len(pending), timeout,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    # ── internals ──────────────────────────────────────────────────

    def _register(self, bucket: list[Any], callback: Listener) -> Discard:
        bucket.append(callback)
        removed = False

        def discard() -> None:
            nonlocal removed
            if removed:
                return
            removed = True
            with contextlib.suppress(ValueError):
                bucket.remove(callback)

        return discard

    @staticmethod
    async def _invoke(callback: Listener, event: object) -> None:
        # Sync/async is the listener's own logic — the same shape as ChannelBuilder.
        if inspect.iscoroutinefunction(callback):
            await callback(event)
        else:
            await asyncio.to_thread(callback, event)

    async def _run(self, name: str, listeners: list[Listener], event: object) -> None:
        """Blocking dispatch: run every listener in parallel, await all, isolate errors."""
        if not listeners:
            return
        callbacks = list(listeners)
        results = await asyncio.gather(
            *(self._invoke(callback, event) for callback in callbacks),
            return_exceptions=True,
        )
        for callback, result in zip(callbacks, results):
            if isinstance(result, BaseException):
                self._logger.error(
                    "ego lifecycle listener failed: %s (%s)",
                    name,
                    getattr(callback, "__qualname__", callback),
                    exc_info=result,
                )

    def _spawn(self, name: str, listeners: list[Listener], event: object) -> None:
        """Non-blocking dispatch: one task per listener, tracked for drain/cancel."""
        if self._closing:
            self._logger.debug("ego lifecycle %s fired after close; dropped", name)
            return
        for callback in list(listeners):
            task = asyncio.create_task(self._invoke(callback, event), name=f"ego-lifecycle:{name}")
            self._tasks.add(task)
            task.add_done_callback(self._on_task_done)

    def _on_task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._logger.error("ego lifecycle task failed: %s", task.get_name(), exc_info=exc)
