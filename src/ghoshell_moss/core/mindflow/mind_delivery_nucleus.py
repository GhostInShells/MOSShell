"""MindDeliveryNucleus — a buffer for the ghost's own asynchronous mind deliveries.

A *mind delivery* is a result produced beside the main path: a memento chat answer, a
beside-think conclusion, a future concurrent-mind result. It must not wake the main path,
must not preempt it, and must not be lost when an attention ends.

So this nucleus buffers impulses and delivers **all buffered messages together** on the
next turn:

- every impulse is ``role='mind'`` and ``mode=next``;
- ``peek`` returns the highest-priority buffered impulse **without consuming it**;
- ``attended`` merges every buffered impulse into one carrier and clears the buffer;
- ``suppress`` keeps the buffer for the next re-rank — losing one round never loses content;
- ``clear`` is explicit (mindflow clear / shutdown), never per attention.

The signal name is ``mind/delivery``; producers set ``kind`` (``memento.chat``,
``beside.think`` ...), the messages, and an optional priority.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from ghoshell_container import IoCContainer
from pydantic import Field
from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.blueprint.mindflow import (
    ChallengeMode,
    Impulse,
    Nucleus,
    NucleusMeta,
    Priority,
    Signal,
    SignalMeta,
    SignalName,
)
from ghoshell_moss.message import ContextType, unique_id

__all__ = [
    "MindDeliveryNucleus",
    "MindDeliveryNucleusMeta",
    "MindDeliverySignalMeta",
    "new_mind_delivery_signal",
]

CAPABILITY = "mind/delivery"
"""Signal name — one delivery protocol for every beside-path result."""


class MindDeliverySignalMeta(SignalMeta):
    """A result from a beside path — buffered and delivered on the next turn.

    ``kind`` names the producer (``memento.chat`` / ``beside.think`` ...). The nucleus
    forces ``role='mind'`` and ``mode='next'``; the producer only supplies the content,
    an optional priority, and a description.
    """

    kind: str = Field(
        default="",
        description="what produced this delivery: memento.chat / beside.think / ...",
    )

    @classmethod
    def signal_name(cls) -> SignalName:
        return CAPABILITY

    @classmethod
    def priority(cls) -> Priority:
        return Priority.NOTICE


class MindDeliveryNucleus(Nucleus):
    """Buffer the ghost's own mind deliveries; send them together on the next turn."""

    NAME = "mind_delivery_nucleus"

    def __init__(
            self,
            *,
            name: str = NAME,
            logger: LoggerItf | None = None,
            suppress_seconds: float = 0.5,
    ) -> None:
        self._name = name
        self._logger = logger or get_moss_logger()
        self._suppress_seconds = suppress_seconds
        self._suppress_until = 0.0
        self._buffer: list[Impulse] = []
        self._fire_impulse: Callable[[Impulse], None] | None = None
        self._running = False

    # ── Nucleus ABC ────────────────────────────────────────────────

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return "buffer mind deliveries and send them together on the next turn"

    def status(self) -> str:
        if not self._buffer:
            return ""
        top = self._best()
        detail = f", top: {top.description[:50]}" if top is not None and top.description else ""
        return f"pending: {len(self._buffer)}{detail}"

    def signals(self) -> list[SignalName]:
        return [CAPABILITY]

    def clear(self) -> None:
        self._buffer.clear()
        self._suppress_until = 0.0

    def add_signal(self, signal: Signal) -> None:
        if not self._running or signal.name != CAPABILITY:
            return
        if not signal.messages:
            # A delivery without content says nothing; the producer should send the
            # failure text as a message, not as an empty body.
            return
        impulse = Impulse(
            id=signal.id,
            source=self._name,
            priority=signal.priority,
            messages=list(signal.messages),
            description=signal.description,
            role="mind",
            mode=ChallengeMode.next.value,
            complete=True,
        )
        self._buffer.append(impulse)
        if self._fire_impulse is not None and time.monotonic() >= self._suppress_until:
            self._fire_impulse(impulse)

    def suppress(self, suppress_by: Impulse, suppressed: Impulse | None = None) -> None:
        # Keep the buffer: a lost arbitration is not a lost delivery. Only stop firing
        # for a moment; the next re-rank can still peek the same buffer.
        self._suppress_until = time.monotonic() + self._suppress_seconds

    def attended(self, impulse: Impulse) -> Impulse | None:
        if not self._buffer:
            return impulse
        buffered = list(self._buffer)
        self._buffer.clear()
        messages = [message for item in buffered for message in item.messages]
        if not messages:
            return impulse
        priority = max((item.priority for item in buffered), default=impulse.priority)
        return impulse.model_copy(update={
            "id": unique_id(),
            "priority": priority,
            "messages": messages,
            "role": "mind",
            "mode": ChallengeMode.next.value,
            "complete": True,
        })

    def peek(self, no_stale: bool = True) -> Impulse | None:
        """Return the highest-priority buffered impulse, without consuming it."""
        if no_stale:
            self._buffer = [item for item in self._buffer if not item.is_stale()]
        return self._best()

    def ignored(self, impulse: Impulse) -> None:
        self._buffer = [item for item in self._buffer if item.id != impulse.id]

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            fire_impulse: Callable[[Impulse], None],
    ) -> None:
        self._fire_impulse = fire_impulse

    def is_running(self) -> bool:
        return self._running

    async def __aenter__(self) -> Self:
        self._running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._running = False
        self.clear()

    # ── internals ──────────────────────────────────────────────────

    def _best(self) -> Impulse | None:
        if not self._buffer:
            return None
        # max() keeps the first maximum; ties stay in arrival order.
        return max(self._buffer, key=lambda item: int(item.priority))


class MindDeliveryNucleusMeta(NucleusMeta):
    """Factory meta — lets the default mindflow and manifests discover the delivery nucleus."""

    def name(self) -> str:
        return MindDeliveryNucleus.NAME

    def description(self) -> str:
        return "buffer mind deliveries and send them together on the next turn"

    def signals(self) -> Iterable[type[SignalMeta]]:
        yield MindDeliverySignalMeta

    def factory(self, container: IoCContainer) -> Nucleus:
        return MindDeliveryNucleus(logger=container.get(LoggerItf))


def new_mind_delivery_signal(
        *messages: ContextType,
        kind: str = "",
        priority: Priority = Priority.NOTICE,
        description: str = "",
        hint: str = "",
) -> Signal:
    """Build one ``mind/delivery`` signal — a result from a beside path.

    The nucleus adds ``role='mind'`` and ``mode='next'``. ``kind`` names the producer;
    ``messages`` is the content (an answer, a failure note, a conclusion).
    """
    return MindDeliverySignalMeta(kind=kind).to_signal(
        *messages,
        description=description,
        priority=priority,
        hint=hint,
    )
