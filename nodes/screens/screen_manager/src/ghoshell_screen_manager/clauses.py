"""The stage's ear on the conversation: ``ClauseTopic`` in, danmaku out.

A clause is one finalized sentence, tagged with which side of the conversation
produced it (``role`` = ``user`` / ``ghost``). That is the whole input this feed
needs: the listening side publishes his sentences as he finishes them, the speaking
side publishes mine as TTS renders them, and both arrive here over the same bus.

Why a feed and not a command: the stage should mirror the conversation *while* it
happens, and a command can only be emitted by the model — which would put the
model in the loop of reporting itself. The bus already carries the fact; this just
listens to it.

Off by default. ``stage(talk=True)`` on the channel turns it on, because a screen
that narrates every sentence is worse than one that says nothing.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from ghoshell_moss.types.topics.audio import ClauseTopic
from ghoshell_moss.types.topics.ghost import GhostEventTopic

if TYPE_CHECKING:
    from ghoshell_moss.core.concepts.topic import TopicService

__all__ = ["TopicClauseFeed", "TopicEventFeed"]

ClauseHandler = Callable[[ClauseTopic], Awaitable[None]]
EventHandler = Callable[[GhostEventTopic], Awaitable[None]]


class TopicClauseFeed:
    """Subscribe ``ClauseTopic`` off the bus and hand every clause to a callback.

    Cross-process by construction: the listener and the speech side publish in
    their own processes, and the subscription rides the Matrix topic bridge here.
    """

    def __init__(self, topics: "TopicService") -> None:
        self._topics = topics
        self._on_clause: ClauseHandler | None = None
        self._task: asyncio.Task | None = None
        self._seen = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def stats(self) -> dict:
        return {"subscribed": self.running, "clauses": self._seen}

    async def start(self, on_clause: ClauseHandler) -> None:
        """Begin listening. The handler is supplied by whoever owns the surface."""
        self._on_clause = on_clause
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="screen-manager-clauses")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        subscriber = self._topics.subscribe_model(ClauseTopic, maxsize=64)
        async with subscriber:
            while True:
                try:
                    clause = await subscriber.poll_model(timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if clause is None:
                    continue
                text = (clause.text or "").strip()
                if not text:
                    continue
                self._seen += 1
                # A handler that throws must not kill the subscription: one bad
                # clause is not worth losing the conversation for.
                if self._on_clause is not None:
                    with contextlib.suppress(BaseException):
                        await self._on_clause(clause)


class TopicEventFeed:
    """Subscribe ``GhostEventTopic`` off the bus — the ghost's own state and actions.

    The other half of the mirror. A clause says what was *said*; this says what is
    *happening*: thinking started, a tool was called, the answer finished. Both are
    published by the reasoning kernel in its own process, so both ride the bus.

    Deliberately the same shape as the clause feed — subscribe, poll, hand each
    item to the owner — because two feeds that differ in plumbing are two things
    to learn. What differs is only what the consumer does with an event.
    """

    def __init__(self, topics: "TopicService") -> None:
        self._topics = topics
        self._on_event: EventHandler | None = None
        self._task: asyncio.Task | None = None
        self._seen = 0

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def stats(self) -> dict:
        return {"subscribed": self.running, "events": self._seen}

    async def start(self, on_event: EventHandler) -> None:
        self._on_event = on_event
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="screen-manager-events")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    async def _run(self) -> None:
        subscriber = self._topics.subscribe_model(GhostEventTopic, maxsize=64)
        async with subscriber:
            while True:
                try:
                    event = await subscriber.poll_model(timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if event is None:
                    continue
                self._seen += 1
                if self._on_event is not None:
                    with contextlib.suppress(BaseException):
                        await self._on_event(event)
