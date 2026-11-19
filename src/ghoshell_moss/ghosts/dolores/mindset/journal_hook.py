"""JournalHook — wire the journal's calendar to the ego's lifecycle.

The journal is a ghost organ; this hook is its ego-side adapter. It opens the current day
when the registry starts, re-opens it when the ego crosses midnight, and records every
durable anchor on the anchor's own day. It writes no prose: the writer remains the ghost.
"""

from __future__ import annotations

from datetime import date

from ghoshell_container import INSTANCE, BootstrapProvider, IoCContainer, Provider
from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger

from .ego_lifecycle import CommitEvent, Discard, EgoLifecycle, ThinkingStartEvent
from .ego_lifecycle_hook import EgoLifecycleRegistry

__all__ = ["EgoLifecycleRegistryProvider", "JournalHook", "JournalHookProvider"]


class JournalHook:
    """Open the journal's day at start, keep it current, and record anchors by their own day."""

    def __init__(self, container: IoCContainer, *, logger: LoggerItf | None = None) -> None:
        self._container = container
        self._logger = logger or get_moss_logger()
        self._journal = None
        self._day: date | None = None
        self._discards: list[Discard] = []

    async def __aenter__(self) -> Self:
        from ..journal.contract import DoloresJournal

        self._journal = self._container.force_fetch(DoloresJournal)
        self._day = await self._journal.establish()
        return self

    async def listen(self, lifecycle: EgoLifecycle) -> None:
        if self._journal is None:
            return
        self._discards = [
            lifecycle.on_thinking_start(self._check_day),
            lifecycle.on_commit(self._record_commit),
        ]

    async def __aexit__(self, exc_type, exc, tb) -> None:
        for discard in self._discards:
            discard()
        self._discards.clear()
        self._journal = None

    async def _check_day(self, event: ThinkingStartEvent) -> None:
        if self._journal is None:
            return
        today = self._journal.today()
        if today == self._day:
            return
        self._day = await self._journal.establish(today)
        self._logger.info("journal hook: opened day %s", today.isoformat())

    async def _record_commit(self, event: CommitEvent) -> None:
        if self._journal is None:
            return
        await self._journal.record_commit(event.coord, event.at)


class EgoLifecycleRegistryProvider(Provider[EgoLifecycleRegistry]):
    """The hook catalog itself — a singleton, registered before the hook providers."""

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        return EgoLifecycleRegistry(logger=con.get(LoggerItf))


class JournalHookProvider(BootstrapProvider[JournalHook]):
    """Register the journal hook at container bootstrap; the runtime starts it later."""

    def singleton(self) -> bool:
        return False

    def contract(self) -> type[JournalHook]:
        return JournalHook

    def factory(self, con: IoCContainer) -> INSTANCE:
        return JournalHook(con, logger=con.get(LoggerItf))

    def bootstrap(self, con: IoCContainer) -> None:
        registry = con.force_fetch(EgoLifecycleRegistry)
        registry.register("journal", self.factory(con), critical=False)
