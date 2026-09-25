"""ReminderLoop — the calendar's three-tier push into the ghost's mindflow.

A calendar reminder is not one signal but three, each with its own claim on attention:

============================  ==========  =========================================
tier                          signal      claim
============================  ==========  =========================================
interval total (day / hour)   ``aside``   never preempts — just joins what is seen
at the time                   ``input``   competes for attention, weighted by level
overdue                       ``notify``  forces the next turn, but only when loud
============================  ==========  =========================================

The tiering is the point. A ticking clock that shouts every time is a clock the ghost
learns to ignore; letting the loud ones through and the quiet ones wait is what makes the
loud ones mean something.

``send_signal``, ``broadcast`` and the clock are all injected, so the loop can be driven
by a test without a Matrix, a socket or real time.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Awaitable, Callable

from ghoshell_moss.contracts.logger import get_moss_logger
from ghoshell_moss.core.blueprint.mindflow import Priority, Signal
from ghoshell_moss.message import Message
from ghoshell_moss.signals import AsideSignalMeta, InputSignalMeta, NotifySignalMeta

from .store import CalendarStore, level_name

LEVEL_PRIORITY = (Priority.INFO, Priority.NOTICE, Priority.WARNING, Priority.ERROR)
"""``level`` 0..3 → the priority a reminder about that event carries."""

OVERDUE_ESCALATE_ABOVE = Priority.NOTICE
"""Overdue events must be *louder* than this to earn a ``notify(next=True)``.

An event that has already started and is still unhandled is worth interrupting for only
if it mattered in the first place — otherwise every stale low-level entry would demand
the ghost's next turn and the escalation tier would mean nothing.
"""

POLL_CAP = 30.0
"""Ceiling on one sleep, so a write from another process (or a clock jump) still lands
within half a minute even though nothing local set ``store.changed``."""


def _hhmm(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%H:%M")


def human_edit_signal(text: str) -> Signal:
    """Announce a change the human just made on the page.

    Tiered as a low ``aside``: the ghost should know the schedule moved, but someone
    editing their own calendar is not a reason to take the ghost's attention away from
    whatever it is already holding.
    """
    return AsideSignalMeta().to_signal(
        Message.new(tag="calendar", name="edit").with_content(text),
        description=f"calendar: {text}",
        priority=Priority.INFO,
    )


def _fmt_event(ev: dict[str, Any]) -> str:
    when = _hhmm(ev["start_ts"])
    lvl = level_name(ev["level"])
    suffix = "" if lvl == "normal" else f" [{lvl}]"
    return f"{when} {ev['title']}{suffix}"


class ReminderLoop:
    """Watches the store and turns due events into signals."""

    def __init__(
        self,
        store: CalendarStore,
        *,
        send_signal: Callable[[Signal], None],
        broadcast: Callable[[dict[str, Any]], Awaitable[None]],
        now: Callable[[], float] = time.time,
        poll_cap: float = POLL_CAP,
        overdue_grace: float = 300.0,
    ) -> None:
        self._store = store
        self._send_signal = send_signal
        self._broadcast = broadcast
        self._now = now
        self._poll_cap = poll_cap
        self._overdue_grace = overdue_grace
        self._logger = get_moss_logger()
        # Interval buckets already announced. In-memory on purpose: a restart re-announces
        # the current interval once, which is the right failure mode for a body coming back up.
        self._day_key: str | None = None
        self._hour_key: str | None = None
        self._task: asyncio.Task | None = None

    # -- lifecycle --

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        task = self._task
        self._task = None
        if task is None or task.done():
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                # One bad event must not kill the ticker — a calendar that silently stops
                # reminding is worse than one that logs and carries on.
                self._logger.exception("calendar reminder tick failed")

    async def _tick(self) -> None:
        now = self._now()
        await self._announce_intervals(now)
        await self._fire_at_time(now)
        await self._fire_overdue(now)

        nxt = self._store.next_due_ts(now)
        delay = self._poll_cap if nxt is None else min(max(nxt - now, 0.0), self._poll_cap)
        try:
            await asyncio.wait_for(self._store.changed.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass
        finally:
            self._store.changed.clear()

    # -- tier 1: interval totals --

    async def _announce_intervals(self, now: float) -> None:
        dt = datetime.fromtimestamp(now)
        day_key = dt.strftime("%Y-%m-%d")
        hour_key = dt.strftime("%Y-%m-%d %H")

        if day_key != self._day_key:
            self._day_key = day_key
            day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
            today = self._store.between(day_start, day_start + 86400, include_done=False)
            if today:
                listing = " / ".join(_fmt_event(e) for e in today[:8])
                more = f" (+{len(today) - 8} more)" if len(today) > 8 else ""
                self._push(
                    AsideSignalMeta(),
                    "interval",
                    f"{day_key} has {len(today)} event(s): {listing}{more}",
                    f"calendar: {len(today)} event(s) today",
                    Priority.INFO,
                )

        if hour_key != self._hour_key:
            self._hour_key = hour_key
            hour_start = dt.replace(minute=0, second=0, microsecond=0).timestamp()
            this_hour = self._store.between(hour_start, hour_start + 3600, include_done=False)
            # An empty hour is not news. Announcing it would train the ghost to skim the
            # interval notices, which is exactly what this tier must not do.
            if this_hour:
                listing = " / ".join(_fmt_event(e) for e in this_hour)
                self._push(
                    AsideSignalMeta(),
                    "interval",
                    f"this hour: {listing}",
                    f"calendar: this hour holds {len(this_hour)} event(s)",
                    Priority.INFO,
                )

    # -- tier 2: at the time --

    async def _fire_at_time(self, now: float) -> None:
        for ev in self._store.due_at_time(now):
            self._store.mark_fired(ev["id"], now)
            self._store.log_edit("reminder", f"fired {_fmt_event(ev)}")
            await self._broadcast({"type": "reminder.fired", "tier": "at_time", "event": _wire(ev)})
            self._push(
                InputSignalMeta(),
                "reminder",
                f"{_fmt_event(ev)} starts now",
                f"calendar reminder: {_fmt_event(ev)}",
                LEVEL_PRIORITY[ev["level"]],
            )

    # -- tier 3: overdue --

    async def _fire_overdue(self, now: float) -> None:
        for ev in self._store.due_overdue(now, self._overdue_grace):
            # Marked either way: the escalation is decided once, and re-deciding it on every
            # tick would turn a quiet event into a recurring nag through the back door.
            self._store.mark_overdue(ev["id"], now)
            priority = LEVEL_PRIORITY[ev["level"]]
            if priority <= OVERDUE_ESCALATE_ABOVE:
                continue
            late = int((now - ev["start_ts"]) / 60)
            self._store.log_edit("overdue", f"escalated {_fmt_event(ev)}")
            await self._broadcast({"type": "reminder.fired", "tier": "overdue", "event": _wire(ev)})
            self._push(
                NotifySignalMeta(next=True),
                "overdue",
                f"{_fmt_event(ev)} started {late}m ago and is still unhandled",
                f"calendar overdue: {_fmt_event(ev)}",
                priority,
            )

    # -- signal plumbing --

    def _push(self, meta: Any, name: str, text: str, description: str, priority: Priority) -> None:
        self._send_signal(
            meta.to_signal(
                Message.new(tag="calendar", name=name).with_content(text),
                description=description,
                priority=priority,
            )
        )


def _wire(ev: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": ev["id"],
        "title": ev["title"],
        "start_ts": ev["start_ts"],
        "level": ev["level"],
    }
