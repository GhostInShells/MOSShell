"""The calendar channel — the ghost's half of the body.

Two things meet here and are deliberately kept apart:

* **What the ghost can do** — read and write the same events the human sees on the page.
* **What the ghost notices** — the current interval, carried on ``named_notices``. The
  delta semantics are what make this cheap: a fragment re-emits only when the interval it
  describes actually changes, so the ghost always knows which window it stands in without
  paying for it on every frame. Hot ``context_messages`` are deliberately not used — a
  calendar's "now" is a slow variable and has no business in the per-frame tier.

Reminders never travel this channel. They are pushed as signals by the reminder loop, so a
due event reaches the ghost even while it is busy thinking about something else.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from ghoshell_moss.core.blueprint.channel_builder import new_channel

from .reminders import ReminderLoop
from .store import LEVEL_NAMES, MAX_LEVEL, CalendarStore, level_name
from .surface import CalendarSurface

_WHEN_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d", "%H:%M")


def parse_when(text: str, *, now: float) -> float:
    """Parse the loose time strings a model naturally writes into an epoch timestamp.

    Accepts ``YYYY-MM-DD HH:MM[:SS]``, a bare ``YYYY-MM-DD`` (midnight), and a bare
    ``HH:MM`` meaning today. Local time throughout — a personal calendar that shifts
    under a timezone table is a bug, not a feature.
    """
    text = text.strip()
    for fmt in _WHEN_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if fmt == "%H:%M":
            today = datetime.fromtimestamp(now)
            parsed = today.replace(hour=parsed.hour, minute=parsed.minute, second=0, microsecond=0)
        return parsed.timestamp()
    raise ValueError(
        f"cannot read '{text}' as a time — use 'YYYY-MM-DD HH:MM', 'YYYY-MM-DD' or 'HH:MM'"
    )


def parse_remind(text: str) -> float | None:
    """Parse a lead time like ``15m`` / ``2h`` / ``1d`` / ``0`` into seconds.

    Empty input means *no reminder*, which is a different thing from ``0`` (fire exactly at
    the start). Keeping the two apart lets an event be a plain note on the page.
    """
    text = text.strip().lower()
    if not text or text in ("none", "off"):
        return None
    units = {"s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}
    if text[-1] in units:
        return float(text[:-1]) * units[text[-1]]
    return float(text)


def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M")


def _enumerate_levels() -> str:
    return ", ".join(f"{i}={name}" for i, name in enumerate(LEVEL_NAMES))


async def _publish(store: CalendarStore, surface: CalendarSurface) -> None:
    """Push the changed events to every open page.

    The channel is a second writer on the page's own data, so a change made by the ghost
    has to travel back or the human would be looking at a stale grid.
    """
    horizon = store.now()
    await surface.broadcast(
        {
            "type": "state",
            "events": store.between(horizon - 60 * 86400, horizon + 365 * 86400),
            "edits": store.recent_edits(10),
        }
    )


def new_calendar_channel(store: CalendarStore, surface: CalendarSurface, loop: ReminderLoop):
    chan = new_channel(
        name="calendar",
        description=(
            "a shared calendar — the human sees and edits the same events on a live web "
            "page. add / update / remove / done / agenda write and read that shared store. "
            "Reminders arrive on their own as signals: a daily and hourly interval notice, "
            "an at-the-time reminder weighted by the event's level, and an overdue "
            "escalation for the loud ones."
        ),
    )

    @chan.build.instruction
    def instruction() -> str:
        return (
            "You share this calendar with a human who edits it on a live web page (its URL "
            "is in the `url` notice). You and the human write the same events — a change "
            "either of you makes is visible to both.\n"
            "Time arguments read 'YYYY-MM-DD HH:MM', 'YYYY-MM-DD' (midnight) or 'HH:MM' "
            "(today). `level` has "
            f"{len(LEVEL_NAMES)} steps: {_enumerate_levels()} — it decides how hard a "
            "reminder about this event pushes. `remind` is a lead time like '15m' / '1h' / "
            "'0' (at the start); leave it empty and the event is a plain entry that will "
            "not interrupt you.\n"
            "You do not have to poll for reminders. A due event reaches you as a signal on "
            "its own; the interval notice tells you which day and hour you are standing in."
        )

    @chan.build.named_notices
    def named_notices() -> dict[str, str | None]:
        now = store.now()
        dt = datetime.fromtimestamp(now)
        day_start = dt.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()

        today = store.between(day_start, day_start + 86400, include_done=False)
        if today:
            upcoming = [e for e in today if e["start_ts"] >= now]
            if upcoming:
                head = f"next {_fmt(upcoming[0]['start_ts'])} {upcoming[0]['title']}"
            else:
                head = "all past"
            today_text = f"{dt.strftime('%Y-%m-%d %a')} · {len(today)} event(s) · {head}"
        else:
            today_text = f"{dt.strftime('%Y-%m-%d %a')} · clear"

        hour_start = dt.replace(minute=0, second=0, microsecond=0).timestamp()
        this_hour = store.between(hour_start, hour_start + 3600, include_done=False)
        # None = the fragment is gone. An empty hour has no interval to describe, and
        # announcing it would spend tokens telling the ghost nothing.
        hour_text = None
        if this_hour:
            hour_text = f"{dt.strftime('%H:00')} · " + " / ".join(
                f"{_fmt(e['start_ts'])} {e['title']}"
                + ("" if e["level"] == 1 else f" [{level_name(e['level'])}]")
                for e in this_hour
            )

        return {"url": surface.url, "db": str(store.path), "today": today_text, "hour": hour_text}

    @chan.build.notice
    def notice() -> str:
        pending = store.pending_reminders(store.now())
        edits = store.recent_edits(4)
        parts = [f"{pending} reminder(s) armed"]
        if edits:
            parts.append("recent changes: " + "; ".join(edits))
        return " | ".join(parts)

    @chan.build.command(always_observe=False)
    async def add(
        title: str,
        start: str,
        end: str = "",
        remind: str = "",
        level: int = 1,
        notes: str = "",
        all_day: bool = False,
    ) -> str:
        """Put an event on the shared calendar.

        :param title: what the event is.
        :param start: when it starts — 'YYYY-MM-DD HH:MM', 'YYYY-MM-DD' or 'HH:MM' (today).
        :param end: when it ends, same formats. Empty leaves the end open.
        :param remind: lead time before the start, like '15m' / '1h' / '0' (exactly at the
            start). Empty means no reminder — the event shows on the page but will not
            interrupt you.
        :param level: how loud a reminder about this event may get — 0=low, 1=normal,
            2=high, 3=critical. High and critical escalate to a next-turn notify if the
            event ends up overdue.
        :param notes: free-form detail.
        :param all_day: an all-day entry, anchored to midnight of the start date.
        """
        start_ts = parse_when(start, now=store.now())
        if all_day:
            start_ts = datetime.fromtimestamp(start_ts).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).timestamp()
        event = store.add(
            title=title,
            start_ts=start_ts,
            end_ts=parse_when(end, now=store.now()) if end.strip() else None,
            all_day=all_day,
            notes=notes,
            level=max(0, min(MAX_LEVEL, int(level))),
            remind_before=parse_remind(remind),
        )
        store.log_edit("ghost", f"added {title}")
        await _publish(store, surface)
        return (
            f"#{event['id']} {title} @ {_fmt(start_ts)} "
            f"({level_name(event['level'])}, remind={remind or 'none'})"
        )

    @chan.build.command(always_observe=False)
    async def update(
        event_id: int,
        title: str = "",
        start: str = "",
        end: str = "",
        remind: str = "",
        level: int = -1,
        notes: str = "",
    ) -> str:
        """Change an existing event. Every argument is optional; empty means "leave it".

        Moving the start or the reminder lead time re-arms the reminder, so an event that
        already fired can fire again at its new time.

        :param event_id: the id returned when the event was added.
        :param title: new title.
        :param start: new start time.
        :param end: new end time.
        :param remind: new lead time, or 'none' to disarm the reminder.
        :param level: new level 0..3. Pass -1 (the default) to leave it unchanged.
        :param notes: new notes.
        """
        patch: dict[str, Any] = {}
        if title.strip():
            patch["title"] = title
        if start.strip():
            patch["start_ts"] = parse_when(start, now=store.now())
        if end.strip():
            patch["end_ts"] = parse_when(end, now=store.now())
        if remind.strip():
            patch["remind_before"] = parse_remind(remind)
        if level >= 0:
            patch["level"] = max(0, min(MAX_LEVEL, int(level)))
        if notes.strip():
            patch["notes"] = notes
        event = store.update(event_id, **patch)
        if event is None:
            return f"no event #{event_id}"
        store.log_edit("ghost", f"updated {event['title']}")
        await _publish(store, surface)
        return f"#{event['id']} {event['title']} @ {_fmt(event['start_ts'])}"

    @chan.build.command(always_observe=False)
    async def remove(event_id: int) -> str:
        """Delete an event from the shared calendar.

        :param event_id: the id returned when the event was added.
        """
        event = store.get(event_id)
        if event is None or not store.remove(event_id):
            return f"no event #{event_id}"
        store.log_edit("ghost", f"removed {event['title']}")
        await _publish(store, surface)
        return f"removed #{event_id} {event['title']}"

    @chan.build.command(always_observe=False)
    async def done(event_id: int, done: bool = True) -> str:
        """Mark an event handled, or put it back. A handled event stops reminding.

        :param event_id: the id returned when the event was added.
        :param done: True marks it handled, False reopens it.
        """
        event = store.set_done(event_id, done)
        if event is None:
            return f"no event #{event_id}"
        store.log_edit("ghost", f"marked {event['title']} {'done' if done else 'open'}")
        await _publish(store, surface)
        return f"#{event_id} {event['title']} is now {'done' if done else 'open'}"

    @chan.build.command(always_observe=True)
    def agenda(days: int = 7, limit: int = 20) -> str:
        """List what is coming up on the shared calendar.

        :param days: how far ahead to look.
        :param limit: at most this many entries.
        """
        now = store.now()
        horizon = now + days * 86400
        events = [e for e in store.upcoming(now, limit) if e["start_ts"] <= horizon]
        if not events:
            return f"nothing scheduled in the next {days} day(s)"
        body = "\n".join(
            f"#{e['id']} {_fmt(e['start_ts'])} {e['title']} ({level_name(e['level'])})"
            + (f" — {e['notes']}" if e["notes"] else "")
            for e in events
        )
        return f"{body}\ndatabase: {store.path}"

    @chan.build.startup
    async def _startup() -> None:
        await surface.start()
        loop.start()

    @chan.build.close
    async def _close() -> None:
        await loop.stop()
        await surface.stop()
        store.close()

    return chan
