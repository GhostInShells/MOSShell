"""CalendarStore — the shared truth of the calendar node.

The sqlite file is where the human (the FullCalendar page) and the ghost (the channel)
meet: both read and write the same table, so the file itself is the bus and no protocol
is needed between them. ``WAL`` + ``busy_timeout`` is what makes that safe when another
process reads it while this one writes — see the ``sqlite-channel`` workstream.

Reminder bookkeeping lives on the row it belongs to. One event carries one at-time
reminder (``remind_before`` seconds before ``start_ts``); ``fired_ts`` doubles as the
dedup guard and the ledger entry, so there is no second table to keep in sync.
"""

from __future__ import annotations

import asyncio
import sqlite3
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    start_ts REAL NOT NULL,
    end_ts REAL,
    all_day INTEGER NOT NULL DEFAULT 0,
    notes TEXT NOT NULL DEFAULT '',
    level INTEGER NOT NULL DEFAULT 1,
    remind_before REAL,
    fired_ts REAL,
    overdue_ts REAL,
    done_ts REAL,
    created_ts REAL NOT NULL,
    updated_ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS events_start_idx ON events(start_ts);
"""

LEVEL_NAMES = ("low", "normal", "high", "critical")
"""``level`` 0..3 — the single knob deciding how loud a reminder about this event gets."""

MAX_LEVEL = len(LEVEL_NAMES) - 1


def level_name(level: int) -> str:
    if 0 <= level <= MAX_LEVEL:
        return LEVEL_NAMES[level]
    return f"level{level}"


def _row(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


class CalendarStore:
    """One sqlite connection over the shared calendar file.

    All access happens on the owning event loop, so the connection is used by a single
    thread; ``check_same_thread=False`` only exists so a test or the reminder loop may
    straddle a thread boundary without tripping sqlite's guard.
    """

    def __init__(self, db_path: str | Path, *, now: Callable[[], float] = time.time) -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._now = now
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        # Set on every write so the reminder loop can re-evaluate immediately instead of
        # waiting out its poll cap when an event is created a few seconds from now.
        self.changed = asyncio.Event()
        self._edits: deque[tuple[float, str, str]] = deque(maxlen=40)

    # -- lifecycle --

    def close(self) -> None:
        self._conn.close()

    def now(self) -> float:
        return self._now()

    # -- edits log (in-memory; the page and the channel both surface it) --

    def log_edit(self, source: str, text: str) -> None:
        self._edits.append((self._now(), source, text))

    def recent_edits(self, n: int = 5) -> list[str]:
        out = []
        for at, source, text in list(self._edits)[-n:]:
            out.append(f"[{time.strftime('%H:%M:%S', time.localtime(at))}] {source}: {text}")
        return out

    # -- writes --

    def add(
        self,
        title: str,
        start_ts: float,
        *,
        end_ts: float | None = None,
        all_day: bool = False,
        notes: str = "",
        level: int = 1,
        remind_before: float | None = None,
    ) -> dict[str, Any]:
        now = self._now()
        cur = self._conn.execute(
            "INSERT INTO events (title, start_ts, end_ts, all_day, notes, level, "
            "remind_before, created_ts, updated_ts) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                title, float(start_ts), None if end_ts is None else float(end_ts),
                1 if all_day else 0, notes, int(level), remind_before, now, now,
            ),
        )
        self._commit()
        return self.get(cur.lastrowid)  # type: ignore[arg-type]

    def update(self, event_id: int, **fields: Any) -> dict[str, Any] | None:
        """Patch an event. Moving the time or the reminder offset re-arms the reminder.

        Re-arming matters: an event dragged to a later hour on the page must be allowed to
        fire again, so ``fired_ts`` / ``overdue_ts`` are cleared whenever the schedule moves.
        """
        current = self.get(event_id)
        if current is None:
            return None
        allowed = {"title", "start_ts", "end_ts", "all_day", "notes", "level", "remind_before"}
        patch = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not patch:
            return current
        if "start_ts" in patch:
            patch["start_ts"] = float(patch["start_ts"])
        if "end_ts" in patch:
            patch["end_ts"] = float(patch["end_ts"])
        if "remind_before" in patch:
            patch["remind_before"] = float(patch["remind_before"])
        if "all_day" in patch:
            patch["all_day"] = 1 if patch["all_day"] else 0
        re_arm = "start_ts" in patch or "remind_before" in patch
        patch["updated_ts"] = self._now()
        sets = ", ".join(f"{k}=?" for k in patch)
        self._conn.execute(f"UPDATE events SET {sets} WHERE id=?", (*patch.values(), event_id))
        if re_arm:
            self._conn.execute(
                "UPDATE events SET fired_ts=NULL, overdue_ts=NULL WHERE id=?", (event_id,)
            )
        self._commit()
        return self.get(event_id)

    def remove(self, event_id: int) -> bool:
        cur = self._conn.execute("DELETE FROM events WHERE id=?", (event_id,))
        self._commit()
        return cur.rowcount > 0

    def set_done(self, event_id: int, done: bool = True) -> dict[str, Any] | None:
        if self.get(event_id) is None:
            return None
        self._conn.execute(
            "UPDATE events SET done_ts=?, updated_ts=? WHERE id=?",
            (self._now() if done else None, self._now(), event_id),
        )
        self._commit()
        return self.get(event_id)

    def mark_fired(self, event_id: int, ts: float | None = None) -> None:
        self._conn.execute(
            "UPDATE events SET fired_ts=? WHERE id=?", (self._now() if ts is None else ts, event_id)
        )
        self._commit()

    def mark_overdue(self, event_id: int, ts: float | None = None) -> None:
        self._conn.execute(
            "UPDATE events SET overdue_ts=? WHERE id=?",
            (self._now() if ts is None else ts, event_id),
        )
        self._commit()

    def _commit(self) -> None:
        self._conn.commit()
        self.changed.set()

    # -- reads --

    def get(self, event_id: int) -> dict[str, Any] | None:
        row = self._conn.execute("SELECT * FROM events WHERE id=?", (event_id,)).fetchone()
        return _row(row) if row is not None else None

    def between(self, start_ts: float, end_ts: float, *, include_done: bool = True) -> list[dict[str, Any]]:
        """Events starting within ``[start_ts, end_ts)``, ordered by start."""
        sql = "SELECT * FROM events WHERE start_ts >= ? AND start_ts < ?"
        args: list[Any] = [start_ts, end_ts]
        if not include_done:
            sql += " AND done_ts IS NULL"
        sql += " ORDER BY start_ts ASC"
        return [_row(r) for r in self._conn.execute(sql, args).fetchall()]

    def upcoming(self, from_ts: float, limit: int = 20, *, include_done: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM events WHERE start_ts >= ?"
        if not include_done:
            sql += " AND done_ts IS NULL"
        sql += " ORDER BY start_ts ASC LIMIT ?"
        return [_row(r) for r in self._conn.execute(sql, (from_ts, int(limit))).fetchall()]

    def pending_reminders(self, from_ts: float) -> int:
        """How many future events still carry an armed at-time reminder."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM events WHERE remind_before IS NOT NULL AND "
            "fired_ts IS NULL AND done_ts IS NULL AND start_ts > ?",
            (from_ts,),
        ).fetchone()
        return int(row["n"])

    def due_at_time(self, now_ts: float) -> list[dict[str, Any]]:
        """Events whose reminder moment has arrived and which have not fired yet.

        An event with no reminder offset is never due here — a calendar entry without a
        reminder is a note on the page, not something the ghost should be interrupted for.
        """
        rows = self._conn.execute(
            "SELECT * FROM events WHERE remind_before IS NOT NULL AND fired_ts IS NULL AND "
            "done_ts IS NULL AND start_ts - remind_before <= ? ORDER BY start_ts ASC",
            (now_ts,),
        ).fetchall()
        return [_row(r) for r in rows]

    def due_overdue(self, now_ts: float, grace: float) -> list[dict[str, Any]]:
        """Events that started more than ``grace`` seconds ago, unhandled, and never escalated.

        Covers everything that already fired its at-time reminder; the escalation gate
        (whether the event's level is loud enough) is the caller's call, not the store's.
        """
        rows = self._conn.execute(
            "SELECT * FROM events WHERE overdue_ts IS NULL AND done_ts IS NULL AND "
            "start_ts + ? <= ? ORDER BY start_ts ASC",
            (grace, now_ts),
        ).fetchall()
        return [_row(r) for r in rows]

    def next_due_ts(self, now_ts: float) -> float | None:
        """Earliest future reminder moment, or None when nothing is armed."""
        row = self._conn.execute(
            "SELECT MIN(start_ts - remind_before) AS t FROM events WHERE remind_before IS NOT NULL "
            "AND fired_ts IS NULL AND done_ts IS NULL AND start_ts - remind_before > ?",
            (now_ts,),
        ).fetchone()
        return None if row is None or row["t"] is None else float(row["t"])
