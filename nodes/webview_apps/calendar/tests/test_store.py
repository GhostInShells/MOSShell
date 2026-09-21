"""CalendarStore — the shared truth, and the queries the reminder loop's three tiers hang on."""

from __future__ import annotations

import sqlite3
import time

import pytest

from ghoshell_calendar.store import CalendarStore


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def clock() -> Clock:
    return Clock(1_771_000_000.0)


@pytest.fixture()
def store(tmp_path, clock) -> CalendarStore:
    s = CalendarStore(tmp_path / "calendar.db", now=clock)
    yield s
    s.close()


def test_add_and_read_back(store):
    event = store.add("standup", 1_771_003_600.0, level=2, remind_before=600)
    assert event["title"] == "standup"
    assert event["level"] == 2
    assert event["remind_before"] == 600
    assert event["fired_ts"] is None
    assert store.get(event["id"]) == event


def test_between_is_half_open_and_ordered(store):
    base = 1_771_000_000.0
    store.add("late", base + 300)
    store.add("early", base + 100)
    store.add("outsider", base + 10_000)
    found = store.between(base, base + 1000)
    assert [e["title"] for e in found] == ["early", "late"]


def test_only_events_with_a_reminder_are_ever_due(store, clock):
    """A calendar entry with no reminder is a note on the page, not an interruption."""
    store.add("note only", clock.t + 10)
    store.add("reminded", clock.t + 10, remind_before=30)
    due = store.due_at_time(clock.t)
    assert [e["title"] for e in due] == ["reminded"]


def test_at_time_due_fires_once(store, clock):
    event = store.add("soon", clock.t + 100, remind_before=30)
    assert store.due_at_time(clock.t + 50) == []
    assert [e["id"] for e in store.due_at_time(clock.t + 70)] == [event["id"]]
    store.mark_fired(event["id"], clock.t + 70)
    assert store.due_at_time(clock.t + 1000) == []


def test_done_event_stops_being_due(store, clock):
    event = store.add("soon", clock.t + 10, remind_before=0)
    store.set_done(event["id"])
    assert store.due_at_time(clock.t + 10) == []
    assert store.due_overdue(clock.t + 10_000, grace=60) == []


def test_overdue_needs_the_grace_to_elapse(store, clock):
    store.add("started", clock.t - 10)
    assert store.due_overdue(clock.t, grace=60) == []
    assert len(store.due_overdue(clock.t + 60, grace=60)) == 1


def test_next_due_ts_ignores_past_and_fired(store, clock):
    store.add("past", clock.t - 1000, remind_before=0)
    fired = store.add("fired", clock.t + 500, remind_before=0)
    store.mark_fired(fired["id"])
    future = store.add("future", clock.t + 900, remind_before=60)
    assert store.next_due_ts(clock.t) == future["start_ts"] - 60


def test_moving_the_time_rearms_the_reminder(store, clock):
    event = store.add("movable", clock.t + 100, remind_before=0)
    store.mark_fired(event["id"], clock.t + 100)
    store.mark_overdue(event["id"], clock.t + 200)
    moved = store.update(event["id"], start_ts=clock.t + 5000)
    assert moved["fired_ts"] is None
    assert moved["overdue_ts"] is None
    assert store.due_at_time(clock.t + 5000) != []


def test_update_ignores_unknown_and_none_fields(store):
    event = store.add("plain", 1_771_003_600.0)
    store.update(event["id"], nonsense="ignored", notes=None)
    assert store.get(event["id"])["notes"] == ""


def test_update_missing_event_returns_none(store):
    assert store.update(999, title="ghost") is None


def test_remove(store):
    event = store.add("doomed", 1_771_003_600.0)
    assert store.remove(event["id"]) is True
    assert store.remove(event["id"]) is False
    assert store.get(event["id"]) is None


def test_pending_reminders_counts_only_armed_future_ones(store, clock):
    store.add("armed", clock.t + 500, remind_before=60)
    store.add("no reminder", clock.t + 500)
    store.add("past", clock.t - 500, remind_before=0)
    closed = store.add("done", clock.t + 500, remind_before=0)
    store.set_done(closed["id"])
    assert store.pending_reminders(clock.t) == 1


def test_writes_set_the_changed_event(store):
    """The reminder loop waits on this instead of sleeping out its poll cap."""
    store.changed.clear()
    store.add("x", 1_771_003_600.0)
    assert store.changed.is_set()


def test_a_second_connection_reads_what_this_one_wrote(tmp_path, store):
    """The file is the bus: another process opening the same db sees the write.

    WAL is what makes this safe — without it this read would block or come back empty.
    """
    store.add("shared", 1_771_003_600.0, remind_before=60)

    other = sqlite3.connect(str(store.path))
    try:
        assert other.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        rows = other.execute("SELECT title, remind_before FROM events").fetchall()
        assert rows == [("shared", 60.0)]
    finally:
        other.close()


def test_database_survives_reopen(tmp_path, clock):
    first = CalendarStore(tmp_path / "calendar.db", now=clock)
    first.add("persisted", clock.t + 100, remind_before=60)
    first.close()

    second = CalendarStore(tmp_path / "calendar.db", now=time.time)
    try:
        assert [e["title"] for e in second.upcoming(clock.t)] == ["persisted"]
    finally:
        second.close()
