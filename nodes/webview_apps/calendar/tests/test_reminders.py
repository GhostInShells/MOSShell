"""ReminderLoop — the three tiers, driven by a fake clock so no test sleeps.

The tiers are the design: quiet things wait, loud things push, and only the loud ones are
allowed to claim the ghost's next turn when they go overdue. Each of those is a claim
worth pinning down.

The clock is anchored at local noon, so "today" and "this hour" are unambiguous and an
event an hour out is still the same day.
"""

from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from ghoshell_moss.core.blueprint.mindflow import Priority
from ghoshell_calendar.reminders import ReminderLoop
from ghoshell_calendar.store import CalendarStore

pytestmark = pytest.mark.asyncio

BASE = 1_771_000_000.0


def _noon(t: float) -> float:
    return datetime.fromtimestamp(t).replace(hour=12, minute=0, second=0, microsecond=0).timestamp()


class Clock:
    def __init__(self, t: float) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


class Recorder:
    """Collects what the loop pushed, split by the signal each push belongs to."""

    def __init__(self) -> None:
        self.signals = []
        self.frames = []

    def send(self, signal) -> None:
        self.signals.append(signal)

    async def broadcast(self, frame) -> None:
        self.frames.append(frame)

    def of(self, name: str) -> list:
        return [s for s in self.signals if s.name == name]

    def tiers(self) -> list[str]:
        return [f["tier"] for f in self.frames if f["type"] == "reminder.fired"]


@pytest.fixture()
def clock() -> Clock:
    return Clock(_noon(BASE))


@pytest.fixture()
def store(tmp_path, clock) -> CalendarStore:
    s = CalendarStore(tmp_path / "calendar.db", now=clock)
    yield s
    s.close()


@pytest.fixture()
def loop(store, clock):
    rec = Recorder()
    instance = ReminderLoop(
        store,
        send_signal=rec.send,
        broadcast=rec.broadcast,
        now=clock,
        poll_cap=0.0,
        overdue_grace=60.0,
    )
    return instance, rec


# -- tier 1: interval totals -------------------------------------------------------


async def test_interval_notice_fires_once_per_bucket(store, clock, loop):
    instance, rec = loop
    store.add("this hour's thing", clock.t + 600)

    await instance._tick()
    assert len(rec.of("aside")) == 2, "the day and the hour should each announce once"

    await instance._tick()  # same day, same hour — nothing changed, nothing re-sent
    assert len(rec.of("aside")) == 2


async def test_interval_notice_is_low_priority_and_never_preempts(store, clock, loop):
    instance, rec = loop
    store.add("this hour's thing", clock.t + 600, level=3)
    await instance._tick()
    assert {s.priority for s in rec.of("aside")} == {Priority.INFO}


async def test_an_empty_day_and_hour_say_nothing(store, clock, loop):
    instance, rec = loop
    await instance._tick()
    assert rec.signals == []


async def test_interval_bucket_rollover_announces_again(store, clock, loop):
    instance, rec = loop
    store.add("next hour", clock.t + 3600)

    await instance._tick()
    assert len(rec.of("aside")) == 1, "the day announces; the coming hour is not this hour"

    clock.t += 3600
    await instance._tick()
    fresh = rec.of("aside")[1:]
    assert len(fresh) == 1
    assert "next hour" in fresh[0].messages[0].to_content_string()


async def test_interval_listing_caps_the_enumeration(store, clock, loop):
    instance, rec = loop
    for i in range(12):
        store.add(f"e{i}", clock.t + 600 + i)
    await instance._tick()
    day = [s for s in rec.of("aside") if "12 event" in s.description]
    assert day, "the day aggregate did not mention all twelve events"
    # The enumeration is capped so a crowded day cannot blow up the ghost's context.
    assert "(+4 more)" in day[0].messages[0].to_content_string()


# -- tier 2: at the time -----------------------------------------------------------


async def test_at_time_reminder_arrives_as_input_weighted_by_level(store, clock, loop):
    instance, rec = loop
    store.add("loud", clock.t + 100, level=3, remind_before=30)
    store.add("quiet", clock.t + 200, level=0, remind_before=30)

    clock.t += 70
    await instance._tick()
    assert [s.priority for s in rec.of("input")] == [Priority.ERROR]

    clock.t += 100
    await instance._tick()
    assert [s.priority for s in rec.of("input")] == [Priority.ERROR, Priority.INFO]


async def test_at_time_reminder_fires_once(store, clock, loop):
    instance, rec = loop
    store.add("once", clock.t + 100, remind_before=0)
    clock.t += 100
    await instance._tick()
    await instance._tick()
    await instance._tick()
    assert len(rec.of("input")) == 1


async def test_an_event_without_a_reminder_never_interrupts(store, clock, loop):
    instance, rec = loop
    store.add("note only", clock.t + 100)
    clock.t += 200
    await instance._tick()
    assert rec.of("input") == []


async def test_reminder_also_reaches_the_page(store, clock, loop):
    instance, rec = loop
    store.add("visible", clock.t + 100, remind_before=0)
    clock.t += 100
    await instance._tick()
    assert rec.tiers() == ["at_time"]
    fired = [f for f in rec.frames if f["type"] == "reminder.fired"][0]
    assert fired["event"]["title"] == "visible"


async def test_a_write_wakes_the_tick_before_its_poll_cap(store, clock, loop):
    """An event created a few seconds out must not wait out the poll cap."""
    instance, rec = loop
    instance._poll_cap = 30.0
    task = asyncio.create_task(instance._run())
    await asyncio.sleep(0)  # let the loop enter its first wait
    store.add("just now", clock.t + 0.5, remind_before=0)  # sets store.changed
    clock.t += 1
    await asyncio.sleep(0.1)  # let the wake propagate into a fresh tick
    try:
        assert len(rec.of("input")) == 1
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


# -- tier 3: overdue ---------------------------------------------------------------


async def test_overdue_escalates_to_a_next_turn_notify(store, clock, loop):
    instance, rec = loop
    store.add("critical", clock.t - 100, level=3)
    await instance._tick()

    notifies = rec.of("notify")
    assert len(notifies) == 1
    assert notifies[0].metadata["next"] is True
    assert notifies[0].priority == Priority.ERROR
    assert rec.tiers() == ["overdue"]


async def test_overdue_quiet_events_do_not_claim_the_next_turn(store, clock, loop):
    """The escalation tier only means something if it is scarce."""
    instance, rec = loop
    store.add("low", clock.t - 100, level=0)
    store.add("normal", clock.t - 100, level=1)
    await instance._tick()

    assert rec.of("notify") == []
    assert rec.of("input") == []


async def test_overdue_waits_for_the_grace(store, clock, loop):
    instance, rec = loop
    store.add("critical", clock.t - 10, level=3)
    await instance._tick()
    assert rec.of("notify") == []

    clock.t += 120
    await instance._tick()
    assert len(rec.of("notify")) == 1


async def test_overdue_escalates_only_once(store, clock, loop):
    instance, rec = loop
    store.add("critical", clock.t - 100, level=3)
    await instance._tick()
    await instance._tick()
    await instance._tick()
    assert len(rec.of("notify")) == 1


async def test_a_done_event_never_escalates(store, clock, loop):
    instance, rec = loop
    event = store.add("handled", clock.t - 100, level=3)
    store.set_done(event["id"])
    await instance._tick()
    assert rec.of("notify") == []


# -- the ticker itself -------------------------------------------------------------


async def test_a_broken_tick_does_not_kill_the_ticker(store, clock, loop):
    """A calendar that silently stops reminding is worse than one that logs and carries on."""
    instance, rec = loop
    calls: list[float] = []
    survived = asyncio.Event()

    async def flaky():
        # _run calls self._tick() with no arguments, so the replacement is called bare.
        calls.append(len(calls))
        if len(calls) == 1:
            raise RuntimeError("storage hiccup")
        # Park instead of returning: this keeps _run on its second tick without ever
        # spinning, so the test can observe "one failure did not stop the loop" and then
        # cancel deterministically.
        survived.set()
        await asyncio.Event().wait()

    instance._tick = flaky  # type: ignore[method-assign]
    task = asyncio.create_task(instance._run())
    try:
        await asyncio.wait_for(survived.wait(), timeout=2.0)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert len(calls) == 2
