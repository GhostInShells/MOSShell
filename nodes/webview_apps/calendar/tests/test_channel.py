"""The channel — time parsing and the round-trip through a real CTML shell.

The store and the reminder loop are tested on their own; here the question is narrower:
does the ghost's half parse what a model naturally writes, and do the commands land in the
same shared store that a human would read on the page.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from ghoshell_calendar.channel import new_calendar_channel, parse_remind, parse_when
from ghoshell_calendar.store import CalendarStore


# -- parsing ----------------------------------------------------------------------


def test_parse_when_full_date():
    assert parse_when("2026-09-22 14:30", now=1_771_000_000.0) == datetime(2026, 9, 22, 14, 30).timestamp()


def test_parse_when_date_only_is_midnight():
    assert parse_when("2026-09-22", now=1_771_000_000.0) == datetime(2026, 9, 22).timestamp()


def test_parse_when_bare_time_means_today():
    now = datetime(2026, 9, 22, 14, 0).timestamp()
    assert parse_when("09:00", now=now) == datetime(2026, 9, 22, 9, 0).timestamp()


def test_parse_when_rejects_garbage():
    with pytest.raises(ValueError):
        parse_when("next tuesday", now=1_771_000_000.0)


def test_parse_remind_units():
    assert parse_remind("15m") == 900
    assert parse_remind("2h") == 7200
    assert parse_remind("1d") == 86400
    assert parse_remind("0") == 0.0


def test_parse_remind_empty_means_no_reminder():
    assert parse_remind("") is None
    assert parse_remind("none") is None


# -- channel round-trip -----------------------------------------------------------


class _StubSurface:
    url = "http://127.0.0.1:1234"
    started = False
    stopped = False

    async def broadcast(self, frame):
        self.last = frame

    async def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


class _StubLoop:
    started = False
    stopped = False

    def start(self):
        self.started = True

    async def stop(self):
        self.stopped = True


@pytest.fixture()
def clock():
    return datetime(2026, 9, 22, 12, 0, 0).timestamp()


@pytest.fixture()
def store(tmp_path, clock):
    s = CalendarStore(tmp_path / "calendar.db", now=lambda: clock)
    yield s
    s.close()


@pytest.mark.asyncio
async def test_add_then_agenda_round_trip(store, surface=None, loop=None):
    from ghoshell_moss.core.ctml import new_ctml_shell

    surface = surface or _StubSurface()
    loop = loop or _StubLoop()
    chan = new_calendar_channel(store, surface, loop)

    shell = new_ctml_shell()
    shell.main_channel.import_channels(chan)

    async def run(logos: str) -> str:
        interpreter = await shell.interpreter()
        tasks = await interpreter.run(logos)
        return "\n".join(str(t.result(throw=False)) for t in tasks.values())

    async with shell:
        result = await run('<calendar:add title="standup" start="2026-09-22 15:00" remind="10m" level="2"/>')
        assert "standup" in result and result.startswith("#")

        agenda = await run('<calendar:agenda days="7"/>')
        assert "standup" in agenda

        await shell.refresh_metas()
        meta = shell.channel_metas().get("calendar")
        assert meta is not None
        assert meta.named_notices["url"] == "http://127.0.0.1:1234"
        assert meta.named_notices["db"] == str(store.path)
        # The interval fragment is a delta: it exists today because there is an event.
        assert "standup" in (meta.named_notices["today"] or "")

    assert surface.started and surface.stopped
    assert loop.started and loop.stopped


@pytest.mark.asyncio
async def test_done_and_remove(store):
    from ghoshell_moss.core.ctml import new_ctml_shell

    chan = new_calendar_channel(store, _StubSurface(), _StubLoop())
    shell = new_ctml_shell()
    shell.main_channel.import_channels(chan)

    async def run(logos: str) -> None:
        interpreter = await shell.interpreter()
        await interpreter.run(logos)

    async with shell:
        await run('<calendar:add title="x" start="2026-09-23 09:00"/>')
        ev = store.upcoming(datetime(2026, 9, 22, 12, 0).timestamp())[0]

        await run(f'<calendar:done event_id="{ev["id"]}"/>')
        assert store.get(ev["id"])["done_ts"] is not None

        await run(f'<calendar:remove event_id="{ev["id"]}"/>')
        assert store.get(ev["id"]) is None
