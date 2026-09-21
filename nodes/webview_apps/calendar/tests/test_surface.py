"""CalendarSurface — the visible web: it serves a page and turns human edits into store writes.

This is the part the whole node exists for. The store and the loop are tested in isolation;
here the question is whether the two faces actually meet over one port.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from websockets.asyncio.client import connect

from ghoshell_calendar.store import CalendarStore
from ghoshell_calendar.surface import CalendarSurface

pytestmark = pytest.mark.asyncio


@pytest.fixture()
def store(tmp_path) -> CalendarStore:
    s = CalendarStore(tmp_path / "calendar.db")
    yield s
    s.close()


@pytest_asyncio.fixture
async def surface(store) -> CalendarSurface:
    edits: list[str] = []
    s = CalendarSurface(
        store,
        html_path=Path(__file__).resolve().parent.parent / "index.html",
        on_human_edit=edits.append,
        host="127.0.0.1",
        port=0,
    )
    await s.start()
    yield s, edits
    await s.stop()


async def test_serves_the_page(surface):
    s, _ = surface
    async with httpx.AsyncClient() as client:
        r = await client.get(s.url + "/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "FullCalendar" in r.text


async def test_state_frame_and_human_edit_round_trip(surface, store):
    s, edits = surface
    async with connect(s.url.replace("http", "ws") + "/ws") as ws:
        state = json.loads(await ws.recv())
        assert state["type"] == "state"
        assert state["events"] == []

        await ws.send(json.dumps({
            "type": "event.add",
            "title": "from the page",
            "start_ts": 1_790_000_000.0,
            "level": 2,
            "remind_before": 600,
        }))
        upsert = json.loads(await ws.recv())
        assert upsert["type"] == "event.upsert"
        assert upsert["event"]["title"] == "from the page"
        assert upsert["event"]["level"] == 2

    # The edit was written to the shared store and announced to the ghost's hook.
    assert store.upcoming(0)[0]["title"] == "from the page"
    assert edits and "from the page" in edits[0]


async def test_two_clients_both_see_a_write(surface):
    s, _ = surface
    async with connect(s.url.replace("http", "ws") + "/ws") as a, \
               connect(s.url.replace("http", "ws") + "/ws") as b:
        json.loads(await a.recv())
        json.loads(await b.recv())

        await a.send(json.dumps({
            "type": "event.add", "title": "shared", "start_ts": 1_790_000_000.0,
        }))
        upsert_a = json.loads(await a.recv())
        upsert_b = json.loads(await b.recv())
        assert upsert_a["event"]["title"] == "shared"
        assert upsert_b["event"]["title"] == "shared"
