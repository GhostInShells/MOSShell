import asyncio
import json
from pathlib import Path

import pytest
from websockets.asyncio.client import connect

from ghoshell_file_editor.store import DocStore
from ghoshell_file_editor.surface import FileEditorSurface

_INDEX_HTML = Path(__file__).resolve().parent.parent / "index.html"


@pytest.fixture
def store(tmp_path):
    return DocStore(drafts_dir=tmp_path / "drafts", root=tmp_path)


@pytest.fixture
def signals():
    return []


async def _surface(store, signals, toggles=None):
    surface = FileEditorSurface(
        store,
        send_signal=signals.append,
        self_identity="cell:test",
        on_toggle=(toggles.append if toggles is not None else None),
        host="127.0.0.1",
        port=0,
        html_path=_INDEX_HTML,
    )
    await surface.start()
    return surface


def _pending_export(store):
    store.open("t", "doc")
    action = store.begin("t", "append", "add")
    store.feed("t", action.n, "final\n")
    store.tail("t", action.n)
    exp = store.record(
        "t", "export", "export to /tmp/x", text="final\n",
        payload="/tmp/x", state="awaiting",
    )
    return exp


async def _recv(ws, kind, timeout=3.0):
    while True:
        frame = json.loads(await asyncio.wait_for(ws.recv(), timeout))
        if frame["type"] == kind:
            return frame


async def _http_get(port: int) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        f"GET / HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n".encode()
    )
    await writer.drain()
    data = await reader.read()
    writer.close()
    return data


@pytest.mark.asyncio
async def test_serves_the_page_and_snapshots_on_connect(store, signals):
    surface = await _surface(store, signals)
    try:
        page = await _http_get(surface.port)
        assert b"moss file_editor" in page

        exp = _pending_export(store)
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            snap = await _recv(ws, "snapshot")
            assert snap["threads"][0]["id"] == "t"
            kinds = [a["kind"] for a in snap["actions"]]
            assert kinds == ["append", "export"]
            assert snap["actions"][-1]["n"] == exp.n
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_accept_settles_and_signals(store, signals):
    surface = await _surface(store, signals)
    try:
        exp = _pending_export(store)
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "accept", "thread": "t", "n": exp.n}))
            # a repeat click inside the debounce window is the same decision
            await ws.send(json.dumps({"type": "accept", "thread": "t", "n": exp.n}))
            await asyncio.sleep(0.05)
        # settle is handed down; the channel (absent here) does the state move
        assert len(signals) == 1
        assert "accepted" in signals[0].messages[0].to_content_string()
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_ask_records_dialogue_and_notifies(store, signals):
    surface = await _surface(store, signals)
    try:
        exp = _pending_export(store)
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps(
                {"type": "ask", "thread": "t", "n": exp.n, "text": "why?"}
            ))
            frame = await _recv(ws, "action")
            assert frame["dialogue"][0]["text"] == "why?"
        assert "why?" in signals[0].messages[0].to_content_string()
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_detail_returns_source_and_full(store, signals):
    surface = await _surface(store, signals)
    try:
        exp = _pending_export(store)
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "detail", "thread": "t", "n": exp.n}))
            frame = await _recv(ws, "detail")
            assert frame["source"] == "final\n"
            assert frame["full"] == "final\n"
            assert "writes /tmp/x" in frame["effect"]
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_toggle_reaches_the_gate(store, signals):
    toggles = []
    surface = await _surface(store, signals, toggles=toggles)
    try:
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "toggle", "enabled": False}))
            await asyncio.sleep(0.05)
        assert toggles == [False]
    finally:
        await surface.stop()
