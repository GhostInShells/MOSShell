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


@pytest.mark.asyncio
async def test_layout_is_the_model_side_of_the_presentation_contract(store, signals):
    surface = await _surface(store, signals)
    try:
        # the default is stated, not implied — the page can always name its author.
        # split: a card opens under itself, inside the list; page is the exception
        # for reading one card alone.
        assert surface.layout_state() == ("split", "default")
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            snap = await _recv(ws, "snapshot")
            assert snap["layout"] == "split"
            assert snap["layout_by"] == "default"
            # every connected page learns the change, so the marker cannot lie
            assert await surface.set_layout("page") == "page"
            frame = await _recv(ws, "layout")
            assert frame["layout"] == "page"
            assert frame["by"] == "model"
        assert surface.layout_state() == ("page", "model")
        with pytest.raises(ValueError):
            await surface.set_layout("cinema")
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_flipping_trust_is_state_not_a_signal(store, signals, tmp_path):
    """Trusting a directory must not reach out and interrupt: it is in the notice
    every turn already, and a push here aborted whatever the model was saying."""
    work = tmp_path / "work"
    work.mkdir()
    surface = FileEditorSurface(
        store,
        send_signal=signals.append,
        self_identity="cell:test",
        host="127.0.0.1",
        port=0,
        html_path=_INDEX_HTML,
    )
    await surface._set_dir_trust(str(work), True)
    await surface._set_global_trust(True)
    assert store.trusted() == [str(work)]
    assert store.global_trust() is True
    assert signals == [], "state is read, not announced"


async def _http_get_path(port: int, path: str) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        f"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n".encode()
    )
    await writer.drain()
    data = await reader.read()
    writer.close()
    return data


@pytest.mark.asyncio
async def test_the_renderer_library_is_a_directory_served_as_modules(store, signals):
    """A renderer is a file: adding one is writing a file, and the page finds it.

    The listing is read from the files' own header comments, so discovering the
    library never executes model-written code — discovery has no side effects.
    """
    surface = await _surface(store, signals)
    try:
        listing = surface.list_renderers()
        names = {r["name"] for r in listing}
        assert {"raw", "markdown", "code", "plain"} <= names
        md = next(r for r in listing if r["name"] == "markdown")
        assert ".md" in md["ext"] and md["description"]

        raw = await _http_get_path(surface.port, "/renderers")
        assert b'"renderers"' in raw
        body = await _http_get_path(surface.port, "/renderers/markdown.js")
        head = body.split(b"\r\n\r\n", 1)[-1]
        assert b"export default" in head, "served as the module it is"
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_the_renderer_route_cannot_reach_outside_its_directory(store, signals):
    """Only files in the library: the route takes a name, not a path."""
    surface = await _surface(store, signals)
    try:
        body = await _http_get_path(surface.port, "/renderers/..%2f..%2findex.html")
        assert b"404" in body.split(b"\r\n", 1)[0]
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_format_accepts_a_name_from_the_library_and_a_draft(store, signals):
    surface = await _surface(store, signals)
    try:
        assert await surface.set_format("code") == "code"
        assert await surface.set_format("tmp") == "tmp"
        with pytest.raises(ValueError):
            await surface.set_format("html")
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_a_pushed_preview_reaches_the_page_as_a_draft(store, signals):
    """The draft layer: a script pushed at runtime becomes one more face."""
    surface = await _surface(store, signals)
    try:
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            shown = await surface.push_preview("t", "return '<b>x</b>'", "只看标题")
            frame = await _recv(ws, "preview")
            assert frame["script"] == "return '<b>x</b>'"
            assert frame["label"] == "只看标题"
            assert frame["thread"] == "t" and frame["seq"] == 1
            assert shown["seq"] == 1
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_drafts_are_a_string_of_materials_per_thread(store, signals):
    """A draft is not replaced and forgotten: it stays, and can be drawn again.

    The ring is per thread because a draft reads that thread's working copy —
    two documents do not share a face.
    """
    surface = await _surface(store, signals)
    try:
        first = await surface.push_preview("t", "return 'a'", "第一版")
        second = await surface.push_preview("t", "return 'b'", "第二版")
        other = await surface.push_preview("u", "return 'c'", "别的文档")
        assert (first["seq"], second["seq"], other["seq"]) == (1, 2, 1)

        ring = surface.drafts("t")
        assert [d["label"] for d in ring] == ["第一版", "第二版"]
        assert surface.draft_threads() == ["t", "u"]

        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await surface.show_draft("t", 1)          # go back to the first face
            frame = await _recv(ws, "preview_show")
            assert frame == {"type": "preview_show", "thread": "t", "seq": 1}

        with pytest.raises(ValueError):
            await surface.show_draft("t", 99)
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_the_humans_edit_lands_as_an_action_and_releases_the_lock(store, signals):
    """One action, author `u`, same history — so "who wrote this line" has one
    answer. The lock goes away with the commit, not with a timeout."""
    surface = await _surface(store, signals)
    store.open("t", "doc")
    act = store.begin("t", "write", "seed")     # the ordinary path: stream, then land
    store.feed("t", act.n, "one\ntwo\nthree\n")
    store.tail("t", act.n)
    try:
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            lock = await surface.begin_edit("t", 2, 2)
            assert lock["start"] == 2 and surface.locked("t")
            await _recv(ws, "lock")

            action = await surface.commit_edit("t", "2-2", "TWO\n", "he fixed a word", "c-1")
            assert action.author == "u"
            assert store.get("t").content.split("\n")[1] == "TWO"
            assert not surface.locked("t")
            await _recv(ws, "lock")          # released first, then the action
            landed = await _recv(ws, "landed")
            assert landed["client_id"] == "c-1" and landed["by"] == "human"
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_only_the_claiming_window_can_edit(store, signals):
    """An edit lands on the copy we both see — two live windows would be two hands."""
    surface = await _surface(store, signals)
    try:
        exp = _pending_export(store)
        url = f"ws://127.0.0.1:{surface.port}/ws"
        async with connect(url) as first:
            await _recv(first, "snapshot")
            await _recv(first, "current")
            async with connect(url) as second:
                await _recv(second, "snapshot")
                mask = await _recv(first, "mask")
                assert "another window" in mask["reason"]

                await first.send(json.dumps({"type": "accept", "thread": "t", "n": exp.n}))
                await asyncio.sleep(0.1)
                assert signals == [], "a reading window settled an export"

                await second.send(json.dumps({"type": "accept", "thread": "t", "n": exp.n}))
                await asyncio.sleep(0.1)
                assert signals and "accepted" in signals[0].messages[0].to_content_string()
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_the_editor_surface_can_be_taken_back(store, signals):
    surface = await _surface(store, signals)
    try:
        url = f"ws://127.0.0.1:{surface.port}/ws"
        async with connect(url) as first:
            await _recv(first, "snapshot")
            await _recv(first, "current")
            async with connect(url) as second:
                await _recv(second, "snapshot")
                await _recv(first, "mask")

                await first.send(json.dumps({"type": "claim"}))
                assert (await _recv(first, "current"))["type"] == "current"
                assert (await _recv(second, "mask"))["type"] == "mask"
    finally:
        await surface.stop()
