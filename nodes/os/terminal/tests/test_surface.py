import asyncio
import json
from pathlib import Path

import pytest
from websockets.asyncio.client import connect

from ghoshell_terminal.card import CardState, CardType
from ghoshell_terminal.store import CardStore
from ghoshell_terminal.surface import StopHandles, TerminalSurface

_INDEX_HTML = Path(__file__).resolve().parent.parent / "index.html"


class _FakeLLM:
    def __init__(self, answer="it lists files"):
        self.answer = answer
        self.prompts = []

    async def call(self, *, instruction, prompt, **kw):
        self.prompts.append(prompt)
        return _FakeResult(self.answer)


class _FakeResult:
    def __init__(self, content):
        self.content = content


@pytest.fixture
def store(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    return CardStore(root=root, outputs_dir=tmp_path / "out")


@pytest.fixture
def signals():
    return []


@pytest.fixture
def stops():
    return StopHandles()


async def _surface(store, signals, stops, llm_funcs=None, groundset=None):
    surface = TerminalSurface(
        store,
        send_signal=signals.append,
        self_identity="cell:test",
        host="127.0.0.1",
        port=0,
        html_path=_INDEX_HTML,
        stops=stops,
        llm_funcs=llm_funcs,
        groundset=groundset,
    )
    await surface.start()
    return surface


def _pending(store, content="ls -la"):
    card = store.new_card(CardType.COMMAND, title="dev", thread="dev", cwd=str(store.root))
    store.append_content(card.id, content)
    store.set_state(card.id, CardState.AWAITING)
    return card


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
async def test_serves_the_page_and_snapshots_on_connect(store, signals, stops):
    surface = await _surface(store, signals, stops)
    try:
        page = await _http_get(surface.port)
        assert b"moss terminal" in page

        card = _pending(store)
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            snap = await _recv(ws, "snapshot")
            assert snap["mode"] == "approval"
            assert [c["id"] for c in snap["cards"]] == [card.id]
            assert snap["cards"][0]["interactions"] == ["accept", "deny", "ask"]
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_accept_settles_and_tells_the_ghost(store, signals, stops):
    surface = await _surface(store, signals, stops)
    try:
        card = _pending(store)
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "accept", "id": card.id}))
            await asyncio.sleep(0.1)

        assert store.waiter(card.id).result() == "accept"
        assert len(signals) == 1
        assert signals[0].name == "aside"
        assert "accepted" in signals[0].messages[0].to_content_string()
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_a_second_verdict_on_the_same_card_is_swallowed(store, signals, stops):
    surface = await _surface(store, signals, stops)
    try:
        card = _pending(store)
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "accept", "id": card.id}))
            await ws.send(json.dumps({"type": "deny", "id": card.id}))
            await ws.send(json.dumps({"type": "accept", "id": card.id}))
            await asyncio.sleep(0.1)

        assert len(signals) == 1, "one card, one decision"
        assert store.waiter(card.id).result() == "accept"
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_a_settled_card_takes_no_verdict_at_all(store, signals, stops):
    surface = await _surface(store, signals, stops)
    try:
        card = _pending(store)
        store.set_state(card.id, CardState.RUNNING)
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "accept", "id": card.id}))
            await asyncio.sleep(0.1)
        assert signals == []
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_ask_records_dialogue_and_decides_nothing(store, signals, stops):
    surface = await _surface(store, signals, stops)
    try:
        card = _pending(store)
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "ask", "id": card.id, "text": "why -la?"}))
            full = await _recv(ws, "card.full")
            assert full["card"]["dialogue"][0]["text"] == "why -la?"

        assert card.state is CardState.AWAITING
        assert store.waiter(card.id).done() is False
        assert len(signals) == 1, "ask is a question, not a decision — no batch notify"
        assert signals[0].name == "notify"
        assert "why -la?" in signals[0].messages[0].to_content_string()
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_thread_auto_toggle_reaches_the_store(store, signals, stops):
    surface = await _surface(store, signals, stops)
    try:
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "thread_auto", "name": "root", "auto": True}))
            frame = await _recv(ws, "threads")
            assert any(t["name"] == "root" and t["auto"] is True for t in frame["threads"])
        assert store.get_thread("root").auto is True
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_auto_toggle_approves_pending_cards_in_the_thread(store, signals, stops):
    surface = await _surface(store, signals, stops)
    try:
        store.open_thread("dev", ".")
        store.open_thread("ops", ".")
        dev = store.new_card(CardType.COMMAND, title="a", thread="dev", cwd=str(store.root))
        store.set_state(dev.id, CardState.AWAITING)
        ops = store.new_card(CardType.COMMAND, title="b", thread="ops", cwd=str(store.root))
        store.set_state(ops.id, CardState.AWAITING)

        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "thread_auto", "name": "dev", "auto": True}))
            await asyncio.sleep(0.1)

        assert store.waiter(dev.id).result() == "accept", "auto settles the dev card"
        assert store.waiter(ops.id).done() is False, "ops card is untouched"
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_accept_with_text_records_the_note(store, signals, stops):
    surface = await _surface(store, signals, stops)
    try:
        card = _pending(store)
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "accept", "id": card.id, "text": "ok but watch -la"}))
            await asyncio.sleep(0.1)
        assert store.waiter(card.id).result() == "accept"
        assert card.dialogue[0].text == "ok but watch -la"
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_analyze_returns_a_zero_context_answer(store, signals, stops):
    llm = _FakeLLM()
    surface = await _surface(store, signals, stops, llm_funcs=lambda: llm)
    try:
        card = _pending(store, "rm -rf /")
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "analyze", "id": card.id, "text": "is this safe?"}))
            frame = await _recv(ws, "analyze")
        assert frame["text"] == "it lists files"
        prompt = llm.prompts[0]
        assert "rm -rf /" in prompt
        assert "is this safe?" in prompt
        assert "dev" not in prompt, "the issuing model's title must not leak into the analyzer"
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_analyze_reports_unregistered(store, signals, stops):
    surface = await _surface(store, signals, stops, llm_funcs=lambda: None)
    try:
        card = _pending(store)
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "analyze", "id": card.id, "text": "is this safe?"}))
            frame = await _recv(ws, "error")
            assert "not registered" in frame["text"]
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_ground_view_renders_the_field(store, signals, stops, tmp_path):
    from ghoshell_moss.ground import DefaultGroundSet

    root = tmp_path / "groot"
    root.mkdir()
    (root / "GROUND.md").write_text("---\nname: p\n---\n\nwelcome field\n")
    store2 = CardStore(root=root, outputs_dir=tmp_path / "out2")
    groundset = DefaultGroundSet(workspace_root=root, materialize=False)
    surface = await _surface(store2, signals, stops, groundset=groundset)
    try:
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "ground", "thread": "root"}))
            frame = await _recv(ws, "ground")
            assert "welcome field" in frame["text"]
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_mode_switch_broadcasts_and_is_seen_by_the_store(store, signals, stops):
    surface = await _surface(store, signals, stops)
    try:
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "mode", "mode": "disabled"}))
            frame = await _recv(ws, "mode")
        assert frame["mode"] == "disabled"
        assert store.mode == "disabled"

        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await ws.send(json.dumps({"type": "mode", "mode": "nonsense"}))
            assert (await _recv(ws, "error"))["type"] == "error"
        assert store.mode == "disabled"
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_accept_all_settles_every_pending_card(store, signals, stops):
    surface = await _surface(store, signals, stops)
    try:
        cards = [_pending(store, f"cmd {i}") for i in range(3)]
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "accept_all"}))
            await asyncio.sleep(0.1)
        assert [store.waiter(c.id).result() for c in cards] == ["accept"] * 3
        assert len(signals) == 3
        assert [s.name for s in signals] == ["aside", "aside", "aside"]
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_deny_all_settles_every_pending_card(store, signals, stops):
    surface = await _surface(store, signals, stops)
    try:
        cards = [_pending(store, f"cmd {i}") for i in range(3)]
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "deny_all"}))
            await asyncio.sleep(0.1)
        assert [store.waiter(c.id).result() for c in cards] == ["deny"] * 3
        assert len(signals) == 3
        assert [s.name for s in signals] == ["aside", "aside", "aside"]
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_stop_buttons_reach_the_channel_handles(store, signals, stops):
    called = []
    surface = await _surface(store, signals, stops)

    async def stop_card(card_id):
        called.append(("stop", card_id))

    async def stop_all():
        called.append(("all", 0))

    stops.stop = stop_card
    stops.stop_all = stop_all
    try:
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
            await ws.send(json.dumps({"type": "stop", "id": 7}))
            await ws.send(json.dumps({"type": "stop_all"}))
            await asyncio.sleep(0.1)
        assert called == [("stop", 7), ("all", 0)]
    finally:
        await surface.stop()


@pytest.mark.asyncio
async def test_disconnect_is_survivable(store, signals, stops):
    surface = await _surface(store, signals, stops)
    try:
        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            await _recv(ws, "snapshot")
        await surface.broadcast({"type": "mode", "mode": "auto"})

        async with connect(f"ws://127.0.0.1:{surface.port}/ws") as ws:
            assert (await _recv(ws, "snapshot"))["mode"] == "approval"
    finally:
        await surface.stop()
