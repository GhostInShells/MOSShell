"""WebViewBridge: the mesh-view → store reconciliation, with the client mocked.

The bridge only talks to ``WebViewClient`` through ``items()`` and ``on_change``,
so a stand-in is enough — no operator, no mesh, no zenoh.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from ghoshell_screen_manager.bridge import WebViewBridge
from ghoshell_screen_manager.model import ScreenModel

from fakes import Recorder


@dataclass
class FakeDeclaration:
    url: str
    title: str
    icon: str | None = None


@dataclass
class FakeState:
    """The condition fields the bridge reads. Defaults mirror the real ones."""

    face_line: str | None = None
    face_art: str | None = None


@dataclass
class FakeItem:
    address: str
    declaration: FakeDeclaration
    state: FakeState = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.state is None:
            self.state = FakeState()


class FakeClient:
    """A WebViewClient stand-in: a mutable item list + change callbacks."""

    def __init__(self, items=None):
        self._items = {i.address: i for i in (items or [])}
        self._callbacks = []

    def items(self):
        return list(self._items.values())

    def on_change(self, callback):
        self._callbacks.append(callback)

        class Handle:
            def __init__(self, owner, cb):
                self._owner = owner
                self._cb = cb
                self._closed = False

            def close(self):
                if not self._closed:
                    self._closed = True
                    self._owner._callbacks.remove(self._cb)

        return Handle(self, callback)

    async def fire(self):
        for cb in list(self._callbacks):
            await cb()

    def set(self, items):
        self._items = {i.address: i for i in items}


def _client(*addrs):
    return FakeClient([FakeItem(a, FakeDeclaration("http://x/" + a, a)) for a in addrs])


@pytest.mark.asyncio
async def test_adopts_live_views_onto_the_desktop():
    model = ScreenModel()
    rec = Recorder()
    client = _client("cell/a/webview", "cell/b/webview")
    bridge = WebViewBridge(model, client, emit=rec.broadcast)
    await bridge.start()

    assert len(model.desktop_items()) == 2
    assert set(model.adopted().keys()) == {"cell/a/webview", "cell/b/webview"}
    # The two addresses must map to distinct item ids.
    assert len(set(model.adopted().values())) == 2


@pytest.mark.asyncio
async def test_releases_views_that_went_away():
    model = ScreenModel()
    rec = Recorder()
    client = _client("cell/a/webview", "cell/b/webview")
    bridge = WebViewBridge(model, client, emit=rec.broadcast)
    await bridge.start()

    client.set([FakeItem("cell/a/webview", FakeDeclaration("http://x/a", "a"))])
    await client.fire()

    assert list(model.adopted().keys()) == ["cell/a/webview"]


@pytest.mark.asyncio
async def test_a_destroyed_view_is_not_resurrected():
    model = ScreenModel()
    rec = Recorder()
    client = _client("cell/a/webview")
    bridge = WebViewBridge(model, client, emit=rec.broadcast)
    await bridge.start()

    item_id = model.desktop_items()[0]
    model.destroy(item_id)

    # A change event fires: the service is still live, but the tombstone holds.
    await client.fire()
    assert model.adopted() == {}
    assert model.desktop_items() == []


@pytest.mark.asyncio
async def test_an_adopted_view_arrives_with_its_face():
    """A view that already has something to say says it at birth.

    Otherwise the tile shows a bare title until the node happens to speak again —
    and a node that speaks once, at start, would never grow a face at all.
    """
    model = ScreenModel()
    rec = Recorder()
    item = FakeItem("cell/a/webview", FakeDeclaration("http://x/a", "a"))
    item.state = FakeState(face_line="2 cards waiting", face_art="http://x/a/still.png")
    bridge = WebViewBridge(model, FakeClient([item]), emit=rec.broadcast)

    await bridge.start()

    opened = rec.of("open")[-1]["item"]
    assert opened["face_line"] == "2 cards waiting"
    assert opened["face_art"] == "http://x/a/still.png"


@pytest.mark.asyncio
async def test_a_face_change_sends_a_face_frame_not_the_whole_arrangement():
    """A face moves on its own clock, so it travels as its own delta.

    Re-sending the arrangement to deliver one line would make every node's idle
    chatter repaint the stage — and the stage is what someone is looking at.
    """
    model = ScreenModel()
    rec = Recorder()
    item = FakeItem("cell/a/webview", FakeDeclaration("http://x/a", "a"))
    client = FakeClient([item])
    bridge = WebViewBridge(model, client, emit=rec.broadcast)
    await bridge.start()
    rec.frames.clear()

    item.state = FakeState(face_line="streaming")
    await client.fire()

    assert rec.types() == ["face"]
    assert rec.frames[-1]["face_line"] == "streaming"


@pytest.mark.asyncio
async def test_the_same_face_again_sends_nothing():
    """Idle chatter is not a change: re-asserting one face is not news."""
    model = ScreenModel()
    rec = Recorder()
    item = FakeItem("cell/a/webview", FakeDeclaration("http://x/a", "a"))
    item.state = FakeState(face_line="idle")
    client = FakeClient([item])
    bridge = WebViewBridge(model, client, emit=rec.broadcast)
    await bridge.start()
    rec.frames.clear()

    await client.fire()

    assert rec.frames == []
