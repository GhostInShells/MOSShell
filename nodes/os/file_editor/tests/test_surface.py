from pathlib import Path

import pytest

from ghoshell_file_editor.store import ThreadStore
from ghoshell_file_editor.surface import FileEditorSurface


@pytest.fixture
def store():
    return ThreadStore()


class _Sink:
    def __init__(self):
        self.signals = []

    def __call__(self, signal):
        self.signals.append(signal)


def _surface(store, *, toggles=None):
    sink = _Sink()
    toggles = toggles if toggles is not None else []
    surf = FileEditorSurface(
        store,
        send_signal=sink,
        self_identity="fe_ab12cd",
        on_toggle=lambda v: toggles.append(v),
        host="127.0.0.1",
        port=0,
        html_path=Path("index.html"),
    )
    return surf, sink, toggles


@pytest.mark.asyncio
async def test_confirm_mutates_store_and_signals_notify(store):
    store.open_thread("t", "doc", base_content="one\n")
    store.append_action("t", "g", "write", "write", "two\n")

    surf, sink, _ = _surface(store)
    await surf._confirm("t", 1)

    assert store.get_action("t", 1).verdict == "confirmed"
    assert store.get_action("t", 1).verdict_by == "u"

    assert len(sink.signals) == 1
    sig = sink.signals[0]
    assert sig.name == "notify"
    msg = sig.messages[0]
    assert msg.meta.tag == "file_editor"
    assert msg.meta.name == "fe_ab12cd"
    assert msg.meta.attributes == {"thread": "t", "seq": "1"}


@pytest.mark.asyncio
async def test_reject_mutates_store_and_signals_notify(store):
    store.open_thread("t", "doc", base_content="one\n")
    store.append_action("t", "g", "write", "write", "two\n")

    surf, sink, _ = _surface(store)
    await surf._reject("t", 1)

    assert store.get_action("t", 1).verdict == "rejected"
    assert sink.signals[0].name == "notify"


@pytest.mark.asyncio
async def test_reply_mutates_store_and_signals_silent(store):
    store.open_thread("t", "doc", base_content="one\n")
    store.append_action("t", "g", "write", "write", "two\n")

    surf, sink, _ = _surface(store)
    await surf._reply("t", 1, "why?", "effect")

    assert store.get_action("t", 1).replies[0].text == "why?"
    assert store.get_action("t", 1).verdict == "pending"
    assert sink.signals[0].name == "aside"


@pytest.mark.asyncio
async def test_toggle_is_passed_through(store):
    surf, _, toggles = _surface(store)
    surf._on_toggle(False)
    assert toggles == [False]
