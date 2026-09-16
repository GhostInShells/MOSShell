import pytest

from ghoshell_file_editor.channel import build_file_editor_channel, stream_write
from ghoshell_file_editor.store import ThreadStore


class _Recorder:
    def __init__(self):
        self.frames = []

    async def broadcast(self, frame):
        self.frames.append(frame)


def _chunks(parts):
    async def gen():
        for p in parts:
            yield p
    return gen()


@pytest.fixture
def store():
    return ThreadStore()


@pytest.mark.asyncio
async def test_channel_exposes_the_command_set(store):
    chan = build_file_editor_channel(store)
    async with chan.bootstrap() as runtime:
        for name in (
            "open", "write", "rewind", "reference", "export",
            "confirm", "reject", "reply", "threads", "thread", "history",
        ):
            assert runtime.get_command(name) is not None, name


@pytest.mark.asyncio
async def test_disabled_gate_flips_command_availability(store):
    gate = [True]
    chan = build_file_editor_channel(store, enabled=lambda: gate[0])
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert runtime.get_command("open").meta().available is True

        gate[0] = False
        await runtime.refresh_metas()
        assert runtime.get_command("open").meta().available is False


@pytest.mark.asyncio
async def test_open_then_thread_query(store):
    chan = build_file_editor_channel(store)
    async with chan.bootstrap() as runtime:
        result = await runtime.execute_command("open", args=("t",), kwargs={"label": "doc"})
        assert "t" in result

        out = await runtime.execute_command("thread", args=("t",))
        assert "doc" in out
        assert "0 actions" in out


@pytest.mark.asyncio
async def test_confirm_and_history(store):
    store.open_thread("t", "doc", base_content="one\n")
    store.append_action("t", "g", "write", "write", "two\n")
    chan = build_file_editor_channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("confirm", args=("t", 1))

        out = await runtime.execute_command("thread", args=("t",))
        assert "✓" in out
        assert "head: 1" in out

        hist = await runtime.execute_command("history", args=("t",))
        assert "two" in hist


@pytest.mark.asyncio
async def test_reply_attaches_but_decides_nothing(store):
    store.open_thread("t", "doc", base_content="one\n")
    store.append_action("t", "g", "write", "write", "two\n")
    chan = build_file_editor_channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("reply", args=("t", 1, "why?"), kwargs={"anchor": "effect"})

        action = store.get_action("t", 1)
        assert action.verdict == "pending"
        assert action.replies[0].text == "why?"


@pytest.mark.asyncio
async def test_stream_write_broadcasts_head_deltas_full(store):
    rec = _Recorder()
    store.open_thread("t", "doc", base_content="base\n")

    out = await stream_write(store, rec.broadcast, "t", _chunks(["hello ", "world\n"]), "hi")

    types = [f["type"] for f in rec.frames]
    assert types == ["action.head", "action.delta", "action.delta", "action.full"]
    assert rec.frames[0]["seq"] == 1
    assert rec.frames[-1]["effect"]["before"] == "base\n"
    assert rec.frames[-1]["effect"]["after"] == "hello world\n"
    assert "chars" in out
