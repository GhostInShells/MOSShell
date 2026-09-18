import asyncio
import time

import pytest

from ghoshell_file_editor.channel import build_file_editor_channel, parse_ops
from ghoshell_file_editor.store import DocStore

from fakes import Recorder, chunks


@pytest.fixture
def store(tmp_path):
    return DocStore(drafts_dir=tmp_path / "drafts", root=tmp_path)


def _channel(store, rec=None, enabled=None):
    rec = rec or Recorder()
    chan = build_file_editor_channel(
        store, surface=rec, signaler=rec, enabled=enabled
    )
    return chan, rec


async def _until(predicate, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true")


def _signal_text(rec):
    return "\n".join(s.messages[0].to_content_string() for s in rec.signals)


def test_parse_ops_accepts_one_or_a_list():
    assert parse_ops('[{"old_str": "a", "new_str": "b"}]') == [("a", "b")]
    assert parse_ops('{"old_str": "a", "new_str": "b"}') == [("a", "b")]
    assert parse_ops('{"edits": [{"old_str": "a", "new_str": "b"}]}') == [("a", "b")]


def test_parse_ops_refuses_garbage():
    for raw in ("", "not json", "[]", '[{"old_str": "a"}]'):
        with pytest.raises(ValueError):
            parse_ops(raw)


@pytest.mark.asyncio
async def test_channel_exposes_the_command_set(store):
    chan, _ = _channel(store)
    async with chan.bootstrap() as runtime:
        for name in (
            "open", "close", "read", "write", "append", "str_replace",
            "rewind", "export", "threads", "history",
        ):
            assert runtime.get_command(name) is not None, name


@pytest.mark.asyncio
async def test_disabled_gate_flips_command_availability(store):
    gate = [True]
    chan, _ = _channel(store, enabled=lambda: gate[0])
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert runtime.get_command("open").meta().available is True
        gate[0] = False
        await runtime.refresh_metas()
        assert runtime.get_command("open").meta().available is False


@pytest.mark.asyncio
async def test_open_registers_a_thread_and_broadcasts_it(store):
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        out = await runtime.execute_command("open", args=("t",), kwargs={"label": "doc"})
        assert "t" in out
        threads = rec.of("threads")
        assert threads[-1]["threads"][0]["label"] == "doc"


@pytest.mark.asyncio
async def test_read_returns_immediately_and_leaves_a_card(store):
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command("append", args=("t", chunks(["hello\n"])))
        out = await runtime.execute_command("read", args=("t",))
        assert out == "hello\n"
        cards = rec.of("action")
        assert cards[-1]["kind"] == "read"
        assert cards[-1]["effect"] == "none — read only"


@pytest.mark.asyncio
async def test_write_and_append_land_in_memory(store):
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command("write", args=("t", chunks(["all\n"])))
        assert store.get("t").content == "all\n"
        await runtime.execute_command("append", args=("t", chunks(["more\n"])))
        assert store.get("t").content == "all\nmore\n"
        assert store.get("t").version == 2
        assert rec.types().count("action") >= 2


@pytest.mark.asyncio
async def test_str_replace_emits_one_card_per_op(store):
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command("append", args=("t", chunks(["a b\n"])))
        out = await runtime.execute_command(
            "str_replace",
            args=("t", '[{"old_str": "a", "new_str": "x"}, {"old_str": "b", "new_str": "y"}]'),
        )
        assert "2 card(s)" in out
        assert store.get("t").content == "x y\n"
        kinds = [f["kind"] for f in rec.of("action")]
        assert kinds.count("str_replace") == 2


@pytest.mark.asyncio
async def test_str_replace_rejects_bad_json(store):
    chan, _ = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        with pytest.raises(Exception):
            await runtime.execute_command("str_replace", args=("t", "not json"))


@pytest.mark.asyncio
async def test_export_returns_a_receipt_and_waits_for_approval(store, tmp_path):
    chan, rec = _channel(store)
    target = tmp_path / "out.md"
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command("append", args=("t", chunks(["final\n"])))
        out = await runtime.execute_command(
            "export", args=("t",), kwargs={"path": str(target)}
        )
        assert "awaiting approval" in out
        assert not target.exists(), "nothing lands before a verdict"
        exp = store.get("t").actions[-1]
        assert exp.state == "awaiting"

        store.settle("t", exp.n, "accept")
        await _until(target.exists)
        assert target.read_text() == "final\n"
        assert store.get("t").state == "exported"
        assert "exported" in _signal_text(rec)


@pytest.mark.asyncio
async def test_auto_export_writes_without_asking(store, tmp_path):
    target = tmp_path / "out.md"
    target.write_text("base\n")
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",), kwargs={"path": str(target)})
        await runtime.execute_command("append", args=("t", chunks(["x\n"])))
        store.set_thread_auto("t", True)
        out = await runtime.execute_command("export", args=("t",))
        assert "auto-exporting" in out
        await _until(lambda: store.get("t").state == "exported")
        assert target.read_text() == "base\nx\n"
        assert "exported" in _signal_text(rec)


@pytest.mark.asyncio
async def test_auto_does_not_cover_a_new_target(store, tmp_path):
    target = tmp_path / "out.md"
    target.write_text("base\n")
    other = tmp_path / "other.md"
    chan, _ = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",), kwargs={"path": str(target)})
        await runtime.execute_command("append", args=("t", chunks(["x\n"])))
        store.set_thread_auto("t", True)
        out = await runtime.execute_command(
            "export", args=("t",), kwargs={"path": str(other)}
        )
        assert "awaiting approval" in out  # a brand-new target still asks
        assert store.get("t").actions[-1].state == "awaiting"


@pytest.mark.asyncio
async def test_export_deny_marks_rejected(store):
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command("append", args=("t", chunks(["final\n"])))
        await runtime.execute_command("export", args=("t", "x.md"))
        exp = store.get("t").actions[-1]
        store.settle("t", exp.n, "deny")
        await _until(lambda: exp.state == "rejected")
        assert store.get("t").state == "live"
        assert "denied" in _signal_text(rec)


@pytest.mark.asyncio
async def test_history_is_an_index(store):
    chan, _ = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command("append", args=("t", chunks(["x\n"])))
        out = await runtime.execute_command("history", args=("t",))
        assert "v1" in out
        assert "append" in out


@pytest.mark.asyncio
async def test_notice_carries_the_current_version(store):
    chan, _ = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",), kwargs={"label": "readme"})
        await runtime.execute_command("append", args=("t", chunks(["x\n"])))
        await runtime.refresh_metas()
        notices = runtime.self_meta().named_notices
        assert "v1" in notices["thread_t"]
        assert "readme" in notices["thread_t"]
