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


def test_parse_ops_reads_the_plain_text_form():
    """一对是 old||new, 多对用一行 ||| 隔开 —— 正文里没有第二种语言要转义."""
    assert parse_ops("a||b") == [("a", "b")]
    assert parse_ops("a||b\n|||\nc||d") == [("a", "b"), ("c", "d")]
    assert parse_ops("x||") == [("x", "")], "删掉一段就是新的那半空着"
    assert parse_ops("a\\b||c") == [("a\\b", "c")], "反斜杠是字面量, 不是转义"


def test_parse_ops_names_the_pair_it_cannot_split():
    with pytest.raises(ValueError, match="pair 2"):
        parse_ops("a||b\n|||\nc||d||e")
    with pytest.raises(ValueError, match="pair 1"):
        parse_ops("no separator here")


def test_parse_ops_takes_a_custom_separator():
    assert parse_ops("a=>b", sep="=>") == [("a", "b")]
    assert parse_ops("a||b=>c", sep="=>") == [("a||b", "c")]


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
            "rewind", "export", "threads", "history", "layout", "point", "format",
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
        # The thread stays open, and the notice says which version is on disk —
        # "I already exported" is the belief that goes stale otherwise.
        t = store.get("t")
        assert t.state == "live"
        assert t.exported_at == t.version
        assert not t.unsaved
        assert "wrote v" in _signal_text(rec)


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
        await _until(lambda: store.get("t").exported_at == store.get("t").version)
        assert target.read_text() == "base\nx\n"
        assert store.get("t").state == "live", "auto-export does not close the thread"
        assert "wrote v" in _signal_text(rec)


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


@pytest.mark.asyncio
async def test_layout_is_readable_and_settable(store):
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        # reading it back says who chose: the default is stated, never implied
        out = await runtime.execute_command("layout", args=())
        assert "page" in out and "default" in out

        out = await runtime.execute_command("layout", args=("split",))
        assert "split" in out and "model" in out
        assert rec.layout_state() == ("split", "model")


@pytest.mark.asyncio
async def test_a_card_carries_the_motive_and_the_mechanics(store):
    """Two readers, one card.

    The mechanics are the node's to fill in (which region, how many lines); the
    motive is only ever the model's. A card with just the mechanics tells the
    human nothing about whether to open it.
    """
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",), kwargs={"label": "doc"})
        await runtime.execute_command("write", args=("t", chunks(["alpha\n", "beta\n"])))

        await runtime.execute_command(
            "read", args=("t",), kwargs={"label": "confirm the header"}
        )
        card = rec.of("action")[-1]
        assert card["kind"] == "read"
        assert card["label"] == "confirm the header"
        assert card["fact"] == "all · 2 lines"

        # no motive: the title falls back to the mechanics, and the fact still
        # says what happened rather than going blank
        await runtime.execute_command("read", args=("t",))
        card = rec.of("action")[-1]
        assert card["label"] == "read all"
        assert card["fact"] == "all · 2 lines"

        # an edit's fact is the size of the change, known only once it lands
        edit = [f for f in rec.of("action") if f["kind"] == "write"][-1]
        assert edit["fact"] == "+2 lines"


@pytest.mark.asyncio
async def test_a_big_file_reads_as_metadata_not_text(store):
    """Past the threshold, the first answer is the file's shape, not its body.

    A working copy that is only held costs bytes; one that is read costs the
    model's whole frame, and a frame blown by a huge log does not come back. So
    read() reports size / lines / endings and the three moves that work.
    """
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command(
            "append", args=("t", chunks(["line %d\n" % i for i in range(2500)]))
        )
        out = await runtime.execute_command("read", args=("t",))
        assert out.startswith("[file_editor]"), out[:80]
        assert "2500 lines" in out
        assert "find(" in out and 'read(region="a-b")' in out
        assert "line 2499" not in out, "the body must not be pasted into the frame"


@pytest.mark.asyncio
async def test_a_region_read_on_a_big_file_is_capped(store):
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command(
            "append", args=("t", chunks(["line %d\n" % i for i in range(2500)]))
        )
        out = await runtime.execute_command("read", args=("t",), kwargs={"region": "1-900"})
        assert "region capped" in out
        assert "line 0" in out and "line 899" not in out


@pytest.mark.asyncio
async def test_find_returns_line_numbers_not_excerpts(store):
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command(
            "append", args=("t", chunks(["alpha\n", "beta needle\n", "gamma\n"]))
        )
        out = await runtime.execute_command("find", args=("t", "needle"))
        assert "2:" in out and "beta needle" in out
        assert "1 match" in out
        assert rec.of("action")[-1]["kind"] == "find"


@pytest.mark.asyncio
async def test_reading_a_region_moves_the_cursor_onto_it(store):
    """Reading is also pointing: the human should not have to guess which part
    of the file the sentence is about."""
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command(
            "append", args=("t", chunks([f"line {i}\n" for i in range(1, 40)]))
        )
        await runtime.execute_command("read", args=("t", "10-14"))
        assert rec.cursor_of("t")["start"] == 10 and rec.cursor_of("t")["end"] == 14
        assert rec.cursor_of("t")["by"] == "model"


@pytest.mark.asyncio
async def test_format_reads_back_and_refuses_an_unknown_renderer(store):
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        assert "auto" in await runtime.execute_command("format", args=("",))
        assert "markdown" in await runtime.execute_command("format", args=("markdown",))
        assert rec.fmt == ("markdown", "model")
        with pytest.raises(Exception):
            await runtime.execute_command("format", args=("html",))


@pytest.mark.asyncio
async def test_point_moves_the_cursor_without_reading(store):
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command("append", args=("t", chunks(["a\n", "b\n", "c\n"])))
        before = len(rec.of("action"))
        out = await runtime.execute_command("point", args=("t", "2-3"))
        assert "2-3" in out
        mark = rec.cursor_of("t")
        assert (mark["start"], mark["end"]) == (2, 3)
        # a point is a gesture, not a record — it leaves no card behind
        assert len(rec.of("action")) == before


@pytest.mark.asyncio
async def test_point_with_no_thread_reads_the_cursor_back(store):
    """The other half of the contract: where the *human* scrolled is a value I
    can pull, not a signal that has to win my attention to arrive."""
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command("append", args=("t", chunks(["a\n", "b\n", "c\n"])))
        empty = await runtime.execute_command("point", args=("",))
        assert "unset" in empty
        await rec.set_cursor("t", 2, 3, by="model")
        await rec.set_view_at("t", 40)
        await rec.set_selection("t", 2, 3, "b\nc")
        out = await runtime.execute_command("point", args=("",))
        assert "2-3" in out and "line 40" in out
        assert "selected" in out and "b" in out


@pytest.mark.asyncio
async def test_find_puts_the_cursor_on_the_first_hit(store):
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command(
            "append", args=("t", chunks(["alpha\n", "beta needle\n", "gamma\n"]))
        )
        await runtime.execute_command("find", args=("t", "needle"))
        mark = rec.cursor_of("t")
        assert (mark["start"], mark["end"]) == (2, 2)


@pytest.mark.asyncio
async def test_edit_replaces_the_named_lines_without_quoting_them(store):
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command(
            "append", args=("t", chunks(["one\n", "two\n", "three\n"]))
        )
        await runtime.execute_command(
            "edit", args=("t", "2-2"), kwargs={"text__": "TWO\n", "label": "rename two"}
        )
        assert store.get("t").content == "one\nTWO\nthree\n"
        assert rec.of("action")[-1]["kind"] == "edit"


@pytest.mark.asyncio
async def test_orphans_lists_unclaimed_copies_and_discard_clears_them(store, tmp_path):
    """A leftover working copy is pulled on demand, and can be acted on.

    Previously the names were pushed into the notice every turn and there was no
    verb for them — state I could read and could not touch. Now the notice carries
    a count, ``orphans()`` names them, and ``discard`` removes the file and the
    human's chip in one move.
    """
    drafts = tmp_path / "drafts"
    drafts.mkdir(parents=True)
    (drafts / "old-probe.txt").write_text("stale\n", encoding="utf-8")
    store.sweep()

    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        listed = await runtime.execute_command("orphans")
        assert "old-probe.txt" in listed and "bytes" in listed

        out = await runtime.execute_command("discard", args=("old-probe.txt",))
        assert "discarded" in out

    assert not (drafts / "old-probe.txt").exists()
    assert store.recoverable() == []
    assert rec.of("threads")[-1]["drafts"] == [], "the human's chip follows the delete"


@pytest.mark.asyncio
async def test_discard_refuses_a_live_threads_working_copy(store):
    """A live thread's draft is that thread's text — deleting it would cut the
    crash net under an edit in progress."""
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",), kwargs={"label": "doc"})
        name = store.get("t").draft
        with pytest.raises(Exception) as err:
            await runtime.execute_command("discard", args=(name,))
    assert "live thread" in str(err.value)
    assert (store.drafts_dir / name).exists()


@pytest.mark.asyncio
async def test_the_notice_carries_a_count_not_the_names(store, tmp_path):
    """Eleven stale names pushed every turn bury the one that matters; a count is
    the part that can change a decision."""
    drafts = tmp_path / "drafts"
    drafts.mkdir(parents=True)
    (drafts / "a.txt").write_text("x", encoding="utf-8")
    (drafts / "b.txt").write_text("y", encoding="utf-8")
    store.sweep()

    chan, _ = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",), kwargs={"label": "doc"})
        await runtime.refresh_metas()
        notice = runtime.self_meta().named_notices.get("drafts", "")
    assert "2 unclaimed" in notice
    assert "a.txt" not in notice


@pytest.mark.asyncio
async def test_a_trusted_directory_exports_without_asking(store, tmp_path):
    """Trust is the human's grant, and it is scoped: the same write that asked a
    moment ago goes straight through once the directory it lands in is trusted.

    There is deliberately no command to grant it — the model can read `trusted`
    and point at the switch, but the switch is the human's hand.
    """
    work = tmp_path / "work"
    work.mkdir()
    target = work / "out.md"
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command("append", args=("t", chunks(["final\n"])))

        first = await runtime.execute_command("export", args=("t",), kwargs={"path": str(target)})
        assert "awaiting approval" in first
        store.settle("t", store.get("t").actions[-1].n, "accept")
        await _until(target.exists)

        # The human trusts the directory on their own surface.
        store.set_dir_trust(str(work), True)
        await runtime.execute_command("append", args=("t", chunks(["more\n"])))
        out = await runtime.execute_command("export", args=("t",), kwargs={"path": str(target)})
        assert "auto-exporting" in out and "directory is trusted" in out
        await _until(lambda: target.read_text() == "final\nmore\n")

    assert target.read_text() == "final\nmore\n"


def test_directory_trust_covers_files_below_and_stops_at_the_edge(store, tmp_path):
    work = tmp_path / "work"
    other = tmp_path / "other"
    work.mkdir()
    other.mkdir()
    store.set_dir_trust(str(work), True)
    assert store.is_trusted(str(work / "a.md"))
    assert store.is_trusted(str(work / "deep" / "b.md"))
    assert not store.is_trusted(str(other / "c.md"))
    store.set_dir_trust(str(work), False)
    assert not store.is_trusted(str(work / "a.md"))


@pytest.mark.asyncio
async def test_the_notice_reports_which_directories_are_trusted(store, tmp_path):
    """I have to be able to read whether the gate is open — otherwise "it wrote
    by itself" and "somebody approved it" look the same from here."""
    work = tmp_path / "work"
    work.mkdir()
    store.set_dir_trust(str(work), True)
    chan, _ = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        notice = runtime.self_meta().named_notices.get("trust", "")
    assert str(work) in notice


@pytest.mark.asyncio
async def test_preview_pushes_a_draft_and_renderers_lists_the_library(store):
    """Two halves of the same idea: the library is discovered, and a draft is
    pushed. Neither writes a file — that is what `export`/the file path is for."""
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        out = await runtime.execute_command(
            "preview", args=("t", "return '<b>x</b>'", "只看标题")
        )
        assert "只看标题" in out and "#1" in out
        assert rec.previews[0]["thread"] == "t"
        assert rec.previews[0]["label"] == "只看标题"

        listing = await runtime.execute_command("renderers")
        assert "raw" in listing and "markdown" in listing


@pytest.mark.asyncio
async def test_a_write_waits_while_the_human_holds_the_pen(store):
    """The refusal is the state, in one sentence — not a bare 'busy'."""
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t",))
        await runtime.execute_command("append", args=("t", chunks(["a\n", "b\n"])))
        rec.locks["t"] = {"start": 1, "end": 2, "since": "12:20:00"}
        with pytest.raises(Exception) as err:
            await runtime.execute_command("edit", args=("t", "1-2", "x\n"))
        assert "editing lines 1–2" in str(err.value)
        rec.locks.clear()
        assert "v" in await runtime.execute_command("edit", args=("t", "1-2", "x\n"))


@pytest.mark.asyncio
async def test_a_file_changed_underneath_us_is_visible_and_reloadable(store, tmp_path):
    """No lock: we let the file move, we just refuse to be unaware of it."""
    doc = tmp_path / "note.md"
    doc.write_text("one\n", encoding="utf-8")
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t", str(doc), "note"))
        thread = store.get("t")
        assert store.disk_drift(thread) == ""

        doc.write_text("one\ntwo\n", encoding="utf-8")   # somebody else edited it
        drift = store.disk_drift(store.get("t"))
        assert "changed on disk" in drift, "the working copy is not the file any more"

        out = await runtime.execute_command("reload", args=("t",))
        assert "reloaded" in out
        assert store.get("t").content == "one\ntwo\n"
        assert store.disk_drift(store.get("t")) == "", "reload settles the drift"


@pytest.mark.asyncio
async def test_diff_against_the_file_is_an_action(store, tmp_path):
    """Comparing is a decision someone made — so it lands on the line."""
    doc = tmp_path / "note.md"
    doc.write_text("one\ntwo\n", encoding="utf-8")
    chan, rec = _channel(store)
    async with chan.bootstrap() as runtime:
        await runtime.execute_command("open", args=("t", str(doc), "note"))
        out = await runtime.execute_command("diff", args=("t",))
        assert "identical" in out, "nothing has changed yet"

        await runtime.execute_command("edit", args=("t", "2-2", "TWO\n"))
        out = await runtime.execute_command("diff", args=("t",))
        assert "-two" in out and "+TWO" in out
        kinds = [a.kind for a in store.get("t").actions]
        assert kinds.count("diff") == 2, "a diff is recorded, not just printed"
