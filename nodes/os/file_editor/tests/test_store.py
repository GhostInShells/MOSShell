import asyncio
import time

import pytest

from ghoshell_file_editor.store import DocStore


@pytest.fixture
def store(tmp_path):
    return DocStore(drafts_dir=tmp_path / "drafts", root=tmp_path)


def _append(store, tid, segment, label="append"):
    action = store.begin(tid, "append", label)
    store.feed(tid, action.n, segment)
    return store.tail(tid, action.n)


def test_open_blank_thread(store):
    t = store.open("t", "doc")
    assert t.content == ""
    assert t.version == 0
    assert (store.drafts_dir / t.draft).read_text() == ""


def test_open_from_file(store, tmp_path):
    f = tmp_path / "a.md"
    f.write_text("# hi\n")
    t = store.open("t", "doc", path=str(f))
    assert t.content == "# hi\n"
    assert t.path == str(f.resolve())


def test_open_refuses_outside_the_allowed_roots(store):
    with pytest.raises(ValueError, match="escapes"):
        store.open("t", path="/etc/passwd")


def test_resolve_target_allows_the_system_tempdir(store):
    import os
    import tempfile

    target = os.path.join(tempfile.gettempdir(), "fe-scratch.txt")
    assert store.resolve_target(target)  # the temp dir is a trusted throwaway root


def test_open_adopts_a_draft(store, tmp_path):
    draft = store.drafts_dir
    draft.mkdir(parents=True)
    (draft / "orphan.txt").write_text("recovered\n")
    t = store.open("t", "doc", draft="orphan.txt")
    assert t.content == "recovered\n"
    assert t.draft == "orphan.txt"


def test_open_refuses_duplicate_id(store):
    store.open("t")
    with pytest.raises(ValueError, match="already exists"):
        store.open("t")


def test_open_refuses_a_second_live_thread_on_a_path(store, tmp_path):
    f = tmp_path / "a.md"
    f.write_text("x\n")
    store.open("t1", path=str(f))
    with pytest.raises(ValueError, match="already editing"):
        store.open("t2", path=str(f))


def test_open_refuses_beyond_the_cap(store):
    for i in range(16):
        store.open(f"t{i}")
    with pytest.raises(ValueError, match="export or close"):
        store.open("overflow")


def test_stream_append_lands_and_mirrors(store):
    store.open("t", "doc", path=None)
    _append(store, "t", "# hi\n")
    t = store.get("t")
    assert t.content == "# hi\n"
    assert t.version == 1
    assert (store.drafts_dir / t.draft).read_text() == "# hi\n"


def test_write_replaces_whole_content(store):
    store.open("t", "doc")
    a = store.begin("t", "write", "rewrite")
    store.feed("t", a.n, "all new\n")
    store.tail("t", a.n)
    assert store.get("t").content == "all new\n"


def test_replace_applies_ops_in_order_one_card_each(store):
    store.open("t", "doc")
    _append(store, "t", "hello world\n")
    actions = store.replace("t", [("hello", "goodbye"), ("world", "friend")], "fix")
    assert len(actions) == 2
    assert actions[0].effect.after == "goodbye world\n"
    assert actions[1].effect.after == "goodbye friend\n"
    assert store.get("t").content == "goodbye friend\n"


def test_replace_refuses_ambiguous_old_str(store):
    store.open("t", "doc")
    _append(store, "t", "a b a\n")
    with pytest.raises(ValueError, match="unique"):
        store.replace("t", [("a", "x")])


def test_rewind_to_base_and_to_version(store):
    store.open("t", "doc")
    _append(store, "t", "one\n")
    _append(store, "t", "two\n")
    store.rewind("t", 1)
    assert store.get("t").content == "one\n"
    assert store.get("t").version == 3
    store.rewind("t", 0)
    assert store.get("t").content == ""


def test_rewind_refuses_a_read_as_target(store):
    store.open("t", "doc")
    store.record("t", "read", "read all")
    with pytest.raises(ValueError, match="moved no text"):
        store.rewind("t", 1)


def test_read_record_has_no_effect(store):
    store.open("t", "doc")
    _append(store, "t", "x\n")
    a = store.record("t", "read", "read all", text="x\n")
    assert a.effect is None
    assert store.get("t").version == 1  # a read does not move the version


def test_close_abandons_the_working_copy(store):
    store.open("t", "doc")
    _append(store, "t", "x\n")
    t = store.get("t")
    draft = store.drafts_dir / t.draft
    assert draft.exists()
    store.close("t")
    assert t.state == "closed"
    assert t.content == "x\n"  # the line stays traceable; reclamation is separate
    assert not draft.exists()


def test_export_keeps_the_line_for_traceability(store):
    store.open("t", "doc")
    _append(store, "t", "final\n")
    t = store.get("t")
    draft = store.drafts_dir / t.draft
    store.mark_exported("t", "/tmp/final.txt")
    assert t.state == "exported"
    assert t.exported_to == "/tmp/final.txt"
    assert t.content == "final\n"
    assert t.version == 1
    assert not draft.exists()
    # the line is intact — the human can still trace what led to the export
    assert t.actions[0].effect is not None
    assert t.actions[0].effect.after == "final\n"


def test_trim_drops_the_oldest_ended_thread(store):
    for i in range(9):
        store.open(f"t{i}", f"d{i}")
        store.mark_exported(f"t{i}", f"/tmp/x{i}.md")
    ids = [t.id for t in store.threads()]
    assert "t0" not in ids, "the oldest ended thread is reclaimed"
    assert "t1" in ids
    assert len(store.threads()) == 8


def test_auto_requires_a_target(store):
    store.open("t", "doc")  # blank, no path
    with pytest.raises(ValueError, match="no path"):
        store.set_thread_auto("t", True)


def test_auto_on_a_path_thread_toggles(store, tmp_path):
    f = tmp_path / "a.md"
    f.write_text("x\n")
    store.open("t", "doc", path=str(f))
    store.set_thread_auto("t", True)
    assert store.get("t").auto is True
    store.set_thread_auto("t", False)
    assert store.get("t").auto is False


def test_verdict_waiter_rendezvous_and_is_not_lost(store):
    store.open("t", "doc")
    action = store.record("t", "export", "e", payload="/tmp/x", state="awaiting")

    async def go():
        # the verdict lands before the channel has parked — it must not be lost
        store.settle("t", action.n, "accept")
        future = store.waiter("t", action.n)
        assert await future == "accept"

    asyncio.run(go())


def test_finish_export_moves_state(store):
    store.open("t", "doc")
    a = store.record("t", "export", "e", payload="/tmp/x", state="awaiting")
    assert store.finish_export("t", a.n, "written").state == "written"


def test_orphans_and_sweep(store):
    drafts = store.drafts_dir
    drafts.mkdir(parents=True)
    young = drafts / "young.txt"
    young.write_text("recent\n")
    old = drafts / "old.txt"
    old.write_text("stale\n")
    old_time = time.time() - 8 * 24 * 3600
    import os
    os.utime(old, (old_time, old_time))

    kept = store.sweep()
    assert kept == ["young.txt"]
    assert not old.exists()
    assert store.recoverable() == ["young.txt"]
