import pytest

from ghoshell_file_editor.store import ThreadStore


@pytest.fixture
def store():
    return ThreadStore()


def _open(store, tid="t", base="base\n", **kw):
    return store.open_thread(tid, "doc", base_content=base, **kw)


def _write(store, payload, tid="t", author="g"):
    return store.append_action(tid, author, "write", "write", payload)


def test_effect_exists_before_any_verdict(store):
    # The whole point of computing effects at append time: the diff is
    # renderable while the human is still deciding.
    _open(store, base="one\n")
    _write(store, "two\n")

    action = store.get_action("t", 1)
    assert action.verdict == "pending"
    assert action.effect.before == "one\n"
    assert action.effect.after == "two\n"


def test_confirm_moves_the_head_without_computing_anything(store):
    _open(store, base="one\n")
    _write(store, "two\n")

    flipped = store.confirm("t", 1, by="u")

    assert [a.seq.n for a in flipped] == [1]
    assert flipped[0].verdict_by == "u"
    assert store.get_thread("t").head.seq.n == 1
    assert store.content("t") == "two\n"


def test_confirm_extends_the_confirmed_prefix(store):
    _open(store, base="base\n")
    _write(store, "a\n")
    _write(store, "b\n")
    _write(store, "c\n")

    flipped = store.confirm("t", 2, by="u")

    assert [a.seq.n for a in flipped] == [1, 2]
    assert store.get_action("t", 3).verdict == "pending"
    assert store.get_thread("t").head.seq.n == 2
    assert store.content("t") == "b\n"


def test_later_action_diffs_against_the_pending_tail(store):
    _open(store, base="zero\n")
    _write(store, "one\n")
    _write(store, "two\n")

    # Nothing is confirmed yet, and action 2 still diffs against action 1.
    assert store.get_action("t", 2).effect.before == "one\n"


def test_reference_confirm_changes_nothing(store):
    _open(store, base="one\n")
    store.append_action("t", "g", "reference", "show", "1,3")

    store.confirm("t", 1, by="u")

    assert store.get_thread("t").head is None
    assert store.content("t") == "one\n"
    assert store.get_thread("t").versions == []


def test_export_confirm_does_not_move_the_head(store):
    _open(store, base="one\n")
    _write(store, "two\n")
    store.append_action("t", "g", "export", "export to /tmp/o.md", "/tmp/o.md")
    store.confirm("t", 2, by="u")

    # The export's own effect carries the text that would land on disk, and the
    # line's content is unchanged by it.
    export = store.get_action("t", 2)
    assert export.effect.after == "two\n"
    assert store.get_thread("t").head.seq.n == 1
    assert [a.seq.n for a in store.get_thread("t").versions] == [1]


def test_reject_cascades_to_later_pending_actions(store):
    _open(store, base="base\n")
    _write(store, "a\n")
    _write(store, "b\n")
    _write(store, "c\n")

    cascaded = store.reject("t", 2, by="u")

    assert [s.n for s in cascaded] == [2, 3]
    assert store.get_action("t", 2).verdict == "rejected"
    assert store.get_action("t", 3).verdict_by == "u"
    assert store.get_action("t", 1).verdict == "pending"


def test_reject_refuses_a_confirmed_action(store):
    _open(store, base="base\n")
    _write(store, "a\n")
    store.confirm("t", 1, by="u")

    with pytest.raises(ValueError):
        store.reject("t", 1, by="u")


def test_rejected_actions_leave_the_tail_alone(store):
    _open(store, base="base\n")
    _write(store, "a\n")
    store.confirm("t", 1, by="u")
    _write(store, "b\n")
    store.reject("t", 2, by="u")

    _write(store, "c\n")

    assert store.get_action("t", 3).effect.before == "a\n"
    assert store.get_action("t", 3).effect.after == "c\n"


def test_rewind_appends_an_ordinary_action(store):
    _open(store, base="v0\n")
    _write(store, "v1\n")
    store.confirm("t", 1, by="u")
    _write(store, "v2\n")
    store.confirm("t", 2, by="u")

    store.append_action("t", "g", "rewind", "rewind to 1", "1")
    store.confirm("t", 3, by="u")

    thread = store.get_thread("t")
    assert thread.head.seq.n == 3
    assert store.content("t") == "v1\n"
    assert [a.seq.n for a in thread.versions] == [1, 2, 3]


def test_rewind_within_a_pending_burst(store):
    _open(store, base="v0\n")
    _write(store, "v1\n")
    _write(store, "v2\n")

    store.append_action("t", "g", "rewind", "back to 1", "1")
    store.confirm("t", 3, by="u")

    assert store.content("t") == "v1\n"


def test_reply_decides_nothing(store):
    _open(store, base="one\n")
    _write(store, "two\n")

    store.reply("t", 1, "u", "effect", diff="-one\n+two", text="why?")

    action = store.get_action("t", 1)
    assert action.verdict == "pending"
    assert action.replies[0].text == "why?"
    assert action.replies[0].anchor == "effect"


def test_open_refuses_a_second_line_on_the_same_path(store):
    _open(store, path="/tmp/doc.md")
    with pytest.raises(ValueError):
        store.open_thread("other", "doc", path="/tmp/doc.md")


def test_open_allows_many_pathless_lines(store):
    _open(store, tid="a")
    _open(store, tid="b")
    assert store.get_thread("b") is not None


def test_replay_rebuilds_state(store, tmp_path):
    log = tmp_path / "t.jsonl"
    s = ThreadStore(log_path=log)
    s.open_thread("t", "doc", path="/tmp/doc.md", base_content="base\n")
    s.append_action("t", "g", "write", "write", "v1\n")
    s.confirm("t", 1, by="u")
    s.append_action("t", "u", "write", "write", "v2\n")
    s.reject("t", 2, by="g")
    s.reply("t", 1, "u", "intent", text="ok")

    rebuilt = ThreadStore.replay(log)

    thread = rebuilt.get_thread("t")
    assert thread.head.seq.n == 1
    assert rebuilt.content("t") == "v1\n"
    assert rebuilt.get_action("t", 1).verdict == "confirmed"
    assert rebuilt.get_action("t", 1).verdict_by == "u"
    assert rebuilt.get_action("t", 2).verdict == "rejected"
    assert rebuilt.get_action("t", 2).author == "u"
    assert rebuilt.get_action("t", 1).replies[0].text == "ok"
    # Effects are recomputed by replay, not read back from the log.
    assert rebuilt.get_action("t", 1).effect.before == "base\n"


def test_replay_survives_missing_log(store, tmp_path):
    assert ThreadStore.replay(tmp_path / "nope.jsonl").get_thread("t") is None


def test_open_action_is_streaming_and_invisible_to_state(store):
    _open(store, base="base\n")
    seq = store.open_action("t", "g", "write", "write")

    action = store.get_action("t", seq.n)
    assert action.state == "streaming"
    assert action.payload == ""
    assert action.effect is None
    assert store.get_thread("t").head is None
    assert store.get_thread("t").versions == []
    assert store.content("t") == "base\n"


def test_streaming_deltas_accumulate_then_tail_finalizes(store):
    _open(store, base="base\n")
    seq = store.open_action("t", "g", "write", "write")
    store.append_delta("t", seq.n, "hello ")
    store.append_delta("t", seq.n, "world\n")

    action = store.tail_action("t", seq.n)

    assert action.state == "tailed"
    assert action.payload == "hello world\n"
    assert action.effect.before == "base\n"
    assert action.effect.after == "hello world\n"
    # tailed but not yet confirmed → the head has not moved.
    assert store.get_thread("t").head is None
    store.confirm("t", seq.n, by="u")
    assert store.get_thread("t").head.seq.n == seq.n
    assert store.content("t") == "hello world\n"


def test_streaming_action_is_skipped_by_the_next_tail(store):
    _open(store, base="base\n")
    first = store.open_action("t", "g", "write", "write")
    store.append_delta("t", first.n, "one\n")
    store.tail_action("t", first.n)

    second = store.open_action("t", "g", "write", "write")
    store.append_delta("t", second.n, "two\n")
    store.tail_action("t", second.n)

    assert store.get_action("t", second.n).effect.before == "one\n"


def test_delta_and_tail_refuse_a_non_streaming_action(store):
    _open(store, base="base\n")
    _write(store, "a\n")

    with pytest.raises(ValueError):
        store.append_delta("t", 1, "x")
    with pytest.raises(ValueError):
        store.tail_action("t", 1)


def test_replay_reconstructs_a_streamed_action(store, tmp_path):
    log = tmp_path / "t.jsonl"
    s = ThreadStore(log_path=log)
    s.open_thread("t", "doc", base_content="base\n")
    seq = s.open_action("t", "g", "write", "write")
    s.append_delta("t", seq.n, "streamed\n")
    s.tail_action("t", seq.n)
    s.confirm("t", seq.n, by="u")

    rebuilt = ThreadStore.replay(log)

    assert rebuilt.get_action("t", 1).state == "tailed"
    assert rebuilt.get_action("t", 1).payload == "streamed\n"
    assert rebuilt.content("t") == "streamed\n"
