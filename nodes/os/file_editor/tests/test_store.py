import pytest

from ghoshell_file_editor.store import ThreadStore


@pytest.fixture
def store():
    return ThreadStore()


def _write(store, tid, payload, n=None, **kw):
    return store.append_action(tid, "g", "write", "write", payload, n=n, **kw)


def test_confirm_write_advances_version_chain(store):
    store.open_thread("t", "doc", path="/tmp/doc.md", base_content="one\n")
    _write(store, "t", "two\n")
    v = store.confirm("t", 1)

    assert v is not None
    assert v.content == "two\n"
    assert v.parent == "t:v0"
    assert v.effect.before == "one\n"
    assert v.effect.after == "two\n"
    assert "one" in v.effect.diff and "two" in v.effect.diff
    assert store.get_thread("t").head == "t:v1"


def test_reference_confirm_does_not_advance(store):
    store.open_thread("t", "doc", base_content="one\n")
    store.append_action("t", "g", "reference", "show", "1,3")
    v = store.confirm("t", 1)

    assert v is None
    assert store.get_thread("t").head == "t:v0"
    assert store.get_thread("t").versions[0].content == "one\n"


def test_reject_cascades_to_later_actions(store):
    store.open_thread("t", "doc", base_content="base\n")
    _write(store, "t", "a\n")
    _write(store, "t", "b\n")
    _write(store, "t", "c\n")

    cascaded = store.reject("t", 2)

    assert [s.n for s in cascaded] == [2, 3]
    assert store.get_action("t", 2).verdict == "rejected"
    assert store.get_action("t", 3).verdict == "rejected"
    assert store.get_action("t", 1).verdict == "pending"


def test_rewind_restores_old_content(store):
    store.open_thread("t", "doc", base_content="v0\n")
    _write(store, "t", "v1\n")
    store.confirm("t", 1)
    _write(store, "t", "v2\n")
    store.confirm("t", 2)

    store.append_action("t", "g", "rewind", "rewind", "t:v0")
    v = store.confirm("t", 3)

    assert v.content == "v0\n"
    assert v.effect.before == "v2\n"
    assert store.get_thread("t").head == "t:v3"


def test_replay_rebuilds_state(store, tmp_path):
    log = tmp_path / "t.jsonl"
    s = ThreadStore(log_path=log)
    s.open_thread("t", "doc", path="/tmp/doc.md", base_content="base\n")
    s.append_action("t", "g", "write", "write", "v1\n")
    s.confirm("t", 1)
    s.append_action("t", "u", "write", "write", "v2\n")
    s.reject("t", 2)

    rebuilt = ThreadStore.replay(log)

    assert rebuilt.get_thread("t").head == "t:v1"
    assert rebuilt.get_thread("t").head_version.content == "v1\n"
    assert rebuilt.get_action("t", 1).verdict == "confirmed"
    assert rebuilt.get_action("t", 2).verdict == "rejected"
    assert rebuilt.get_action("t", 2).author == "u"


def test_replay_survives_missing_log(store, tmp_path):
    assert ThreadStore.replay(tmp_path / "nope.jsonl").get_thread("t") is None
