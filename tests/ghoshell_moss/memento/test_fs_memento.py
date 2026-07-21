"""FsMemento golden tests — FORMAT v2 contract verification."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ghoshell_moss.memento.abc import (
    BranchRef,
    BranchWindow,
    CommitDetail,
    CommitNotFoundError,
    CommitRef,
    CommitView,
    EmptyStagingError,
    LineNotFoundError,
    MementoError,
    MementoHooks,
    MomentFrozenError,
    MomentNotInCommitError,
    MomentRecord,
    new_commit_id,
)
from ghoshell_moss.memento.fs_memento import (
    FsMemento,
    _read_lines,
    _resolve_staging,
    _y_m,
    new_filesystem_memento,
)


@pytest.fixture
def tmp_root() -> Path:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def memento(tmp_root) -> FsMemento:
    return new_filesystem_memento(tmp_root, "ghost.test")


class Collector(MementoHooks):
    def __init__(self):
        self.events: list[tuple[str, dict]] = []

    def on_record_staged(self, line: str, record: MomentRecord) -> None:
        self.events.append(("record_staged", {"line": line, "id": record.id}))

    def on_commit(self, line: str, view: CommitView) -> None:
        self.events.append(("commit", {"line": line, "id": view.id}))

    def on_reinterpreted(self, commit_id: str, view: CommitView) -> None:
        self.events.append(("reinterpreted", {"commit_id": commit_id}))

    def on_line_created(self, name: str, from_ref: BranchRef | None) -> None:
        self.events.append(("line_created", {"name": name}))

    def on_line_deleted(self, name: str) -> None:
        self.events.append(("line_deleted", {"name": name}))


# ── basic lifecycle ─────────────────────────────────────────────────────────


class TestBasicLifecycle:
    def test_create_line_and_record(self, memento):
        line = memento.create_line("main")
        assert line.name == "main"
        assert line.ref is None
        assert not line.readonly
        line.record(MomentRecord(id="m1", type="test/x", payload={"k": "v"}))
        assert len(line.staging()) == 1

    def test_record_overwrite_last_wins(self, memento):
        line = memento.create_line("main")
        line.record(MomentRecord(id="m1", type="t", payload={"v": 1}))
        line.record(MomentRecord(id="m1", type="t", payload={"v": 2}))
        assert line.staging()[0].payload["v"] == 2

    def test_commit_and_show(self, memento):
        line = memento.create_line("main")
        line.record(MomentRecord(id="m1", type="t", payload={"k": "hello"}))
        line.record(MomentRecord(id="m2", type="t", payload={"k": "world"}))
        view = line.commit(text="first", kind="semantic")
        assert view.commit.id.startswith("cmt_")
        assert line.staging() == []
        assert line.ref is not None
        assert line.ref.commit_id == view.id
        detail = memento.show(view.id)
        assert len(detail.moments) == 2
        assert detail.moments[0].id == "m1"

    def test_commit_empty_staging_raises(self, memento):
        line = memento.create_line("main")
        with pytest.raises(EmptyStagingError):
            line.commit(kind="semantic")

    def test_frozen_moment_raises(self, memento):
        line = memento.create_line("main")
        line.record(MomentRecord(id="m1", type="t", payload={}))
        line.commit(kind="semantic")
        with pytest.raises(MomentFrozenError):
            line.record(MomentRecord(id="m1", type="t", payload={}))

    def test_annotate(self, memento):
        line = memento.create_line("main")
        line.record(MomentRecord(id="m1", type="t", payload={}))
        view = line.commit(text="original", kind="semantic")
        v2 = memento.annotate(view.id, title="revised", body="new body")
        assert v2.summary() == "revised"
        with pytest.raises(CommitNotFoundError):
            memento.annotate("cmt_nonexistent")

    def test_notes(self, memento):
        line = memento.create_line("main")
        line.record(MomentRecord(id="m1", type="t", payload={}))
        view = line.commit(text="first", kind="semantic")
        memento.annotate(view.id, title="v2")
        assert len(memento.notes(view.id)) == 2

    def test_boundary_commit(self, memento):
        line = memento.create_line("main")
        for rid in ["a", "b", "c"]:
            line.record(MomentRecord(id=rid, type="t", payload={}))
        view = line.commit(kind="semantic", boundary_moment_id="b")
        assert [m.id for m in memento.show(view.id).moments] == ["a", "b"]
        assert [r.id for r in line.staging()] == ["c"]

    def test_commit_space(self, memento):
        line = memento.create_line("main")
        line.record(MomentRecord(id="m1", type="t", payload={}))
        view = line.commit(kind="semantic")
        assert Path(memento.commit_space(view.id)).exists()

    def test_show_nonexistent_raises(self, memento):
        with pytest.raises(CommitNotFoundError):
            memento.show("cmt_nonexistent")


# ── multi-line ───────────────────────────────────────────────────────────────


class TestMultiLine:
    def test_list_lines(self, memento):
        memento.create_line("main")
        memento.create_line("idea-x")
        assert memento.list_lines() == ["idea-x", "main"]

    def test_parallel_independent_staging(self, memento):
        a = memento.create_line("a")
        b = memento.create_line("b")
        a.record(MomentRecord(id="a1", type="t", payload={}))
        b.record(MomentRecord(id="b1", type="t", payload={}))
        assert a.staging()[0].id == "a1"
        assert b.staging()[0].id == "b1"

    def test_delete_line_keeps_commits(self, memento):
        line = memento.create_line("tmp")
        line.record(MomentRecord(id="x", type="t", payload={}))
        cid = line.commit(kind="mechanical").id
        memento.delete_line("tmp")
        assert memento.show(cid).commit.id == cid

    def test_reset_auto_mechanical_commit(self, memento):
        a = memento.create_line("a")
        b = memento.create_line("b")
        a.record(MomentRecord(id="a1", type="t", payload={}))
        a.commit(text="anchor", kind="semantic")
        b.record(MomentRecord(id="b1", type="t", payload={}))
        vb = b.commit(kind="semantic")
        # staging has content → reset triggers auto mechanical commit
        a.record(MomentRecord(id="a2", type="t", payload={}))
        memento.reset_line("a", BranchRef(origin="ghost.test", commit_id=vb.id))
        assert a.ref.commit_id == vb.id
        assert a.staging() == []

    def test_line_not_found(self, memento):
        with pytest.raises(LineNotFoundError):
            memento.get_line("nonexistent")


# ── log ──────────────────────────────────────────────────────────────────────


class TestLog:
    def test_owner_log(self, memento):
        a = memento.create_line("a")
        b = memento.create_line("b")
        a.record(MomentRecord(id="1", type="t", payload={}))
        a.commit(kind="semantic")
        b.record(MomentRecord(id="2", type="t", payload={}))
        b.commit(kind="mechanical")
        entries = memento.log()
        assert len(entries) == 2
        assert entries[0].branch == "a"
        assert entries[1].branch == "b"

    def test_line_log_parent_chain(self, memento):
        line = memento.create_line("main")
        line.record(MomentRecord(id="m1", type="t", payload={}))
        v1 = line.commit(text="first", kind="semantic")
        line.record(MomentRecord(id="m2", type="t", payload={}))
        v2 = line.commit(text="second", kind="semantic")
        history = line.log()
        assert len(history) == 2
        assert history[0].id == v1.id
        assert history[1].id == v2.id

    def test_window(self, memento):
        line = memento.create_line("main")
        line.record(MomentRecord(id="m1", type="t", payload={}))
        line.commit(text="first", kind="semantic")
        line.record(MomentRecord(id="m2", type="t", payload={}))
        win = line.window(detail_n=3, summary_m=3)
        assert len(win.summaries) == 1
        assert len(win.details) == 1


# ── hooks ────────────────────────────────────────────────────────────────────


class TestHooks:
    def test_hooks_fire(self, tmp_root):
        c = Collector()
        m = new_filesystem_memento(tmp_root, "ghost.test", hooks=c)
        line = m.create_line("main")
        assert c.events[-1][0] == "line_created"
        line.record(MomentRecord(id="m1", type="t", payload={}))
        assert c.events[-1][0] == "record_staged"
        view = line.commit(kind="semantic")
        assert c.events[-1][0] == "commit"
        m.annotate(view.id, title="t")
        assert c.events[-1][0] == "reinterpreted"
        m.delete_line("main")
        assert c.events[-1][0] == "line_deleted"


# ── Y-m utility ──────────────────────────────────────────────────────────────


class TestYM:
    def test_y_m_format(self):
        cid = new_commit_id()
        ym = _y_m(cid)
        assert len(ym) == 7
        assert "-" in ym

    def test_y_m_deterministic(self):
        cid = new_commit_id()
        assert _y_m(cid) == _y_m(cid)


# ── degenerate form ──────────────────────────────────────────────────────────


class TestDegenerate:
    """Invariant #13: no fork vocabulary in dumb-memory usage."""

    def test_no_fork_vocabulary(self, memento):
        line = memento.create_line("main")
        line.record(MomentRecord(id="m1", type="t", payload={"v": 1}))
        v1 = line.commit(kind="mechanical")
        line.record(MomentRecord(id="m2", type="t", payload={"v": 2}))
        v2 = line.commit(kind="mechanical")
        assert len(line.window().summaries) >= 1
        assert len(memento.show(v1.id).moments) == 1
        memento.annotate(v2.id, title="ok")

    def test_annotate_moment(self, memento):
        line = memento.create_line("main")
        line.record(MomentRecord(id="m1", type="t", payload={}))
        view = line.commit(kind="semantic")
        memento.annotate_moment(view.id, "m1", threads=["t1", "t2"])
        # verify threads updated
        detail = memento.show(view.id)
        assert detail.moments[0].threads == ["t1", "t2"]

    def test_annotate_moment_in_staging(self, memento):
        line = memento.create_line("main")
        line.record(MomentRecord(id="m1", type="t", payload={}))
        # annotate before freeze
        memento.annotate_moment("cmt_nonexistent", "m1", threads=["a"])
        assert line.staging()[0].threads == ["a"]


# ── crash recovery ───────────────────────────────────────────────────────────


class TestCrashRecovery:
    def test_recover_truncates_stale_staging(self, tmp_root):
        m = new_filesystem_memento(tmp_root, "ghost.test")
        line = m.create_line("main")
        line.record(MomentRecord(id="m1", type="t", payload={}))
        view = line.commit(kind="semantic")
        # simulate crash: stale staging with frozen id
        sp = m._staging_path("main")
        sp.write_text(
            '{"t":"moment","id":"m1","created":"2026-07-21T00:00:00+00:00","type":"t","payload":{},"threads":[]}\n',
            encoding="utf-8",
        )
        line2 = m.get_line("main")
        assert line2.staging() == []
