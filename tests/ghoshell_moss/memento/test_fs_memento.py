"""FsMemento unit tests — FORMAT v3 behavior verification."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from ghoshell_moss.memento import _storage as store
from ghoshell_moss.memento.abc import (
    BranchRef,
    BranchNotFoundError,
    CommitNotFoundError,
    CommitView,
    EmptyStagingError,
    LineNotFoundError,
    MementoHooks,
    MomentFrozenError,
    MomentRecord,
    new_commit_id,
)
from ghoshell_moss.memento.fs_memento import (
    FsMemento,
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

    def on_branch_checkout(self, branch_identifier: str, from_ref: BranchRef) -> None:
        self.events.append(("branch_checkout", {"uid": branch_identifier}))


# ── basic lifecycle ─────────────────────────────────────────────────────────


class TestBasicLifecycle:
    def test_create_line_and_record(self, memento):
        line = memento.create_line("main")
        assert line.name == "main"
        assert line.branch_identifier.startswith(store.BRANCH_ID_PREFIX)
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
        assert view.commit.id.startswith(store.COMMIT_ID_PREFIX)
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
        assert sorted(memento.list_lines()) == ["idea-x", "main"]

    def test_parallel_independent_staging(self, memento):
        a = memento.create_line("a")
        b = memento.create_line("b")
        a.record(MomentRecord(id="a1", type="t", payload={}))
        b.record(MomentRecord(id="b1", type="t", payload={}))
        assert a.staging()[0].id == "a1"
        assert b.staging()[0].id == "b1"

    def test_delete_line_keeps_commits(self, memento):
        line = memento.create_line("tmp")
        buid = line.branch_identifier
        line.record(MomentRecord(id="x", type="t", payload={}))
        cid = line.commit(kind="mechanical").id
        memento.delete_line("tmp")
        # commit still accessible
        assert memento.show(cid).commit.id == cid
        # workspace survives
        assert "tmp" not in memento.list_lines()
        all_uids = {b.uid for b in memento.list_all_branches()}
        assert buid in all_uids

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
        # v3: branch field holds uid
        assert entries[0].branch == a.branch_identifier
        assert entries[1].branch == b.branch_identifier

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
        assert c.events[-1][0] == "branch_checkout"
        # line_created fires before branch_checkout
        assert c.events[-2][0] == "line_created"
        line.record(MomentRecord(id="m1", type="t", payload={}))
        assert c.events[-1][0] == "record_staged"
        view = line.commit(kind="semantic")
        assert c.events[-1][0] == "commit"
        m.annotate(view.id, title="t")
        assert c.events[-1][0] == "reinterpreted"
        m.delete_line("main")
        assert c.events[-1][0] == "line_deleted"


# ── degenerate form ──────────────────────────────────────────────────────────


class TestDegenerate:
    def test_get_line_main_auto_creates(self, memento):
        """v3 degenerate path: get_line('main') creates the line on first access."""
        line = memento.get_line("main")
        assert line.name == "main"
        assert line.branch_identifier.startswith(store.BRANCH_ID_PREFIX)
        line.record(MomentRecord(id="m1", type="t", payload={"v": 1}))
        line.commit(kind="mechanical")
        assert len(line.log()) == 1

    def test_no_fork_vocabulary_at_api_level(self, memento):
        """Degenerate form: create_line/record/commit — no fork/confluent in use."""
        line = memento.get_line("main")
        line.record(MomentRecord(id="m1", type="t", payload={"v": 1}))
        v1 = line.commit(kind="mechanical")
        line.record(MomentRecord(id="m2", type="t", payload={"v": 2}))
        v2 = line.commit(kind="mechanical")
        assert len(line.window().summaries) >= 1
        assert len(memento.show(v1.id).moments) == 1
        memento.annotate(v2.id, title="ok")

    def test_annotate_moment(self, memento):
        line = memento.get_line("main")
        line.record(MomentRecord(id="m1", type="t", payload={}))
        view = line.commit(kind="semantic")
        memento.annotate_moment(view.id, "m1", threads=["t1", "t2"])
        detail = memento.show(view.id)
        assert detail.moments[0].threads == ["t1", "t2"]


# ── crash recovery ───────────────────────────────────────────────────────────


class TestCrashRecovery:
    def test_recover_truncates_staging_and_repairs_ref(self, tmp_root):
        """v3: commit persisted but staging not truncated and ref rolled back.
        Recovery should truncate staging and repair ref."""
        m = new_filesystem_memento(tmp_root, "ghost.test")
        line = m.create_line("main")
        buid = line.branch_identifier
        line.record(MomentRecord(id="m1", type="t", payload={}))
        view = line.commit(kind="semantic")
        cid = view.id

        # Simulate crash: re-insert m1 into staging, roll back ref
        sp = tmp_root / "ghost.test" / "ws" / buid / "staging.jsonl"
        sp.write_text(
            '{"t":"moment","id":"m1","created":"2026-08-04T00:00:00Z","type":"t","content":"","payload":{}}\n',
            encoding="utf-8",
        )
        rp = tmp_root / "ghost.test" / "ws" / buid / "ref"
        rp.write_text("{}", encoding="utf-8")

        # Re-open to trigger recovery (recovery runs at load time)
        m2 = new_filesystem_memento(tmp_root, "ghost.test")
        line2 = m2.get_line("main")
        assert line2.ref is not None
        assert line2.ref.commit_id == cid
        assert line2.staging() == []

    def test_fork_lines_have_independent_refs(self, tmp_root):
        """Recovery must not overwrite all line refs with the last commits.jsonl entry."""
        m = new_filesystem_memento(tmp_root, "ghost.test")
        main = m.create_line("main")
        main.record(MomentRecord(id="m1", type="t", payload={}))
        v1 = main.commit(kind="semantic")

        idea = m.create_line("idea-x", from_ref=BranchRef(commit_id=v1.id))
        idea.record(MomentRecord(id="i1", type="t", payload={}))
        v2 = idea.commit(kind="semantic")

        main2 = m.get_line("main")
        idea2 = m.get_line("idea-x")
        assert main2.ref.commit_id == v1.id
        assert idea2.ref.commit_id == v2.id


# ── v3 new features ──────────────────────────────────────────────────────────


class TestV3Features:
    def test_branch_uid_stable_across_operations(self, memento):
        """uid never changes across record/commit cycles."""
        line = memento.create_line("main")
        uid = line.branch_identifier
        line.record(MomentRecord(id="m1", type="t", payload={}))
        line.commit(kind="mechanical")
        assert line.branch_identifier == uid

    def test_list_all_branches_active_and_abandoned(self, memento):
        a = memento.create_line("a")
        b = memento.create_line("b")
        a.record(MomentRecord(id="a1", type="t", payload={}))
        a.commit(kind="mechanical")
        memento.delete_line("b")
        branches = memento.list_all_branches()
        assert len(branches) == 2
        uids = {br.uid for br in branches}
        assert a.branch_identifier in uids
        assert b.branch_identifier in uids

    def test_content_field_persists_through_commit(self, memento):
        line = memento.get_line("main")
        line.record(MomentRecord(id="m1", type="test/v1",
                                  payload={"x": 1}, content="the answer"))
        view = line.commit("content commit", kind="mechanical")
        detail = memento.show(view.id)
        assert detail.moments[0].content == "the answer"

    def test_checkouts_indexes_fork_events(self, memento):
        a = memento.create_line("a")
        a.record(MomentRecord(id="a1", type="t", payload={}))
        va = a.commit("a anchor", kind="semantic")
        b = memento.create_line("b", from_ref=BranchRef(commit_id=va.id))
        checkouts = memento.checkouts()
        assert len(checkouts) == 2
        assert checkouts[-1].branch_uid == b.branch_identifier

    def test_confluents_empty_by_default(self, memento):
        memento.get_line("main")
        assert memento.confluents() == []

    def test_get_line_by_uid(self, memento):
        line = memento.create_line("main")
        uid = line.branch_identifier
        same = memento.get_line(uid)
        assert same.branch_identifier == uid

    def test_get_line_by_uid_raises_for_unknown(self, memento):
        with pytest.raises(BranchNotFoundError):
            memento.get_line("brn_nonexistent")
