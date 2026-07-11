"""
FsMemento 契约行为测试 — 旧版验收五条 + FORMAT.md 新增条款.

验收五条: 单 owner 生命周期 / 多 owner 只读边界 / persistence round-trip /
base chain 回溯 / hook fan-out.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ghoshell_moss.core.memento import (
    BranchWindow,
    CommitNotFoundError,
    EmptyStagingError,
    FsMemento,
    MementoError,
    MomentFrozenError,
    MomentRecord,
    ReadonlyBranchError,
    join_trailers,
    new_filesystem_memento,
    split_trailers,
    trailer_values,
)


def _rec(mid: str, **payload) -> MomentRecord:
    return MomentRecord(id=mid, type="test.data/v1", payload=payload or {"n": mid})


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    return tmp_path / "memento"


# ============================================================
# 1. 单 owner 生命周期
# ============================================================


def test_single_owner_lifecycle(root: Path):
    m = new_filesystem_memento(root, "ghost.main")
    branch = m.current()
    assert branch.meta.name == "main"
    assert branch.head() is None
    assert branch.staging() == []

    branch.update(_rec("m1"))
    branch.update(_rec("m2"))
    assert [r.id for r in branch.staging()] == ["m1", "m2"]

    view = branch.commit("first slice", kind="semantic", threads=["alpha"])
    assert view.seq == 1
    assert view.commit.moment_ids == ["m1", "m2"]
    assert view.summary() == "first slice"
    assert view.note.kind() == "semantic"
    assert view.note.threads() == ["alpha"]
    assert branch.staging() == []
    assert branch.head().id == view.id

    branch.update(_rec("m3"))
    view2 = branch.commit(kind="mechanical")
    assert view2.seq == 2
    assert view2.summary() == ""  # 机械快照不编造正文
    assert [v.seq for v in branch.own_commits()] == [1, 2]

    records = branch.commit_records(view.id)
    assert [r.id for r in records] == ["m1", "m2"]
    m.close()


def test_update_overwrite_keeps_staging_order(root: Path):
    m = new_filesystem_memento(root, "o")
    b = m.current()
    b.update(_rec("a", v=1))
    b.update(_rec("b"))
    b.update(_rec("a", v=2))  # 覆盖写: staging 保持首次出现位置序
    assert [r.id for r in b.staging()] == ["a", "b"]
    assert b.staging()[0].payload == {"v": 2}  # last-wins


def test_frozen_moment_rejects_update(root: Path):
    m = new_filesystem_memento(root, "o")
    b = m.current()
    b.update(_rec("a"))
    b.commit(kind="mechanical")
    with pytest.raises(MomentFrozenError):
        b.update(_rec("a", v=2))


def test_empty_staging_rejects_commit(root: Path):
    m = new_filesystem_memento(root, "o")
    with pytest.raises(EmptyStagingError):
        m.current().commit(kind="mechanical")


def test_moment_annotate_threads_after_freeze(root: Path):
    # threads 是释义不是成员 — 冻结后仍可改写 (孔径二在 moment 级的形态)
    m = new_filesystem_memento(root, "o")
    b = m.current()
    b.update(_rec("a"))
    b.commit(kind="mechanical")
    b.annotate_moment("a", ["retagged"], by="tagger")
    assert m.pool.get("a").threads == ["retagged"]


def test_reinterpret_last_wins_and_forensics(root: Path):
    m = new_filesystem_memento(root, "o")
    b = m.current()
    b.update(_rec("a"))
    v0 = b.commit("draft summary", kind="semantic")
    v1 = b.reinterpret(v0.id, join_trailers("better summary", [("Kind", "semantic")]), by="reflector")
    assert v1.note_seq == 1
    assert b.get_commit(v0.id).summary() == "better summary"  # last-wins
    versions = b.notes(v0.id)
    assert len(versions) == 2
    assert versions[0].text() == "draft summary"  # 原版本永远可寻址


def test_switch_and_list(root: Path):
    m = new_filesystem_memento(root, "o")
    b1 = m.current()
    b1.update(_rec("a"))
    b1.commit(kind="mechanical")
    b2 = m.checkout(base_fork="o", base_branch_id=b1.meta.branch_id, name="side")
    m.switch(b2.meta.branch_id)
    assert m.current().meta.branch_id == b2.meta.branch_id
    assert {meta.name for meta in m.list_branches()} == {"main", "side"}


# ============================================================
# 2. 多 owner 只读边界
# ============================================================


def test_cross_owner_readonly(root: Path):
    alice = new_filesystem_memento(root, "alice")
    ab = alice.current()
    ab.update(_rec("a1"))
    committed = ab.commit("alice work", kind="semantic")

    bob = new_filesystem_memento(root, "bob")
    handle = bob.get_branch(ab.meta.branch_id, fork="alice")
    assert handle.readonly
    assert handle.head().id == committed.id  # 读没问题
    with pytest.raises(ReadonlyBranchError):
        handle.update(_rec("b1"))
    with pytest.raises(ReadonlyBranchError):
        handle.commit(kind="mechanical")
    with pytest.raises(ReadonlyBranchError):
        handle.reinterpret(committed.id, "rewrite")
    assert "alice" in bob.list_forks() and "bob" not in bob.list_forks()  # bob 还没写过


# ============================================================
# 3. persistence round-trip
# ============================================================


def test_persistence_round_trip(root: Path):
    m1 = new_filesystem_memento(root, "o")
    b = m1.current()
    b.update(_rec("a", text="早上好"))  # 非 ASCII 保真
    b.update(_rec("b", text="line1\nline2"))  # 换行转义
    view = b.commit("多行\n正文", kind="semantic", threads=["t1", "t2"])
    b.update(_rec("c"))
    m1.close()

    # 全新实例, 冷读
    m2 = new_filesystem_memento(root, "o")
    b2 = m2.current()
    assert b2.meta.branch_id == b.meta.branch_id
    head = b2.head()
    assert head.id == view.id
    assert head.summary() == "多行\n正文"
    assert head.note.threads() == ["t1", "t2"]
    records = b2.commit_records(view.id)
    assert records[0].payload == {"text": "早上好"}
    assert records[1].payload == {"text": "line1\nline2"}
    assert [r.id for r in b2.staging()] == ["c"]  # staging 也存活
    m2.close()


# ============================================================
# 4. base chain 回溯
# ============================================================


def _grow(branch, prefix: str, n: int):
    views = []
    for i in range(n):
        branch.update(_rec(f"{prefix}{i}"))
        views.append(branch.commit(f"{prefix} commit {i}", kind="mechanical"))
    return views


def test_base_chain_traversal_two_levels(root: Path):
    m = new_filesystem_memento(root, "o")
    root_branch = m.current()
    rv = _grow(root_branch, "r", 3)

    # 从 root 的第 2 个 commit 分出 child
    child = m.checkout(
        base_fork="o", base_branch_id=root_branch.meta.branch_id, base_commit_id=rv[1].id, name="child"
    )
    assert len(child.meta.ancestry) == 1
    cv = _grow(child, "c", 2)

    # 从 child 的 head 分出 grandchild — 祖先链应冻结为两段
    grand = m.checkout(base_fork="o", base_branch_id=child.meta.branch_id, name="grand")
    assert len(grand.meta.ancestry) == 2
    assert grand.meta.ancestry[-1] == grand.meta.base

    history = grand.all_commits()
    # root 截至 seq2 (2 个) + child 全部 (2 个)
    assert [v.id for v in history] == [rv[0].id, rv[1].id, cv[0].id, cv[1].id]
    # 回溯 API 能取到祖先的 commit 与成员
    assert grand.get_commit(rv[0].id) is not None
    assert [r.id for r in grand.commit_records(cv[1].id)] == ["c1"]
    # rv[2] 在截断点之后, 不属于 grand 的历史
    assert grand.get_commit(rv[2].id) is None


def test_checkout_from_ancestor_segment(root: Path):
    m = new_filesystem_memento(root, "o")
    rb = m.current()
    rv = _grow(rb, "r", 2)
    child = m.checkout(base_fork="o", base_branch_id=rb.meta.branch_id, name="child")
    _grow(child, "c", 1)
    # 从 child 的历史里取一个"其实属于 root"的 commit 作起点
    grand = m.checkout(
        base_fork="o", base_branch_id=child.meta.branch_id, base_commit_id=rv[0].id, name="grand"
    )
    # base 直接指向 root branch, 不经过 child
    assert grand.meta.base.branch_id == rb.meta.branch_id
    assert grand.meta.base.commit_seq == 1
    assert [v.id for v in grand.all_commits()] == [rv[0].id]


def test_checkout_never_from_staging(root: Path):
    m = new_filesystem_memento(root, "o")
    b = m.current()
    b.update(_rec("a"))  # 只有 staging, 没有 commit
    with pytest.raises(MementoError):
        m.checkout(base_fork="o", base_branch_id=b.meta.branch_id)


def test_checkout_cross_owner_with_overlay(root: Path):
    alice = new_filesystem_memento(root, "alice")
    ab = alice.current()
    ab.update(_rec("a"))
    ab.commit(kind="mechanical")

    bob = new_filesystem_memento(root, "bob")
    nb = bob.checkout(
        base_fork="alice",
        base_branch_id=ab.meta.branch_id,
        overlay={"divergence_prompt": "you are the side thinker"},
    )
    assert nb.meta.fork == "bob"
    assert not nb.readonly
    assert nb.meta.overlay["divergence_prompt"] == "you are the side thinker"
    # overlay 不在对话历史里
    assert nb.staging() == []


def test_ancestry_tamper_detected(root: Path):
    m = new_filesystem_memento(root, "o")
    b = m.current()
    b.update(_rec("a"))
    b.commit(kind="mechanical")
    child = m.checkout(base_fork="o", base_branch_id=b.meta.branch_id)
    meta_path = root / "branches" / "o" / child.meta.branch_id / "meta.json"
    data = json.loads(meta_path.read_text())
    data["ancestry"] = []  # 篡改冻结祖先链
    meta_path.write_text(json.dumps(data))
    with pytest.raises(MementoError):
        m.get_branch(child.meta.branch_id)


# ============================================================
# 5. hook fan-out
# ============================================================


class CollectorHooks:
    def __init__(self):
        self.events: list[tuple] = []

    def on_record_staged(self, branch_id, record):
        self.events.append(("staged", branch_id, record.id))

    def on_commit(self, branch_id, view):
        self.events.append(("commit", branch_id, view.id))

    def on_reinterpreted(self, branch_id, view):
        self.events.append(("reinterpreted", branch_id, view.note_seq))

    def on_branch_created(self, meta):
        self.events.append(("created", meta.branch_id))

    def on_branch_switched(self, branch_id):
        self.events.append(("switched", branch_id))


def test_hook_fan_out(root: Path):
    hooks = CollectorHooks()
    m = new_filesystem_memento(root, "o", hooks=hooks)
    b = m.current()
    bid = b.meta.branch_id
    b.update(_rec("a"))
    v = b.commit("s", kind="semantic")
    b.reinterpret(v.id, "s2")
    nb = m.checkout(base_fork="o", base_branch_id=bid)
    m.switch(nb.meta.branch_id)

    kinds = [e[0] for e in hooks.events]
    assert kinds == ["created", "staged", "commit", "reinterpreted", "created", "switched"]
    assert ("staged", bid, "a") in hooks.events
    assert ("commit", bid, v.id) in hooks.events
    assert ("reinterpreted", bid, 1) in hooks.events


# ============================================================
# 窗口快路径
# ============================================================


def test_window_fast_path(root: Path):
    m = new_filesystem_memento(root, "o")
    b = m.current()
    views = _grow(b, "x", 3)  # 每个 commit 1 帧
    b.update(_rec("live"))

    w: BranchWindow = b.window(detail_n=2, summary_m=-1)
    # 明细 2 帧: staging 的 live + 最新 commit 展开的 x2
    assert [r.id for r in w.details] == ["x2", "live"]
    # 前两个 commit 折叠为摘要
    assert [v.id for v in w.summaries] == [views[0].id, views[1].id]

    w2 = b.window(detail_n=0, summary_m=1)
    assert w2.details == []
    assert [v.id for v in w2.summaries] == [views[2].id]


# ============================================================
# trailer 规范 (abc 纯函数)
# ============================================================


def test_trailer_round_trip():
    body = join_trailers(
        "did the thing.",
        [("Thread", "alpha"), ("Thread", "beta"), ("Resumes", "cmt_X"), ("Kind", "semantic")],
    )
    text, trailers = split_trailers(body)
    assert text == "did the thing."
    assert trailer_values(trailers, "Thread") == ["alpha", "beta"]
    assert trailer_values(trailers, "Resumes") == ["cmt_X"]


def test_trailer_only_body():
    body = join_trailers("", [("Kind", "mechanical")])
    text, trailers = split_trailers(body)
    assert text == ""
    assert trailers == [("Kind", "mechanical")]


def test_trailer_requires_blank_line_separation():
    # 正文紧贴 Key: value 行 (无空行) — 整体视为正文, 不解析 trailer
    text, trailers = split_trailers("prose line\nKind: semantic")
    assert trailers == []
    assert "Kind: semantic" in text


def test_trailer_unknown_key_preserved():
    body = join_trailers("x", [("Kind", "semantic"), ("X-Custom", "kept")])
    _, trailers = split_trailers(body)
    assert trailer_values(trailers, "X-Custom") == ["kept"]
