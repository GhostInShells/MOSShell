"""lean memento (abcd.py) 契约行为测试 — 经 _fs_memento filesystem 实现验证.

每个测试证明一条协议承诺, 不测实现内部状态 / 私有成员 / 框架默认值.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from ghoshell_moss.memento._fs_memento import FsMemento, new_local_memento


@pytest.fixture
def memento(tmp_path: Path) -> FsMemento:
    return new_local_memento(tmp_path / "owner")


# ── branch 生命周期 ──


def test_create_branch_and_list(memento):
    memento.create_branch("main", "primary")
    memento.create_branch("idea")

    names = sorted(b.name for b in memento.list_branches())
    assert names == ["idea", "main"]


def test_create_duplicate_raises(memento):
    memento.create_branch("main")
    with pytest.raises(NameError):
        memento.create_branch("main")


def test_delete_keeps_dir_removes_pointer(memento):
    b = memento.create_branch("main")
    branch_dir = b.path
    b.commit(message="keep me")

    memento.delete_branch("main")

    assert memento.get_branch("main") is None
    assert branch_dir.exists()  # 目录与 commits 保留


# ── commit / note 拆两步 ──


def test_commit_is_bare_anchor_then_note_fills_message(memento):
    b = memento.create_branch("main")
    # step 1: 裸锚点, 无 message (还原钥匙 opaque 存 metadata)
    ref = b.commit(metatype="session", metadata={"session_id": "s1", "tail": 3})
    assert ref.metadata == {"session_id": "s1", "tail": 3}
    assert b.notes() == {}  # 未种子任何 Note

    # step 2: 稍后补 message (旁路)
    b.note(ref.id, "first commit title\nbody here")

    assert b.notes()[ref.id].message == "first commit title\nbody here"


def test_commit_with_message_seeds_note(memento):
    b = memento.create_branch("main")
    ref = b.commit(message="second\nhas body")

    # 便捷糖: message 非空时种子一条 Note, 但单一真值仍在 Note
    assert b.notes()[ref.id].message == "second\nhas body"


def test_note_last_wins(memento):
    b = memento.create_branch("main")
    ref = b.commit()

    b.note(ref.id, "v1 title\nold body")
    b.note(ref.id, "v2 title\nnew body")

    assert b.notes()[ref.id].message == "v2 title\nnew body"
    assert len([n for n in b.notes().values() if n.commit_id == ref.id]) == 1


# ── view 折叠 ──


def test_view_folds_latest_history_and_derives_title(memento):
    b = memento.create_branch("main")
    for i in range(3):
        b.commit(message=f"commit {i}\nbody {i}")

    v = b.view(n=2)

    assert v.commits_total == 3
    assert len(v.latest) == 2
    assert len(v.history) == 1
    # title = message 首行, body = 其余
    assert v.latest[-1].title == "commit 2"
    assert v.latest[-1].body == "body 2"
    assert v.history[0].title == "commit 0"


# ── fork 引用 ──


def test_fork_references_parent_without_copy(memento):
    parent = memento.create_branch("main")
    parent.commit(message="one")
    tip = parent.commit(message="two")

    child = parent.fork("idea")

    assert len(child.commits()) == 0  # 不复制父支 commits
    fr = child.meta().fork_from
    assert fr is not None
    assert fr.branch_id == parent.ref.branch_id
    assert fr.commit_id == tip.id  # 从父支当前 tip 分叉


def test_fork_view_recaps_parent(memento):
    parent = memento.create_branch("main")
    parent.commit(message="one\nbody")
    parent.commit(message="two\nbody")

    child = parent.fork("idea")
    v = child.view()

    assert v.previous is not None
    assert v.previous.name == "main"
    assert v.previous.commits_total == 2


# ── 时间范围查询 ──


def test_query_commits_time_range(memento):
    b = memento.create_branch("main")
    first = b.commit(message="first")
    b.commit(message="second")

    first_created = first.created

    within = b.query_commits(from_date=first_created)
    assert len(within) == 2

    before = b.query_commits(until_date=first_created - timedelta(seconds=1))
    assert before == []


# ── 写门控与异步面 ──


def test_write_gate_fast_fails(memento):
    b = memento.create_branch("main")

    async def inner():
        async with b:
            with pytest.raises(BlockingIOError):
                async with b:
                    pass

    asyncio.run(inner())


def test_async_reads_match_sync(memento):
    b = memento.create_branch("main")
    ref = b.commit(message="title\nbody")

    async def inner():
        commits = await b.acommits()
        notes = await b.anotes()
        view = await b.aview()
        return commits, notes, view

    commits, notes, view = asyncio.run(inner())

    assert [c.id for c in commits] == [c.id for c in b.commits()]
    assert notes == b.notes()
    assert view.commits_total == b.view().commits_total
    assert view.latest[-1].title == "title"


def test_async_write_ops_self_contained(memento):
    b = memento.create_branch("main")

    async def inner():
        await b.acommit(message="x\ny")
        ref = await b.acommit()
        await b.anote(ref.id, "late\ntitle")
        child = await b.afork("idea")
        return child

    child = asyncio.run(inner())

    assert child.meta().fork_from is not None
    assert b.view().commits_total == 2
    assert b.notes()[b.commits()[-1].id].message == "late\ntitle"
