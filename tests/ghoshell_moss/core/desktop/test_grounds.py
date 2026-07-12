"""Integration tests — DefaultGrounds + DefaultGround end-to-end.

覆盖: open/close 生命周期, pin/unpin/update 全链路, sediment/load roundtrip,
label 分配 (缺省+冲突), 幂等 open, context 帧渲染, boundary 校验, KeyError
契约, async with 生命周期.

不覆盖: CTML channel 装配 (下一步 channels/desktop_channel.py 才做);
CLI dogfood (下一步 moss desktop CLI 才做).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from ghoshell_moss.contracts.desktop import (
    GroundConvention,
    PathOutsideRootError,
)
from ghoshell_moss.core.desktop import (
    DEFAULT_L0_FILENAME,
    DefaultGrounds,
)
from ghoshell_moss.core.desktop._l0 import load_l0


def run(coro):
    return asyncio.run(coro)


# ---- open / close / active --------------------------------------------


def test_open_returns_ground_with_correct_root_and_label(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()

    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            ground = await gs.open("sub")
            assert ground.root == (tmp_path / "sub").resolve()
            assert ground.label == "sub"
            assert list(gs.active().keys()) == ["sub"]
            return ground

    run(scenario())


def test_open_absolute_path(tmp_path: Path) -> None:
    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            assert g.root == tmp_path.resolve()

    run(scenario())


def test_open_idempotent_same_dir_returns_same_instance(tmp_path: Path) -> None:
    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g1 = await gs.open(tmp_path)
            g2 = await gs.open(tmp_path, label="ignored")
            g3 = await gs.open(str(tmp_path))
            assert g1 is g2
            assert g1 is g3
            assert len(gs.active()) == 1

    run(scenario())


def test_open_label_conflict_gets_suffix(tmp_path: Path) -> None:
    (tmp_path / "a" / "foo").mkdir(parents=True)
    (tmp_path / "b" / "foo").mkdir(parents=True)

    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g1 = await gs.open(tmp_path / "a" / "foo")
            g2 = await gs.open(tmp_path / "b" / "foo")
            assert g1.label == "foo"
            assert g2.label == "foo-2"

    run(scenario())


def test_open_explicit_label(tmp_path: Path) -> None:
    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path, label="my-scope")
            assert g.label == "my-scope"

    run(scenario())


def test_close_removes_from_active_and_sediments(tmp_path: Path) -> None:
    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            g.pin("some_file.md")  # arbitrary addr; file doesn't need to exist
            assert "some_file.md" in {p.addr for p in g.pins()}
            await gs.close(g.label)
            assert g.label not in gs.active()

        # 落盘后再开, pin 应恢复
        async with DefaultGrounds(workspace_root=tmp_path) as gs2:
            g2 = await gs2.open(tmp_path)
            addrs = {p.addr for p in g2.pins()}
            assert "some_file.md" in addrs

    run(scenario())


def test_close_missing_label_raises_keyerror(tmp_path: Path) -> None:
    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            with pytest.raises(KeyError):
                await gs.close("no-such-label")

    run(scenario())


def test_get_returns_none_for_missing(tmp_path: Path) -> None:
    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            assert gs.get("no-such") is None
            g = await gs.open(tmp_path)
            assert gs.get(g.label) is g

    run(scenario())


def test_aexit_sediments_all_active(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            ga = await gs.open(tmp_path / "a")
            gb = await gs.open(tmp_path / "b")
            ga.pin("a.md")
            gb.pin("b.md")

        # 两份 L0 各自落盘
        a_pins = load_l0(tmp_path / "a").pins
        b_pins = load_l0(tmp_path / "b").pins
        assert {p.addr for p in a_pins} == {"a.md"}
        assert {p.addr for p in b_pins} == {"b.md"}

    run(scenario())


# ---- pin / unpin / update (forwarding) ---------------------------------


def test_grounds_pin_forwards_to_ground(tmp_path: Path) -> None:
    (tmp_path / "target.py").write_text("hello\n")

    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            pin = gs.pin(g.label, "target.py", note="main")
            assert pin.addr == "target.py"
            assert pin.note == "main"
            assert pin.seen_hash is not None  # 观察发生了

    run(scenario())


def test_pin_boundary_check_raises(tmp_path: Path) -> None:
    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            with pytest.raises(PathOutsideRootError):
                g.pin("../escape.py")

    run(scenario())


def test_pin_missing_file_still_pins_with_none_hash(tmp_path: Path) -> None:
    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            pin = g.pin("does-not-exist.md")
            assert pin.seen_mtime is None
            assert pin.seen_hash is None
            assert pin.addr in {p.addr for p in g.pins()}

    run(scenario())


def test_pin_idempotent_repin_refreshes_pinned_at(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("hello")
    import time as _time

    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            p1 = g.pin("a.md", note="first")
            _time.sleep(0.01)
            p2 = g.pin("a.md")  # 无 note, 应保留 "first"
            assert p2.pinned_at > p1.pinned_at
            assert p2.note == "first"

    run(scenario())


def test_pin_order_most_recent_first(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("a")
    (tmp_path / "b.md").write_text("b")
    (tmp_path / "c.md").write_text("c")

    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            g.pin("a.md")
            g.pin("b.md")
            g.pin("c.md")
            addrs = [p.addr for p in g.pins()]
            # 最新在前
            assert addrs == ["c.md", "b.md", "a.md"]

    run(scenario())


def test_unpin_removes_and_raises_on_missing(tmp_path: Path) -> None:
    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            g.pin("x.md")
            g.unpin("x.md")
            assert "x.md" not in {p.addr for p in g.pins()}
            with pytest.raises(KeyError):
                g.unpin("x.md")

    run(scenario())


def test_update_after_edit_marks_changed(tmp_path: Path) -> None:
    target = tmp_path / "a.md"
    target.write_text("v1")

    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            initial = g.pin("a.md")
            assert initial.seen_hash is not None

            target.write_text("v2")
            result = await gs.update(g.label, "a.md")
            assert result.changed is True
            assert result.new_mtime is not None
            assert "content changed" in result.diff_preview

            # 再 update 一次: 无变化
            result2 = await gs.update(g.label, "a.md")
            assert result2.changed is False

    run(scenario())


def test_update_missing_pin_raises_keyerror(tmp_path: Path) -> None:
    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            with pytest.raises(KeyError):
                await gs.update(g.label, "not-pinned.md")

    run(scenario())


# ---- frame / context rendering -----------------------------------------


def test_frame_renders_pins_and_content(tmp_path: Path) -> None:
    (tmp_path / "hello.md").write_text("line1\nline2\nline3\n")

    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            g.pin("hello.md", note="the greeting")
            frame = await gs.frame(g.label)
            assert "hello.md" in frame
            assert "the greeting" in frame
            assert "line1" in frame
            assert "line2" in frame

    run(scenario())


def test_frame_marks_changed_on_disk(tmp_path: Path) -> None:
    target = tmp_path / "shift.md"
    target.write_text("before")

    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            g.pin("shift.md")
            # 修改文件, 不 update
            target.write_text("after — different content")
            frame = await gs.frame(g.label)
            assert "changed on disk" in frame

    run(scenario())


def test_frame_marks_missing(tmp_path: Path) -> None:
    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            g.pin("does-not-exist.md")
            frame = await gs.frame(g.label)
            assert "missing" in frame

    run(scenario())


def test_frame_glob_lists_hits(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.py").write_text("b")

    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            g.pin("*.py")
            frame = await gs.frame(g.label)
            assert "a.py" in frame
            assert "b.py" in frame

    run(scenario())


def test_frame_budget_warning_when_over(tmp_path: Path) -> None:
    big = "X" * 5000
    (tmp_path / "big.md").write_text(big)

    async def scenario():
        # budget = 100, 5000 字符文件明显超
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(
                tmp_path, convention=GroundConvention(context_budget=100)
            )
            g.pin("big.md")
            frame = await gs.frame(g.label)
            assert "over budget" in frame
            # 但内容仍然渲染 (K20 不自动截断)
            assert "X" * 100 in frame

    run(scenario())


# ---- instruction chain -------------------------------------------------


def test_instruction_cache_populated_after_load(tmp_path: Path) -> None:
    (tmp_path / "CLAUDE.md").write_text("law")

    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            assert "law" in g.instruction()

    run(scenario())


def test_instruction_cache_persists_after_file_change(tmp_path: Path) -> None:
    """K17 语义: 法链不自动同步. 文件改了 instruction() 仍返回旧值,
    需 refresh_instruction() 才承认."""
    claude = tmp_path / "CLAUDE.md"
    claude.write_text("old law")

    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            assert "old law" in g.instruction()
            claude.write_text("new law")
            # 不 refresh — 仍是旧的
            assert "old law" in g.instruction()
            assert "new law" not in g.instruction()
            await g.refresh_instruction()
            assert "new law" in g.instruction()
            assert "old law" not in g.instruction()

    run(scenario())


def test_frame_shows_child_hints(tmp_path: Path) -> None:
    (tmp_path / "child").mkdir()
    (tmp_path / "child" / "CLAUDE.md").write_text("child law")

    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            frame = await gs.frame(g.label)
            assert "instruction, unloaded" in frame
            assert "child/CLAUDE.md" in frame

    run(scenario())


# ---- sediment / load roundtrip -----------------------------------------


def test_sediment_load_roundtrip_preserves_state(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("aaa")
    (tmp_path / "b.py").write_text("bbb")

    async def phase1():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            g.pin("a.md", note="first")
            g.pin("b.py", note="second")
            # __aexit__ 自动 sediment

    async def phase2():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            addrs = [p.addr for p in g.pins()]
            notes = {p.addr: p.note for p in g.pins()}
            assert set(addrs) == {"a.md", "b.py"}
            assert notes["a.md"] == "first"
            assert notes["b.py"] == "second"

    run(phase1())
    assert (tmp_path / DEFAULT_L0_FILENAME).is_file()
    run(phase2())


def test_sediment_preserves_existing_body(tmp_path: Path) -> None:
    (tmp_path / DEFAULT_L0_FILENAME).write_text(
        "---\ntree_depth: 3\n---\n\n# Governance\n\nrules here\n"
    )

    async def scenario():
        async with DefaultGrounds(workspace_root=tmp_path) as gs:
            g = await gs.open(tmp_path)
            g.pin("something.md")
            # frontmatter 已加载
            assert g.convention.tree_depth == 3

        text = (tmp_path / DEFAULT_L0_FILENAME).read_text()
        assert "# Governance" in text
        assert "rules here" in text
        assert "tree_depth: 3" in text
        assert "something.md" in text

    run(scenario())
