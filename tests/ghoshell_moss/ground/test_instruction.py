"""Tests for _instruction.py — 法链收集与边界处理."""
from __future__ import annotations

from pathlib import Path

from ghoshell_moss.ground.contract import GroundConvention
from ghoshell_moss.ground._instruction import collect_instructions


def _write(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def test_empty_when_no_files(tmp_path: Path) -> None:
    result = collect_instructions(tmp_path, GroundConvention())
    assert result == ""


def test_collect_from_root_only(tmp_path: Path) -> None:
    _write(tmp_path / "CLAUDE.md", "root law")
    result = collect_instructions(tmp_path, GroundConvention(upward_lookup=False))
    assert "root law" in result
    assert "from:" in result  # 来源标注


def test_upward_walk_root_last(tmp_path: Path) -> None:
    outer = tmp_path
    inner = tmp_path / "sub" / "deep"
    inner.mkdir(parents=True)
    _write(outer / "CLAUDE.md", "MARKER_OUTER")
    _write(inner / "CLAUDE.md", "MARKER_INNER")

    # 从 inner 出发, boundary = outer
    result = collect_instructions(
        inner,
        GroundConvention(upward_boundary=str(outer)),
    )
    # 根最先: OUTER 出现在 INNER 之前
    assert result.index("MARKER_OUTER") < result.index("MARKER_INNER")


def test_upward_stops_at_boundary_inclusive(tmp_path: Path) -> None:
    a = tmp_path
    b = tmp_path / "sub_b"
    c = tmp_path / "sub_b" / "sub_c"
    c.mkdir(parents=True)
    _write(a / "CLAUDE.md", "MARKER_AAA")
    _write(b / "CLAUDE.md", "MARKER_BBB")
    _write(c / "CLAUDE.md", "MARKER_CCC")

    # boundary = b, 从 c 走: 收集 c, b, 到 b 停 (不上到 a)
    result = collect_instructions(
        c,
        GroundConvention(upward_boundary=str(b)),
    )
    assert "MARKER_AAA" not in result
    assert "MARKER_BBB" in result
    assert "MARKER_CCC" in result


def test_workspace_root_as_fallback_boundary(tmp_path: Path) -> None:
    a = tmp_path
    b = tmp_path / "sub_b"
    c = tmp_path / "sub_b" / "sub_c"
    c.mkdir(parents=True)
    _write(a / "CLAUDE.md", "MARKER_AAA")
    _write(b / "CLAUDE.md", "MARKER_BBB")
    _write(c / "CLAUDE.md", "MARKER_CCC")

    # convention 无 boundary, 用 workspace_root=b 兜底
    result = collect_instructions(
        c,
        GroundConvention(),
        workspace_root=b,
    )
    assert "MARKER_AAA" not in result
    assert "MARKER_BBB" in result
    assert "MARKER_CCC" in result


def test_ground_outside_boundary_only_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    inside = tmp_path / "inside"
    outside.mkdir()
    inside.mkdir()
    _write(outside / "CLAUDE.md", "MARKER_OUT")

    # boundary=inside, root=outside → 只收 outside 本层
    result = collect_instructions(
        outside,
        GroundConvention(upward_boundary=str(inside)),
    )
    assert "MARKER_OUT" in result


def test_multiple_instruction_files(tmp_path: Path) -> None:
    _write(tmp_path / "CLAUDE.md", "MARKER_CLAUDE")
    _write(tmp_path / "AGENTS.md", "MARKER_AGENTS")
    result = collect_instructions(
        tmp_path,
        GroundConvention(
            instruction_files=("CLAUDE.md", "AGENTS.md"),
            upward_lookup=False,
        ),
    )
    assert "MARKER_CLAUDE" in result
    assert "MARKER_AGENTS" in result


def test_upward_lookup_false_ignores_ancestors(tmp_path: Path) -> None:
    outer = tmp_path
    inner = tmp_path / "sub"
    inner.mkdir()
    _write(outer / "CLAUDE.md", "MARKER_OUTER")
    _write(inner / "CLAUDE.md", "MARKER_INNER")

    result = collect_instructions(
        inner,
        GroundConvention(upward_lookup=False),
    )
    assert "MARKER_OUTER" not in result
    assert "MARKER_INNER" in result


def test_source_annotation_present(tmp_path: Path) -> None:
    _write(tmp_path / "CLAUDE.md", "content")
    result = collect_instructions(tmp_path, GroundConvention(upward_lookup=False))
    assert "from:" in result
    assert str(tmp_path) in result


def test_missing_instruction_file_names_ignored(tmp_path: Path) -> None:
    _write(tmp_path / "CLAUDE.md", "only-claude")
    result = collect_instructions(
        tmp_path,
        GroundConvention(
            instruction_files=("CLAUDE.md", "NON_EXISTENT.md", "OTHER.md"),
            upward_lookup=False,
        ),
    )
    assert "only-claude" in result
