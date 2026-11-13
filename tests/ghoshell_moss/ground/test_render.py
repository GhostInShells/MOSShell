"""Tests for _render.py — budget truncation, human-readable sizes,
multi-file frontmatter, result blocks.
"""

import asyncio
from pathlib import Path

import pytest

from ghoshell_moss.ground._addr import Anchor
from ghoshell_moss.ground._render import (
    _apply_budget,
    _build_content_with_at,
    _content_file,
    _content_frontmatter,
    _content_glob,
    _content_ls,
    _fmt_size,
)
from ghoshell_moss.ground._hash import Observation
from ghoshell_moss.ground.contract import (
    FileArguments,
    FilePin,
    FrontmatterArguments,
    FrontmatterPin,
    GlobArguments,
    GlobPin,
    ContentArguments,
    ContentPin,
    LsArguments,
    LsPin,
)


def run(coro):
    return asyncio.run(coro)


# -- size formatting -------------------------------------------------------


class TestFmtSize:
    def test_bytes(self):
        assert _fmt_size(0) == "0B"
        assert _fmt_size(500) == "500B"
        assert _fmt_size(1023) == "1023B"

    def test_kilobytes(self):
        assert _fmt_size(1024) == "1K"
        assert _fmt_size(1536) == "2K"
        assert _fmt_size(1024 * 50) == "50K"

    def test_megabytes(self):
        assert _fmt_size(1024 * 1024) == "1.0M"
        assert _fmt_size(int(1.5 * 1024 * 1024)) == "1.5M"

    def test_gigabytes(self):
        assert _fmt_size(1024 * 1024 * 1024) == "1.0G"


# -- budget truncation -----------------------------------------------------


class TestApplyBudget:
    def test_no_budget(self):
        assert _apply_budget("hello world", None) == "hello world"

    def test_within_budget(self):
        assert _apply_budget("short", budget=100) == "short"

    def test_exceeds_budget(self):
        result = _apply_budget("hello world!", budget=8)
        assert result == "hello wo\n[truncated at 8 chars]"

    def test_exact_budget(self):
        result = _apply_budget("12345678", budget=8)
        assert result == "12345678"


# -- glob content (size formatting, no mtime) ------------------------------


class TestContentGlob:
    @pytest.fixture
    def anchor(self, tmp_path):
        # create test files
        (tmp_path / "a.py").write_text("x")
        (tmp_path / "b.py").write_text("y" * 100)
        return Anchor(ground=tmp_path, cwd=tmp_path)

    def test_no_mtime_in_output(self, anchor):
        pin = GlobPin(label="g", arguments=GlobArguments(path="*.py"))
        result = _content_glob(pin, anchor)
        assert "mtime" not in result
        assert "B" in result or "K" in result

    def test_limit_truncation(self, anchor):
        pin = GlobPin(label="g", arguments=GlobArguments(path="*.py", limit=1))
        result = _content_glob(pin, anchor)
        assert "showing 1 of 2" in result

    def test_no_matches(self, anchor):
        pin = GlobPin(label="g", arguments=GlobArguments(path="*.rs"))
        result = _content_glob(pin, anchor)
        assert result == "(no matches)"


# -- ls content ------------------------------------------------------------


class TestContentLs:
    @pytest.fixture
    def anchor(self, tmp_path):
        (tmp_path / "a.md").write_text("a")
        (tmp_path / ".ghost.log").write_text("noise")
        (tmp_path / ".moss").mkdir()
        (tmp_path / ".moss" / "b.log").write_text("noise")
        return Anchor(ground=tmp_path, cwd=tmp_path)

    def test_ignore_filters_files_and_dirs(self, anchor):
        from pathspec import PathSpec

        spec = PathSpec.from_lines("gitignore", [".moss/", "*.log"])
        pin = LsPin(label="ls", arguments=LsArguments(path=".", depth=2))
        result = _content_ls(pin, anchor, ignore=spec)
        assert "a.md" in result
        assert ".ghost.log" not in result
        assert ".moss" not in result

    def test_no_ignore_lists_everything(self, anchor):
        pin = LsPin(label="ls", arguments=LsArguments(path=".", depth=2))
        result = _content_ls(pin, anchor, ignore=None)
        assert ".ghost.log" in result
        assert ".moss" in result


# -- frontmatter pattern mode ----------------------------------------------


class TestContentFrontmatterPattern:
    @pytest.fixture
    def anchor(self, tmp_path):
        (tmp_path / "a").mkdir()
        (tmp_path / "b").mkdir()
        (tmp_path / "a" / "GROUND.md").write_text(
            "---\n$id: moss:a\nlabel: A\n---\n\nbody a\n"
        )
        (tmp_path / "b" / "GROUND.md").write_text(
            "---\n$id: moss:b\nlabel: B\n---\n\nbody b\n"
        )
        return Anchor(ground=tmp_path, cwd=tmp_path)

    def test_pattern_renders_multiple(self, anchor):
        pin = FrontmatterPin(
            label="children", arguments=FrontmatterArguments(path="*/GROUND.md")
        )
        result = _content_frontmatter(pin, anchor)
        assert "moss:a" in result
        assert "moss:b" in result

    def test_pattern_limit(self, anchor):
        pin = FrontmatterPin(
            label="children",
            arguments=FrontmatterArguments(path="*/GROUND.md", limit=1),
        )
        result = _content_frontmatter(pin, anchor)
        assert "showing 1 of 2" in result

    def test_pattern_budget_truncation(self, anchor):
        pin = FrontmatterPin(
            label="children",
            arguments=FrontmatterArguments(path="*/GROUND.md", budget=60),
        )
        result = _content_frontmatter(pin, anchor)
        assert len(result) <= 100  # budget + truncation marker

    def test_pattern_no_matches(self, anchor):
        pin = FrontmatterPin(
            label="none", arguments=FrontmatterArguments(path="*/nonexistent.md")
        )
        result = _content_frontmatter(pin, anchor)
        assert result == "(no matches)"

    def test_pattern_keys_filter(self, anchor):
        pin = FrontmatterPin(
            label="ids",
            arguments=FrontmatterArguments(path="*/GROUND.md", keys=["$id"]),
        )
        result = _content_frontmatter(pin, anchor)
        assert "$id: moss:a" in result
        assert "label" not in result


# -- single-file frontmatter -----------------------------------------------


class TestContentFrontmatterSingle:
    @pytest.fixture
    def anchor(self, tmp_path):
        (tmp_path / "test.md").write_text(
            "---\n$id: x\nlabel: X\n---\n\nbody here\n"
        )
        return Anchor(ground=tmp_path, cwd=tmp_path)

    def test_single_file(self, anchor):
        pin = FrontmatterPin(
            label="fm", arguments=FrontmatterArguments(path="test.md")
        )
        result = _content_frontmatter(pin, anchor)
        assert "$id: x" in result
        assert "body here" not in result  # body excluded

    def test_keys_filter_single(self, anchor):
        pin = FrontmatterPin(
            label="fm", arguments=FrontmatterArguments(path="test.md", keys=["$id"])
        )
        result = _content_frontmatter(pin, anchor)
        assert "$id: x" in result
        assert "label" not in result

    def test_no_frontmatter(self, anchor, tmp_path):
        (tmp_path / "plain.md").write_text("just text")
        pin = FrontmatterPin(
            label="fm", arguments=FrontmatterArguments(path="plain.md")
        )
        result = _content_frontmatter(pin, anchor)
        assert "error" in result


# -- content pin rendering ------------------------------------------------------


class TestContentPinRender:
    """content pin — 文件名向上收集, root-first, 一层 @-展开, 截断."""

    def _tree(self, tmp_path) -> Path:
        (tmp_path / "CLAUDE.md").write_text("# root\n\nroot body\n")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "CLAUDE.md").write_text("# sub\n\nsub body\n")
        return tmp_path.resolve()

    def test_at_root_renders_only_root(self, tmp_path):
        root = self._tree(tmp_path)
        anchor = Anchor(ground=root, cwd=root)
        pin = ContentPin(label="l", arguments=ContentArguments(filename="CLAUDE.md"))
        out = _build_content_with_at(pin, anchor)[0]
        assert "-- CLAUDE.md" in out
        assert "root body" in out
        assert "sub body" not in out

    def test_walk_renders_root_first_chain(self, tmp_path):
        root = self._tree(tmp_path)
        anchor = Anchor(ground=root, cwd=root / "sub")
        pin = ContentPin(label="l", arguments=ContentArguments(filename="CLAUDE.md"))
        out = _build_content_with_at(pin, anchor)[0]
        # 父级向下: root 块在前, cwd 块在后
        assert out.index("-- CLAUDE.md") < out.index("-- sub/CLAUDE.md")
        assert "root body" in out
        assert "sub body" in out

    def test_no_matching_file(self, tmp_path):
        root = self._tree(tmp_path)
        anchor = Anchor(ground=root, cwd=root)
        pin = ContentPin(label="l", arguments=ContentArguments(filename="AGENT.md"))
        assert _build_content_with_at(pin, anchor)[0] == "(no files)"

    def test_one_level_at_expansion(self, tmp_path):
        root = tmp_path.resolve()
        (root / "CLAUDE.md").write_text("# root\n\nsee @notes.md here\n")
        (root / "notes.md").write_text("NOTES CONTENT\n")
        anchor = Anchor(ground=root, cwd=root)
        pin = ContentPin(label="l", arguments=ContentArguments(filename="CLAUDE.md"))
        # @-ref 在 children 里, 不在 content 里展开
        content, children = _build_content_with_at(pin, anchor)
        assert "@notes.md" in content
        assert any("NOTES CONTENT" in c.content for c in children)

    def test_lines_cap(self, tmp_path):
        from ghoshell_moss.ground._render import render_position

        (tmp_path / "GROUND.md").write_text(
            "---\nname: g\npins:\n- label: l\n  verb: content\n"
            "  arguments:\n    filename: CLAUDE.md\n    lines: 2\n---\n\nbody\n"
        )
        (tmp_path / "CLAUDE.md").write_text("l1\nl2\nl3\nl4\n")
        out = asyncio.run(render_position(tmp_path, boundary=tmp_path))
        assert "[truncated at 2 lines]" in out

    def test_budget_cap(self, tmp_path):
        from ghoshell_moss.ground._render import render_position

        (tmp_path / "GROUND.md").write_text(
            "---\nname: g\npins:\n- label: l\n  verb: content\n"
            "  arguments:\n    filename: CLAUDE.md\n    budget: 30\n---\n\nbody\n"
        )
        (tmp_path / "CLAUDE.md").write_text("a" * 100)
        out = asyncio.run(render_position(tmp_path, boundary=tmp_path))
        assert "[truncated at 30 chars]" in out

    def test_walk_expands_positional_and_folds_ground_pins(self, tmp_path):
        """walk 面: $CWD/内容 pin 展开, $GROUND 锚的 pin 折叠成 TOC (SPEC §6.3)."""
        from ghoshell_moss.ground._render import render_position

        root = tmp_path.resolve()
        (root / "GROUND.md").write_text(
            "---\nname: g\npins:\n"
            "- label: here\n  verb: file\n  arguments:\n    path: $CWD/CLAUDE.md\n"
            "- label: rootonly\n  verb: file\n  arguments:\n    path: GROUND.md\n"
            "---\n\nbody\n"
        )
        (root / "CLAUDE.md").write_text("# root content\n")
        (root / "sub").mkdir()
        (root / "sub" / "CLAUDE.md").write_text("# sub content\n")

        out = asyncio.run(render_position(root / "sub", boundary=root))

        assert "# sub content" in out          # positional pin 展开
        assert "<folded>" in out
        assert 'label="rootonly"' in out        # 场锚 pin 只留一行
        assert "<body>" not in out              # walk 不重发正文


# -- frame rendering (integration) -----------------------------------------


class TestFrameBudget:
    def test_file_pin_truncated_in_frame(self, tmp_path):
        from ghoshell_moss.ground._render import render_position

        (tmp_path / "readme.md").write_text("# Hello\n\nThis is a test file.\n" * 100)
        (tmp_path / "GROUND.md").write_text(
            "---\nname: g\npins:\n- label: readme\n  verb: file\n"
            "  arguments:\n    path: readme.md\n    budget: 50\n---\n\nbody\n"
        )
        out = asyncio.run(render_position(tmp_path, boundary=tmp_path))
        assert "[truncated at 50 chars]" in out


# -- position rendering (SPEC 0.3.0) -----------------------------------------


def _write_ground(root: Path, text: str) -> None:
    (root / "GROUND.md").write_text(text, encoding="utf-8")


def test_render_position_honors_ground_ignore(tmp_path: Path):
    """ground 级的 ignore 必须真的生效 —— 这是渲染路径里唯一会构造 PathSpec 的
    分支, 一个只在 TYPE_CHECKING 里导入的 PathSpec 曾在这里抛 NameError。"""
    from ghoshell_moss.ground._render import render_position

    (tmp_path / "seen.txt").write_text("x", encoding="utf-8")
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "hidden.txt").write_text("y", encoding="utf-8")
    _write_ground(
        tmp_path,
        "---\nname: g\nignore:\n- secret/\npins:\n"
        "- label: all\n  verb: glob\n  arguments:\n    path: '**/*'\n---\n\nbody\n",
    )

    out = asyncio.run(render_position(tmp_path, boundary=tmp_path))

    assert "seen.txt" in out
    assert "hidden.txt" not in out


def test_render_position_names_only_the_inlined_children(tmp_path: Path):
    """`inline` 点名的子场递归渲染, 未点名的不出现 (SPEC §3.1/§6.6)."""
    from ghoshell_moss.ground._render import render_position

    for name in ("shown", "hidden"):
        child = tmp_path / name
        child.mkdir()
        _write_ground(child, f"---\nname: {name}\n---\n\n{name} body\n")
    _write_ground(
        tmp_path,
        "---\nname: root\ninline:\n- shown\n---\n\nroot body\n",
    )

    out = asyncio.run(render_position(tmp_path, boundary=tmp_path))

    assert 'name="shown"' in out
    assert 'name="hidden"' not in out


def test_render_position_walk_face_has_cwd_and_no_body(tmp_path: Path):
    """walk 面: 位置在场内子目录, 出 <cwd>, 不出 body (SPEC §6.3)."""
    from ghoshell_moss.ground._render import render_position

    inner = tmp_path / "inner"
    inner.mkdir()
    _write_ground(tmp_path, "---\nname: root\n---\n\nROOT BODY\n")

    out = asyncio.run(render_position(inner, boundary=tmp_path))

    assert f'<cwd path="{inner.resolve()}"/>' in out
    assert "ROOT BODY" not in out


def test_render_position_cycle_is_a_result_not_a_hang(tmp_path: Path):
    """A inline B, B inline A: 环渲染成 truncated, 不是无限递归 (SPEC §6.6)."""
    from ghoshell_moss.ground._render import render_position

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    _write_ground(a, "---\nname: a\ninline:\n- ../b\n---\n\na\n")
    _write_ground(b, "---\nname: b\ninline:\n- ../a\n---\n\nb\n")

    out = asyncio.run(render_position(a, boundary=tmp_path))

    assert 'reason="cycle"' in out
