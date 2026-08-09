"""Tests for _render.py — budget truncation, human-readable sizes,
multi-file frontmatter, result blocks.
"""

import asyncio
from pathlib import Path

import pytest

from ghoshell_moss.ground._addr import Anchor
from ghoshell_moss.ground._render import (
    _apply_budget,
    _content_file,
    _content_frontmatter,
    _content_glob,
    _content_law,
    _content_ls,
    _fmt_size,
    _render_result_block,
    render_context,
    render_walk,
)
from ghoshell_moss.ground._hash import Observation, PinShadow
from ghoshell_moss.ground.contract import (
    FileArguments,
    FilePin,
    FrontmatterArguments,
    FrontmatterPin,
    GlobArguments,
    GlobPin,
    LawArguments,
    LawPin,
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


# -- result block ----------------------------------------------------------


class TestRenderResultBlock:
    @pytest.fixture
    def anchor(self, tmp_path):
        (tmp_path / "hello.txt").write_text("hello")
        return Anchor(ground=tmp_path, cwd=tmp_path)

    def test_stale_mark(self, anchor):
        pin = FilePin(label="f", arguments=FileArguments(path="hello.txt"))
        obs = Observation(exists=True, hash="abc123")
        result = _render_result_block(pin, obs, stale=True, missing=False, anchor=anchor)
        assert "[changed on disk]" in result
        assert "hello" in result

    def test_missing_mark(self, anchor):
        pin = FilePin(label="f", arguments=FileArguments(path="nonexistent.py"))
        obs = Observation(exists=False)
        result = _render_result_block(pin, obs, stale=False, missing=True, anchor=anchor)
        assert "[missing]" in result


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


# -- law pin rendering ------------------------------------------------------


class TestLawPinRender:
    """law pin — 文件名向上收集, root-first, 一层 @-展开, 截断."""

    def _tree(self, tmp_path) -> Path:
        (tmp_path / "CLAUDE.md").write_text("# root\n\nroot body\n")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "CLAUDE.md").write_text("# sub\n\nsub body\n")
        return tmp_path.resolve()

    def test_at_root_renders_only_root(self, tmp_path):
        root = self._tree(tmp_path)
        anchor = Anchor(ground=root, cwd=root)
        pin = LawPin(label="l", arguments=LawArguments(filename="CLAUDE.md"))
        out = _content_law(pin, anchor)
        assert "-- CLAUDE.md" in out
        assert "root body" in out
        assert "sub body" not in out

    def test_walk_renders_root_first_chain(self, tmp_path):
        root = self._tree(tmp_path)
        anchor = Anchor(ground=root, cwd=root / "sub")
        pin = LawPin(label="l", arguments=LawArguments(filename="CLAUDE.md"))
        out = _content_law(pin, anchor)
        # 父级向下: root 块在前, cwd 块在后
        assert out.index("-- CLAUDE.md") < out.index("-- sub/CLAUDE.md")
        assert "root body" in out
        assert "sub body" in out

    def test_no_matching_file(self, tmp_path):
        root = self._tree(tmp_path)
        anchor = Anchor(ground=root, cwd=root)
        pin = LawPin(label="l", arguments=LawArguments(filename="AGENT.md"))
        assert _content_law(pin, anchor) == "(no files)"

    def test_one_level_at_expansion(self, tmp_path):
        root = tmp_path.resolve()
        (root / "CLAUDE.md").write_text("# root\n\nsee @notes.md here\n")
        (root / "notes.md").write_text("NOTES CONTENT\n")
        anchor = Anchor(ground=root, cwd=root)
        pin = LawPin(label="l", arguments=LawArguments(filename="CLAUDE.md"))
        # @-ref 在 children 里, 不在 content 里展开
        from ghoshell_moss.ground._render import _build_law_with_at
        content, children = _build_law_with_at(pin, anchor)
        assert "@notes.md" in content
        assert any("NOTES CONTENT" in c.content for c in children)

    def test_lines_cap(self, tmp_path):
        root = tmp_path.resolve()
        (root / "CLAUDE.md").write_text("l1\nl2\nl3\nl4\n")
        anchor = Anchor(ground=root, cwd=root)
        pin = LawPin(label="l", arguments=LawArguments(filename="CLAUDE.md", lines=2))
        items = run(render_context(
            body="", pins=[pin], shadows={}, anchor=anchor,
        ))
        assert len(items) == 1
        assert items[0].truncated
        assert "[truncated at 2 lines]" in items[0].content

    def test_budget_cap(self, tmp_path):
        root = tmp_path.resolve()
        (root / "CLAUDE.md").write_text("a" * 100)
        anchor = Anchor(ground=root, cwd=root)
        pin = LawPin(label="l", arguments=LawArguments(filename="CLAUDE.md", budget=30))
        items = run(render_context(
            body="", pins=[pin], shadows={}, anchor=anchor,
        ))
        assert items[0].truncated
        assert "[truncated at 30 chars]" in items[0].content

    def test_walk_expands_law_pin(self, tmp_path):
        root = tmp_path.resolve()
        (root / "GROUND.md").write_text("# g\n")
        (root / "CLAUDE.md").write_text("# root law\n")
        (root / "sub").mkdir()
        (root / "sub" / "CLAUDE.md").write_text("# sub law\n")
        pin = LawPin(label="l", arguments=LawArguments(filename="CLAUDE.md"))
        items = run(render_walk(
            cwd=root / "sub",
            ground_root=root,
            doc_path=root / "GROUND.md",
            pins=[pin],
            shadows={},
        ))
        # law pin 是位置依赖 pin — walk 时展开为 pin item, 非折叠 TOC
        pin_items = [i for i in items if i.kind == "law"]
        assert len(pin_items) == 1
        assert "sub law" in pin_items[0].content


# -- frame rendering (integration) -----------------------------------------


class TestFrameBudget:
    @pytest.fixture
    def anchor(self, tmp_path):
        (tmp_path / "readme.md").write_text("# Hello\n\nThis is a test file.\n" * 100)
        return Anchor(ground=tmp_path, cwd=tmp_path)

    def test_file_pin_truncated_in_frame(self, anchor):
        pin = FilePin(
            label="readme",
            arguments=FileArguments(path="readme.md", budget=50),
        )
        items = run(render_context(
            body="", pins=[pin], shadows={}, anchor=anchor,
        ))
        assert items[0].truncated
        assert "[truncated at 50 chars]" in items[0].content
