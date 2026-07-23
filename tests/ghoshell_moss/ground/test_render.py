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
    _content_ls,
    _fmt_size,
    _render_result_block,
    render_context,
)
from ghoshell_moss.ground._hash import Observation, PinShadow
from ghoshell_moss.ground.contract import (
    FileArguments,
    FilePin,
    FrontmatterArguments,
    FrontmatterPin,
    GlobArguments,
    GlobPin,
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
        pin = GlobPin(label="g", arguments=GlobArguments(pattern="*.py"))
        result = _content_glob(pin, anchor)
        assert "mtime" not in result
        assert "B" in result or "K" in result

    def test_limit_truncation(self, anchor):
        pin = GlobPin(label="g", arguments=GlobArguments(pattern="*.py", limit=1))
        result = _content_glob(pin, anchor)
        assert "showing 1 of 2" in result

    def test_no_matches(self, anchor):
        pin = GlobPin(label="g", arguments=GlobArguments(pattern="*.rs"))
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
        result = run(render_context(
            body="", pins=[pin], shadows={}, anchor=anchor,
        ))
        assert "[truncated at 50 chars]" in result
        assert "ground:pin:readme" in result
