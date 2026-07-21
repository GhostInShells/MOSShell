"""Tests for _addr.py — anchor path resolution."""

import pytest
from pathlib import Path

from ghoshell_moss.ground._addr import Anchor, resolve_path, is_glob_pattern
from ghoshell_moss.ground.contract import PathOutsideRootError


class TestAnchor:
    def test_anchor_creation(self):
        a = Anchor(ground=Path("/tmp/a"), cwd=Path("/tmp/b"))
        assert a.ground == Path("/tmp/a")
        assert a.cwd == Path("/tmp/b")

    def test_anchor_frozen(self):
        a = Anchor(ground=Path("/tmp/a"), cwd=Path("/tmp/b"))
        with pytest.raises(Exception):
            a.ground = Path("/tmp/c")  # type: ignore[misc]


class TestResolvePath:
    def test_bare_relative_defaults_to_ground(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        resolved = resolve_path("a.py", anchor)
        assert resolved == (tmp_path / "a.py").resolve()

    def test_explicit_ground_anchor(self, tmp_path):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "f.py").write_text("x")
        anchor = Anchor(ground=(tmp_path / "sub").resolve(), cwd=tmp_path.resolve())
        resolved = resolve_path("$GROUND/f.py", anchor)
        assert resolved == (tmp_path / "sub" / "f.py").resolve()

    def test_explicit_cwd_anchor(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "f.py").write_text("x")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=sub.resolve())
        resolved = resolve_path("$CWD/f.py", anchor)
        assert resolved == (sub / "f.py").resolve()

    def test_home_anchor(self, tmp_path):
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        resolved = resolve_path("$HOME/.bashrc", anchor)
        assert resolved.name == ".bashrc"

    def test_escaped_dollar(self, tmp_path):
        (tmp_path / "$config.json").write_text("{}")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        resolved = resolve_path("\\$config.json", anchor)
        assert resolved == (tmp_path / "$config.json").resolve()

    def test_escape_via_dotdot_rejected(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        anchor = Anchor(ground=sub.resolve(), cwd=sub.resolve())
        with pytest.raises(PathOutsideRootError):
            resolve_path("../escape.py", anchor)

    def test_escape_via_symlink(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "real").mkdir()
        (sub / "link").symlink_to(tmp_path / "real")
        anchor = Anchor(ground=sub.resolve(), cwd=sub.resolve())
        with pytest.raises(PathOutsideRootError):
            resolve_path("link/../real", anchor)

    def test_bare_absolute_path_rejected(self, tmp_path):
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        with pytest.raises(PathOutsideRootError, match="bare absolute"):
            resolve_path("/etc/passwd", anchor)

    def test_anchor_without_slash_rejected(self, tmp_path):
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        with pytest.raises(PathOutsideRootError, match="missing path separator"):
            resolve_path("$GROUND", anchor)

    def test_nonexistent_file_still_resolves(self, tmp_path):
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        resolved = resolve_path("nonexistent.py", anchor)
        assert resolved == (tmp_path / "nonexistent.py").resolve()


class TestIsGlobPattern:
    def test_star_is_glob(self):
        assert is_glob_pattern("*.py")

    def test_double_star_is_glob(self):
        assert is_glob_pattern("**/*.py")

    def test_bracket_is_glob(self):
        assert is_glob_pattern("[abc].py")

    def test_question_is_glob(self):
        assert is_glob_pattern("file?.py")

    def test_plain_path_is_not_glob(self):
        assert not is_glob_pattern("src/main.py")

    def test_dollar_sign_not_glob(self):
        assert not is_glob_pattern("$GROUND/config.yaml")
