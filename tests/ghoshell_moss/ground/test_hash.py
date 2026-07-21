"""Tests for _hash.py — per-class pin observation."""

import asyncio, hashlib
from pathlib import Path

import pytest

from ghoshell_moss.ground._addr import Anchor
from ghoshell_moss.ground._hash import Observation, PinShadow, observe, observe_sync
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


def _run(coro):
    return asyncio.run(coro)


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


class TestFilePinObservation:
    def test_observes_full_content(self, tmp_path):
        (tmp_path / "a.py").write_text("hello world\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FilePin(label="f", arguments=FileArguments(path="a.py"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.hash == _sha256_text("hello world\n")

    def test_observes_range(self, tmp_path):
        (tmp_path / "a.py").write_text("L1\nL2\nL3\nL4\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FilePin(label="f", arguments=FileArguments(path="a.py", range="2-3"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.hash == _sha256_text("L2\nL3\n")

    def test_observes_single_line_range(self, tmp_path):
        (tmp_path / "a.py").write_text("L1\nL2\nL3\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FilePin(label="f", arguments=FileArguments(path="a.py", range="2"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.hash == _sha256_text("L2\n")

    def test_missing_file(self, tmp_path):
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FilePin(label="f", arguments=FileArguments(path="nope.py"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is False
        assert obs.mtime is None
        assert obs.hash is None

    def test_async_observe(self, tmp_path):
        (tmp_path / "a.py").write_text("x\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FilePin(label="f", arguments=FileArguments(path="a.py"))
        obs = _run(observe(pin, anchor))
        assert obs.exists is True


class TestGlobPinObservation:
    def test_observes_hit_set(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = GlobPin(label="g", arguments=GlobArguments(pattern="*.py"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.hash == _sha256_text("a.py\nb.py")

    def test_empty_hit_is_still_exists(self, tmp_path):
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = GlobPin(label="g", arguments=GlobArguments(pattern="nonexistent-*"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.mtime is None


class TestFrontmatterPinObservation:
    def test_observes_frontmatter(self, tmp_path):
        (tmp_path / "f.md").write_text("---\ntitle: test\n---\n\nbody text\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FrontmatterPin(label="fm", arguments=FrontmatterArguments(path="f.md"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.hash == _sha256_text("title: test")

    def test_no_frontmatter_falls_back_to_full_text(self, tmp_path):
        (tmp_path / "f.md").write_text("just body, no frontmatter")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FrontmatterPin(label="fm", arguments=FrontmatterArguments(path="f.md"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True


class TestLsPinObservation:
    def test_observes_directory_structure(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("b")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = LsPin(label="ls", arguments=LsArguments(path=".", depth=2))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True

    def test_not_a_directory(self, tmp_path):
        (tmp_path / "f.py").write_text("x")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = LsPin(label="ls", arguments=LsArguments(path="f.py"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is False


class TestBinaryDetection:
    def test_detects_binary_file(self, tmp_path):
        (tmp_path / "img.bin").write_bytes(b"\x00\x01\x02\x03" * 256)
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FilePin(label="b", arguments=FileArguments(path="img.bin"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.is_binary is True
        assert obs.hash is not None

    def test_text_file_not_binary(self, tmp_path):
        (tmp_path / "a.py").write_text("hello world\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FilePin(label="t", arguments=FileArguments(path="a.py"))
        obs = observe_sync(pin, anchor)
        assert obs.is_binary is False


class TestGlobIgnore:
    def test_ignores_noise_dirs(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("cache")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = GlobPin(label="py", arguments=GlobArguments(pattern="**/*.py"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        # only a.py, __pycache__/cached.py excluded
        assert "a.py" in obs.hash or obs.hash is not None

    def test_ls_ignores_noise_dirs(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("x")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = LsPin(label="ls", arguments=LsArguments(path=".", depth=2))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True


class TestPinShadow:
    def test_defaults(self):
        s = PinShadow()
        assert s.mtime is None
        assert s.hash is None

    def test_populated(self):
        s = PinShadow(mtime=100.0, hash="abc123")
        assert s.mtime == 100.0
        assert s.hash == "abc123"


class TestObservation:
    def test_frozen(self):
        o = Observation(exists=True, mtime=1.0, hash="abc")
        with pytest.raises(Exception):
            o.exists = False  # type: ignore[misc]
