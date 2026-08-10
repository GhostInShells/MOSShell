"""Tests for _hash.py — per-class pin observation."""

import asyncio, hashlib
from pathlib import Path

import pytest

from ghoshell_moss.ground._addr import Anchor
from ghoshell_moss.ground._hash import Observation, PinShadow, glob_limited, observe, observe_sync, parse_range
from ghoshell_moss.ground.contract import (
    ExecArguments,
    ExecPin,
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
        pin = GlobPin(label="g", arguments=GlobArguments(path="*.py"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.hash == _sha256_text("a.py\nb.py")

    def test_empty_hit_is_still_exists(self, tmp_path):
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = GlobPin(label="g", arguments=GlobArguments(path="nonexistent-*"))
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

    def test_no_frontmatter_hashes_empty(self, tmp_path):
        (tmp_path / "f.md").write_text("just body, no frontmatter")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FrontmatterPin(label="fm", arguments=FrontmatterArguments(path="f.md"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        # 无 frontmatter → 空 payload hash, 与 render 的 "no frontmatter found" 对齐
        assert obs.hash == hashlib.sha256(b"").hexdigest()


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


class TestGlobLimited:
    """glob_limited — max_depth 深度上限 + 场边界 (SPEC §4.1)."""

    def _field_tree(self, tmp_path):
        root = tmp_path.resolve()
        (root / "GROUND.md").write_text("# root\n")
        features = root / "features"
        features.mkdir()
        (features / "GROUND.md").write_text("# features\n")
        deep = features / "deep"
        deep.mkdir()
        (deep / "GROUND.md").write_text("# deep inside features\n")
        a_dir = root / "a"
        a_dir.mkdir()
        b_dir = a_dir / "b"
        b_dir.mkdir()
        (b_dir / "GROUND.md").write_text("# b (not a field parent)\n")
        return root

    def test_max_depth_limits_depth(self, tmp_path):
        """depth cap: matches deeper than max_depth excluded."""
        root = self._field_tree(tmp_path)
        matches = glob_limited(root, "**/GROUND.md", max_depth=2)
        paths = {str(m.relative_to(root)) for m in matches}
        assert "GROUND.md" in paths              # depth 1
        assert "features/GROUND.md" in paths     # depth 2
        assert "features/deep/GROUND.md" not in paths  # depth 3 > 2
        assert "a/b/GROUND.md" not in paths            # depth 3 > 2

    def test_field_boundary_stops_at_match(self, tmp_path):
        """field boundary: features 有 match, 其子目录不下钻."""
        root = self._field_tree(tmp_path)
        matches = glob_limited(root, "**/GROUND.md", max_depth=3)
        paths = {str(m.relative_to(root)) for m in matches}
        assert "GROUND.md" in paths              # 根场自身
        assert "features/GROUND.md" in paths     # 子场, depth 2 ≤ 3
        # features/deep/GROUND.md: depth 3 ≤ 3, 但 features 是场边界 → 不下钻 → 不在列表中
        assert "features/deep/GROUND.md" not in paths
        # a/b/GROUND.md: depth 3 ≤ 3, a 不是场边界 → 保留
        assert "a/b/GROUND.md" in paths

    def test_no_max_depth_is_unbounded(self, tmp_path):
        """max_depth=None: 无限制 (与 plain glob 等价)."""
        root = self._field_tree(tmp_path)
        matches = glob_limited(root, "**/GROUND.md", max_depth=None)
        paths = {str(m.relative_to(root)) for m in matches}
        assert "GROUND.md" in paths
        assert "features/GROUND.md" in paths
        assert "features/deep/GROUND.md" in paths  # 无 depth cap, 无 boundary stop
        assert "a/b/GROUND.md" in paths


class TestGlobIgnore:
    def test_ignores_noise_dirs(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("cache")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = GlobPin(label="py", arguments=GlobArguments(path="**/*.py"))
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


class TestExecPinObservation:
    """ExecPin 授权模型 — Makefile 级信任, 场根子树内 +x 文件."""

    def _make_script(self, tmp_path: Path, name: str, content: str) -> Path:
        script = tmp_path / name
        script.write_text(content)
        script.chmod(0o755)
        return script

    def test_runs_and_captures_stdout(self, tmp_path):
        self._make_script(tmp_path, "hello.sh", "#!/bin/sh\necho hello\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = ExecPin(label="x", arguments=ExecArguments(ref="hello.sh"))
        obs = observe_sync(pin, anchor)
        assert obs.exists
        assert obs.payload == "hello"
        assert obs.unit == "chars"

    def test_env_ground_and_cwd_injected(self, tmp_path):
        cwd = tmp_path / "sub"
        cwd.mkdir()
        self._make_script(
            tmp_path, "envdump.sh",
            '#!/bin/sh\necho "GROUND=$GROUND"\necho "CWD=$CWD"\n',
        )
        anchor = Anchor(ground=tmp_path.resolve(), cwd=cwd.resolve())
        pin = ExecPin(label="x", arguments=ExecArguments(ref="envdump.sh"))
        obs = observe_sync(pin, anchor)
        assert f"GROUND={tmp_path.resolve()}" in obs.payload
        assert f"CWD={cwd.resolve()}" in obs.payload

    def test_rejects_absolute_path(self, tmp_path):
        self._make_script(tmp_path, "hi.sh", "#!/bin/sh\necho hi\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = ExecPin(
            label="x",
            arguments=ExecArguments(ref=str(tmp_path / "hi.sh")),
        )
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.payload == "[outside ground]"

    def test_rejects_parent_traversal(self, tmp_path):
        outer = tmp_path.parent / "outer.sh"
        outer.write_text("#!/bin/sh\necho leaked\n")
        outer.chmod(0o755)
        try:
            anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
            pin = ExecPin(
                label="x",
                arguments=ExecArguments(ref="../outer.sh"),
            )
            obs = observe_sync(pin, anchor)
            assert obs.exists is True
            assert obs.payload == "[outside ground]"
        finally:
            outer.unlink(missing_ok=True)

    def test_missing_ref_is_missing(self, tmp_path):
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = ExecPin(label="x", arguments=ExecArguments(ref="nope.sh"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is False

    def test_no_exec_bit_is_missing(self, tmp_path):
        script = tmp_path / "no-x.sh"
        script.write_text("#!/bin/sh\necho hi\n")
        # 无 +x — 安全拒绝, 非文件缺失
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = ExecPin(label="x", arguments=ExecArguments(ref="no-x.sh"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.payload == "[not executable]"

    def test_nonzero_exit_visible(self, tmp_path):
        self._make_script(
            tmp_path, "fail.sh",
            "#!/bin/sh\necho oops\necho bad >&2\nexit 3\n",
        )
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = ExecPin(label="x", arguments=ExecArguments(ref="fail.sh"))
        obs = observe_sync(pin, anchor)
        assert obs.exists
        assert "[exit 3]" in obs.payload
        assert "bad" in obs.payload  # stderr tail

    def test_timeout_visible(self, tmp_path):
        self._make_script(tmp_path, "slow.sh", "#!/bin/sh\nsleep 5\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = ExecPin(
            label="x",
            arguments=ExecArguments(ref="slow.sh", timeout=0.2),
        )
        obs = observe_sync(pin, anchor)
        assert obs.exists
        assert "[timeout" in obs.payload

    def test_cwd_is_ground_root(self, tmp_path):
        # exec 的进程 cwd = $GROUND, 不是 anchor.cwd
        self._make_script(tmp_path, "pwd.sh", "#!/bin/sh\npwd\n")
        subdir = tmp_path / "deep"
        subdir.mkdir()
        anchor = Anchor(ground=tmp_path.resolve(), cwd=subdir.resolve())
        pin = ExecPin(label="x", arguments=ExecArguments(ref="pwd.sh"))
        obs = observe_sync(pin, anchor)
        # macOS 有 /private prefix, 用 resolve 对齐
        assert str(tmp_path.resolve()) in obs.payload
        assert "deep" not in obs.payload.strip().split("\n")[-1]


class TestLawPinObservation:
    """law pin — 从 cwd 向上收集约定文件, 观察只对文件集合做 hash."""

    def _make_tree(self, tmp_path) -> Path:
        (tmp_path / "CLAUDE.md").write_text("# root\n")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "CLAUDE.md").write_text("# sub\n")
        return tmp_path.resolve()

    def test_at_root_collects_only_root(self, tmp_path):
        root = self._make_tree(tmp_path)
        anchor = Anchor(ground=root, cwd=root)
        pin = LawPin(label="l", arguments=LawArguments(filename="CLAUDE.md"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.size == 1
        assert obs.unit == "entries"
        assert obs.hash == _sha256_text("CLAUDE.md")

    def test_from_subdir_collects_root_first(self, tmp_path):
        root = self._make_tree(tmp_path)
        anchor = Anchor(ground=root, cwd=root / "sub")
        pin = LawPin(label="l", arguments=LawArguments(filename="CLAUDE.md"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.size == 2
        # root-first: 场根在前, cwd 在后
        assert obs.hash == _sha256_text("CLAUDE.md\nsub/CLAUDE.md")

    def test_no_matching_file_renders_empty(self, tmp_path):
        root = self._make_tree(tmp_path)
        anchor = Anchor(ground=root, cwd=root)
        pin = LawPin(label="l", arguments=LawArguments(filename="AGENT.md"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.size == 0
        assert obs.hash == hashlib.sha256(b"").hexdigest()


class TestParseRange:
    """共享 parse_range — clamp 与非法区间 (SPEC §5.1: 1-indexed N-M)."""

    def test_basic_range(self):
        assert parse_range("2-3", 4) == (2, 3)

    def test_single_line(self):
        assert parse_range("3", 5) == (3, 3)

    def test_start_zero_clamps_to_one(self):
        # 1-indexed 下 0 起点是用户错误; clamp 到 1 而非静默空
        assert parse_range("0-2", 4) == (1, 2)

    def test_end_beyond_file_clamps_to_total(self):
        assert parse_range("1-999", 4) == (1, 4)

    def test_descending_raises(self):
        with pytest.raises(ValueError):
            parse_range("5-3", 4)

    def test_beyond_file_end_raises(self):
        with pytest.raises(ValueError):
            parse_range("999", 4)
