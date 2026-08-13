"""Tests for _hash.py — per-class pin observation."""

import asyncio
from pathlib import Path

import pytest

from ghoshell_moss.ground._addr import Anchor
from ghoshell_moss.ground._hash import Observation, glob_limited, observe, observe_sync, parse_range
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


class TestFilePinObservation:
    def test_observes_full_content(self, tmp_path):
        (tmp_path / "a.py").write_text("hello world\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FilePin(label="f", arguments=FileArguments(path="a.py"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.size == 12
        assert obs.unit == "B"

    def test_observes_range(self, tmp_path):
        # observe 不读内容 — range 切片是 render 的事, size 报全文件字节
        (tmp_path / "a.py").write_text("L1\nL2\nL3\nL4\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FilePin(label="f", arguments=FileArguments(path="a.py", range="2-3"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.size == 12  # 全文件 "L1\nL2\nL3\nL4\n" = 12 bytes
        assert obs.unit == "B"

    def test_observes_single_line_range(self, tmp_path):
        (tmp_path / "a.py").write_text("L1\nL2\nL3\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FilePin(label="f", arguments=FileArguments(path="a.py", range="2"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.size == 9  # 全文件 "L1\nL2\nL3\n" = 9 bytes
        assert obs.unit == "B"

    def test_missing_file(self, tmp_path):
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FilePin(label="f", arguments=FileArguments(path="nope.py"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is False

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
        assert obs.size == 2
        assert obs.unit == "entries"

    def test_empty_hit_is_still_exists(self, tmp_path):
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = GlobPin(label="g", arguments=GlobArguments(path="nonexistent-*"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.size == 0


class TestFrontmatterPinObservation:
    def test_observes_frontmatter(self, tmp_path):
        (tmp_path / "f.md").write_text("---\ntitle: test\n---\n\nbody text\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FrontmatterPin(label="fm", arguments=FrontmatterArguments(path="f.md"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.size == 1
        assert obs.unit == "entries"

    def test_no_frontmatter_still_observes(self, tmp_path):
        (tmp_path / "f.md").write_text("just body, no frontmatter")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FrontmatterPin(label="fm", arguments=FrontmatterArguments(path="f.md"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.size == 1


class TestLsPinObservation:
    def test_observes_directory_structure(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "b.py").write_text("b")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = LsPin(label="ls", arguments=LsArguments(path=".", depth=2))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.size == 3  # a.py + sub/ + sub/b.py
        assert obs.unit == "entries"

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
        assert obs.size == 1024

    def test_text_file_not_binary(self, tmp_path):
        (tmp_path / "a.py").write_text("hello world\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FilePin(label="t", arguments=FileArguments(path="a.py"))
        obs = observe_sync(pin, anchor)
        assert obs.is_binary is False


class TestGlobLimited:
    """glob_limited — 显式递归, recursion (深度上限) + stop_on_match (防穿透) 正交 (SPEC §4.1)."""

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

    def test_recursion_limits_depth(self, tmp_path):
        """recursion = 目录层数: 1 = 一层子场 (直觉语义, 非 filename-inclusive)."""
        root = self._field_tree(tmp_path)
        matches = glob_limited(root, "**/GROUND.md", recursion=1)
        paths = {str(m.relative_to(root)) for m in matches}
        assert "GROUND.md" in paths              # 根场自身
        assert "features/GROUND.md" in paths     # 一层子场
        assert "features/deep/GROUND.md" not in paths  # 两层 > 1
        assert "a/b/GROUND.md" not in paths            # 两层 > 1

    def test_recursion_two_reaches_two_levels(self, tmp_path):
        root = self._field_tree(tmp_path)
        matches = glob_limited(root, "**/GROUND.md", recursion=2)
        paths = {str(m.relative_to(root)) for m in matches}
        assert "features/deep/GROUND.md" in paths  # 两层 ≤ 2, 无 stop_on_match → 穿透
        assert "a/b/GROUND.md" in paths

    def test_stop_on_match_is_field_boundary(self, tmp_path):
        """防穿透: features 直接含 GROUND.md 是场边界, 其子目录不下钻."""
        root = self._field_tree(tmp_path)
        matches = glob_limited(root, "**/GROUND.md", stop_on_match=True)
        paths = {str(m.relative_to(root)) for m in matches}
        assert "GROUND.md" in paths              # 根场自身
        assert "features/GROUND.md" in paths     # 子场
        # features/deep/GROUND.md: features 是场边界 → 不下钻
        assert "features/deep/GROUND.md" not in paths
        # a/b/GROUND.md: a 不是场边界 → 保留
        assert "a/b/GROUND.md" in paths

    def test_no_recursion_is_unbounded(self, tmp_path):
        """recursion=None: 无限制 (与 plain glob 等价)."""
        root = self._field_tree(tmp_path)
        matches = glob_limited(root, "**/GROUND.md", recursion=None)
        paths = {str(m.relative_to(root)) for m in matches}
        assert "GROUND.md" in paths
        assert "features/GROUND.md" in paths
        assert "features/deep/GROUND.md" in paths  # 无 depth cap, 无 boundary stop
        assert "a/b/GROUND.md" in paths

    def test_star_prefix_does_not_bypass_boundary(self, tmp_path):
        """`*/**/GROUND.md`: `*` 前缀不豁免边界 — 根自身排除, 子场仍防穿透."""
        root = self._field_tree(tmp_path)
        matches = glob_limited(root, "*/**/GROUND.md", stop_on_match=True)
        paths = {str(m.relative_to(root)) for m in matches}
        assert "GROUND.md" not in paths            # `*` 要求 ≥1 层, 排除根自身
        assert "features/GROUND.md" in paths       # 子场
        assert "features/deep/GROUND.md" not in paths  # features 是边界
        assert "a/b/GROUND.md" in paths            # 穿过非 ground 目录


class TestGlobIgnore:
    def test_ignores_noise_dirs(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "cached.py").write_text("cache")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = GlobPin(label="py", arguments=GlobArguments(path="**/*.py"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        # only a.py, __pycache__/cached.py excluded by GLOB_IGNORE
        assert obs.size == 1

    def test_ls_ignores_noise_dirs(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("x")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = LsPin(label="ls", arguments=LsArguments(path=".", depth=2))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True


class TestObservation:
    def test_frozen(self):
        o = Observation(exists=True)
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
    """law pin — 从 cwd 向上收集约定文件."""

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

    def test_from_subdir_collects_root_first(self, tmp_path):
        root = self._make_tree(tmp_path)
        anchor = Anchor(ground=root, cwd=root / "sub")
        pin = LawPin(label="l", arguments=LawArguments(filename="CLAUDE.md"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.size == 2

    def test_no_matching_file_renders_empty(self, tmp_path):
        root = self._make_tree(tmp_path)
        anchor = Anchor(ground=root, cwd=root)
        pin = LawPin(label="l", arguments=LawArguments(filename="AGENT.md"))
        obs = observe_sync(pin, anchor)
        assert obs.exists is True
        assert obs.size == 0


class TestGroundIgnore:
    """Ground-level ignore — pathspec filter 与 GLOB_IGNORE 叠层."""

    @staticmethod
    def _spec(patterns: list[str]) -> object:
        from pathspec import PathSpec
        return PathSpec.from_lines("gitignore", patterns)

    def test_glob_limited_filters_ignored_paths(self, tmp_path):
        (tmp_path / "keep.py").write_text("a")
        (tmp_path / ".moss").mkdir()
        (tmp_path / ".moss" / "noise.py").write_text("b")
        spec = self._spec([".moss/"])
        matches = glob_limited(tmp_path, "**/*.py", ignore=spec)
        rels = {str(m.relative_to(tmp_path)) for m in matches}
        assert "keep.py" in rels
        assert ".moss/noise.py" not in rels

    def test_glob_limited_ignore_no_effect_when_none(self, tmp_path):
        (tmp_path / "a.py").write_text("x")
        (tmp_path / ".moss").mkdir()
        (tmp_path / ".moss" / "b.py").write_text("y")
        matches = glob_limited(tmp_path, "**/*.py", ignore=None)
        rels = {str(m.relative_to(tmp_path)) for m in matches}
        assert "a.py" in rels
        # GLOB_IGNORE doesn't include .moss, so it passes through
        assert ".moss/b.py" in rels

    def test_glob_ignore_via_observe_sync(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / ".moss").mkdir()
        (tmp_path / ".moss" / "b.py").write_text("b")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = GlobPin(label="g", arguments=GlobArguments(path="**/*.py"))
        spec = self._spec([".moss/"])
        obs = observe_sync(pin, anchor, ignore=spec)
        assert obs.exists is True
        assert obs.size == 1  # only a.py — .moss/ tree excluded

    def test_frontmatter_pattern_ignore(self, tmp_path):
        (tmp_path / "a.md").write_text("---\nid: a\n---\nbody\n")
        (tmp_path / ".moss").mkdir()
        (tmp_path / ".moss" / "b.md").write_text("---\nid: b\n---\nbody\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = FrontmatterPin(label="fm", arguments=FrontmatterArguments(path="**/*.md", keys=["id"]))
        spec = self._spec([".moss/"])
        obs = observe_sync(pin, anchor, ignore=spec)
        assert obs.exists is True
        assert obs.size == 1  # only a.md, .moss/b.md excluded

    def test_ls_ignore_skips_ignored_dirs(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / ".moss").mkdir()
        (tmp_path / ".moss" / "b.py").write_text("b")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = LsPin(label="ls", arguments=LsArguments(path=".", depth=2))
        spec = self._spec([".moss/"])
        obs = observe_sync(pin, anchor, ignore=spec)
        assert obs.exists is True
        assert obs.size == 1  # only a.py — .moss/ tree excluded

    def test_ignore_file_merged(self, tmp_path):
        (tmp_path / "keep.py").write_text("a")
        (tmp_path / "skip_me").mkdir()
        (tmp_path / "skip_me" / "b.py").write_text("b")
        (tmp_path / ".groundignore").write_text("skip_me/\n")
        anchor = Anchor(ground=tmp_path.resolve(), cwd=tmp_path.resolve())
        pin = GlobPin(label="g", arguments=GlobArguments(path="**/*.py"))
        # Simulate what _make_ignore_spec does
        from pathspec import PathSpec
        patterns = [".moss/"]
        ignore_file = tmp_path / ".groundignore"
        patterns.extend(
            ln for ln in ignore_file.read_text().splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        )
        spec = PathSpec.from_lines("gitignore", patterns)
        obs = observe_sync(pin, anchor, ignore=spec)
        assert obs.exists is True
        assert obs.size == 1  # only keep.py — skip_me/ tree excluded

    def test_ignore_combined_with_recursion(self, tmp_path):
        """recursion 和 ignore 同时生效 — ignore 在走递归时剪枝."""
        (tmp_path / "a.py").write_text("a")
        deep = tmp_path / "deep"
        deep.mkdir()
        (deep / "b.py").write_text("b")
        deeper = deep / "deeper"
        deeper.mkdir()
        (deeper / "c.py").write_text("c")
        spec = self._spec(["deep/"])
        matches = glob_limited(tmp_path, "**/*.py", recursion=1, ignore=spec)
        rels = {str(m.relative_to(tmp_path)) for m in matches}
        assert "a.py" in rels
        assert "deep/b.py" not in rels  # deep/ 被 ignore 剪枝
        assert "deep/deeper/c.py" not in rels  # 同上


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
