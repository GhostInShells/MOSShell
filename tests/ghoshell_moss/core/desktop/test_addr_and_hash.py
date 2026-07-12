"""Tests for _addr.py + _hash.py — parsing 与对账观察."""
from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from ghoshell_moss.contracts.desktop import PathOutsideRootError
from ghoshell_moss.core.desktop._addr import (
    ParsedAddr,
    parse_addr,
    resolve_file_addr,
    resolve_glob_addr,
)
from ghoshell_moss.core.desktop._hash import Observation, observe


# ---- _addr.parse_addr ---------------------------------------------------


def test_parse_addr_plain_file() -> None:
    p = parse_addr("src/foo.py")
    assert p.kind == "file"
    assert p.path == "src/foo.py"
    assert p.start is None and p.end is None
    assert not p.is_glob


def test_parse_addr_range() -> None:
    p = parse_addr("src/foo.py:80-140")
    assert p.kind == "range"
    assert p.path == "src/foo.py"
    assert p.start == 80
    assert p.end == 140


def test_parse_addr_range_single_line() -> None:
    p = parse_addr("a.py:5-5")
    assert p.kind == "range"
    assert (p.start, p.end) == (5, 5)


def test_parse_addr_glob_star() -> None:
    p = parse_addr("**/*.py")
    assert p.kind == "glob"
    assert p.path == "**/*.py"
    assert p.is_glob


def test_parse_addr_glob_bracket() -> None:
    p = parse_addr("src/[abc]*.md")
    assert p.kind == "glob"


def test_parse_addr_glob_wins_over_range_suffix() -> None:
    # 含 `*` 立即认为是 glob, 不再尝试 range 后缀解析
    p = parse_addr("**/*.py:1-10")
    assert p.kind == "glob"
    assert p.path == "**/*.py:1-10"


def test_parse_addr_path_with_colon_but_not_range() -> None:
    # `:80` 不是 `:N-M` 后缀, 视为文件名一部分
    p = parse_addr("weird:name.py")
    assert p.kind == "file"
    assert p.path == "weird:name.py"


def test_parse_addr_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_addr("")


def test_parse_addr_range_start_greater_than_end() -> None:
    with pytest.raises(ValueError, match="start > end"):
        parse_addr("a.py:20-10")


def test_parse_addr_range_zero_start() -> None:
    with pytest.raises(ValueError, match=r"start must be >= 1"):
        parse_addr("a.py:0-5")


def test_parsed_addr_frozen_hashable() -> None:
    a = parse_addr("a.py:1-10")
    b = parse_addr("a.py:1-10")
    assert a == b
    assert hash(a) == hash(b)
    assert {a: 1}[b] == 1


# ---- _addr.resolve_file_addr -------------------------------------------


def test_resolve_file_addr_in_root(tmp_path: Path) -> None:
    parsed = parse_addr("sub/foo.py")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "foo.py").write_text("x")
    resolved = resolve_file_addr(parsed, tmp_path)
    assert resolved == (tmp_path / "sub" / "foo.py").resolve()


def test_resolve_file_addr_range_uses_path_field(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("x")
    parsed = parse_addr("a.py:1-3")
    resolved = resolve_file_addr(parsed, tmp_path)
    # 用 path 字段, 不带 :1-3
    assert resolved.name == "a.py"


def test_resolve_file_addr_escape_via_dotdot(tmp_path: Path) -> None:
    parsed = parse_addr("../escape.py")
    with pytest.raises(PathOutsideRootError):
        resolve_file_addr(parsed, tmp_path)


def test_resolve_file_addr_rejects_glob(tmp_path: Path) -> None:
    parsed = parse_addr("**/*.py")
    with pytest.raises(ValueError, match="glob"):
        resolve_file_addr(parsed, tmp_path)


def test_resolve_file_addr_absolute_within_root_ok(tmp_path: Path) -> None:
    # parsed.path 为绝对且在 root 内, 仍可解析
    inside = tmp_path / "x.py"
    inside.write_text("x")
    parsed = ParsedAddr(kind="file", raw=str(inside), path=str(inside))
    resolved = resolve_file_addr(parsed, tmp_path)
    assert resolved == inside.resolve()


def test_resolve_file_addr_absolute_outside_root_rejected(tmp_path: Path) -> None:
    other = tmp_path.parent / "not_in_root.py"
    parsed = ParsedAddr(kind="file", raw=str(other), path=str(other))
    with pytest.raises(PathOutsideRootError):
        resolve_file_addr(parsed, tmp_path)


# ---- _addr.resolve_glob_addr -------------------------------------------


def test_resolve_glob_addr_expands_and_sorts(tmp_path: Path) -> None:
    (tmp_path / "b.py").write_text("b")
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "c.py").write_text("c")
    parsed = parse_addr("*.py")
    matches = resolve_glob_addr(parsed, tmp_path)
    names = [m.name for m in matches]
    assert names == sorted(names)
    assert set(names) == {"a.py", "b.py", "c.py"}


def test_resolve_glob_addr_skips_directories(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.py").write_text("x")
    parsed = parse_addr("*")
    matches = resolve_glob_addr(parsed, tmp_path)
    # sub 是目录不进 matches, sub/x.py 也不匹配单星
    assert matches == []


def test_resolve_glob_addr_recursive(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "x.py").write_text("x")
    (tmp_path / "y.py").write_text("y")
    parsed = parse_addr("**/*.py")
    matches = resolve_glob_addr(parsed, tmp_path)
    names = sorted(m.name for m in matches)
    assert names == ["x.py", "y.py"]


def test_resolve_glob_addr_empty_hit(tmp_path: Path) -> None:
    parsed = parse_addr("nonexistent-*.py")
    assert resolve_glob_addr(parsed, tmp_path) == []


def test_resolve_glob_addr_rejects_non_glob() -> None:
    parsed = ParsedAddr(kind="file", raw="a.py", path="a.py")
    with pytest.raises(ValueError, match="glob"):
        resolve_glob_addr(parsed, Path("/tmp"))


# ---- _hash.observe -----------------------------------------------------


def _run(coro):
    return asyncio.run(coro)


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def test_observe_file_hashes_full_content(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("hello world\n")
    parsed = parse_addr("a.py")
    obs = _run(observe(parsed, tmp_path))
    assert obs.exists is True
    assert obs.mtime is not None
    assert obs.hash == _sha256_bytes(b"hello world\n")


def test_observe_file_missing(tmp_path: Path) -> None:
    parsed = parse_addr("nope.py")
    obs = _run(observe(parsed, tmp_path))
    assert obs.exists is False
    assert obs.mtime is None
    assert obs.hash is None


def test_observe_range_hashes_slice(tmp_path: Path) -> None:
    body = "L1\nL2\nL3\nL4\nL5\n"
    (tmp_path / "a.py").write_text(body)
    parsed = parse_addr("a.py:2-4")
    obs = _run(observe(parsed, tmp_path))
    assert obs.exists is True
    # 行 splitkeepends 后 [1:4] 即 "L2\nL3\nL4\n"
    assert obs.hash == _sha256_text("L2\nL3\nL4\n")


def test_observe_range_out_of_bounds_is_empty_slice(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("only one line\n")
    parsed = parse_addr("a.py:5-10")
    obs = _run(observe(parsed, tmp_path))
    # 文件在, 区间空; hash 是空串 hash
    assert obs.exists is True
    assert obs.hash == _sha256_text("")


def test_observe_range_missing_file(tmp_path: Path) -> None:
    parsed = parse_addr("nope.py:1-5")
    obs = _run(observe(parsed, tmp_path))
    assert obs.exists is False


def test_observe_glob_hashes_hit_list(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("a")
    (tmp_path / "b.py").write_text("b")
    parsed = parse_addr("*.py")
    obs = _run(observe(parsed, tmp_path))
    assert obs.exists is True
    assert obs.mtime is not None
    # 命中路径列表相对 root, 排序, 换行连接
    expected = _sha256_text("a.py\nb.py")
    assert obs.hash == expected


def test_observe_glob_empty_hit_is_still_exists(tmp_path: Path) -> None:
    parsed = parse_addr("nonexistent-*.py")
    obs = _run(observe(parsed, tmp_path))
    assert obs.exists is True
    assert obs.mtime is None
    # 空集 hash 是空 bytes 的 sha256
    assert obs.hash == _sha256_bytes(b"")


def test_observe_glob_mtime_is_latest_of_hits(tmp_path: Path) -> None:
    import os
    import time

    (tmp_path / "old.py").write_text("old")
    (tmp_path / "new.py").write_text("new")
    old_time = time.time() - 1000
    os.utime(tmp_path / "old.py", (old_time, old_time))

    parsed = parse_addr("*.py")
    obs = _run(observe(parsed, tmp_path))
    # 新文件 mtime 更大
    assert obs.mtime is not None
    assert obs.mtime > old_time + 100  # sanity


def test_observe_out_of_root_raises(tmp_path: Path) -> None:
    parsed = parse_addr("../escape.py")
    with pytest.raises(PathOutsideRootError):
        _run(observe(parsed, tmp_path))


def test_observe_range_and_file_of_same_file_differ_hash(tmp_path: Path) -> None:
    (tmp_path / "a.py").write_text("L1\nL2\nL3\n")
    file_obs = _run(observe(parse_addr("a.py"), tmp_path))
    range_obs = _run(observe(parse_addr("a.py:1-1"), tmp_path))
    # 全文 hash != 单行 hash — 变更粒度语义正确
    assert file_obs.hash != range_obs.hash


def test_observation_frozen() -> None:
    o = Observation(exists=True, mtime=1.0, hash="abc")
    with pytest.raises(Exception):
        o.exists = False  # type: ignore[misc]
