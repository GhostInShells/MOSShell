"""Pin observation — per-class mtime + hash 对账.

每种子类自带观察逻辑:
- FilePin: 读文件全文 (可选 range 切片) → sha256. binary 文件跳过内容渲染, 仅 hash raw bytes.
- GlobPin: 展开 + hash 命中路径列表 (不读内容). 过 GLOB_IGNORE 过滤噪声目录.
- FrontmatterPin: 读文件 frontmatter → sha256
- LsPin: 展开目录树 + hash 条目列表 (不读内容). 过 GLOB_IGNORE 过滤.

文件不存在 → Observation(exists=False).
"""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from ghoshell_moss.ground._addr import Anchor, resolve_path, is_glob_pattern
from ghoshell_moss.ground.contract import (
    FilePin,
    FrontmatterPin,
    GlobPin,
    LsPin,
    Pin,
    PathOutsideRootError,
)

__all__ = ["Observation", "PinShadow", "observe", "observe_sync", "GLOB_IGNORE"]

_EMPTY_HASH = hashlib.sha256(b"").hexdigest()

# basename 精确匹配, 不解析 .gitignore. glob 展开时每层目录遇到这些就跳过.
# ground root 不一定是 git root, pathspec 从 git root 读会错位.
GLOB_IGNORE: frozenset[str] = frozenset({
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".DS_Store",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".idea",
    ".vscode",
})


@dataclass(frozen=True)
class Observation:
    """一次观察的结果."""

    exists: bool
    mtime: float | None = None
    hash: str | None = None
    is_binary: bool = False


@dataclass
class PinShadow:
    """运行时观察影子 — 不进 GROUND.md.

    每 pin 一个, 存上次承认时的观察状态. context() 时与当前 Observation
    对比: hash 相同 → 不标; hash 不同 → [changed on disk].
    """

    mtime: float | None = None
    hash: str | None = None


# -- async entry (context() 并发调用) ---------------------------------------


async def observe(pin: Pin, anchor: Anchor) -> Observation:
    """观察 pin 当前状态. async wrapper — IO 卸载到线程池."""
    return await asyncio.to_thread(observe_sync, pin, anchor)


# -- sync entry (pin() / update() 内置观察) ---------------------------------


def observe_sync(pin: Pin, anchor: Anchor) -> Observation:
    """同步观察, 按 pin 子类分发."""
    if isinstance(pin, FilePin):
        return _observe_file(pin, anchor)
    if isinstance(pin, GlobPin):
        return _observe_glob(pin, anchor)
    if isinstance(pin, FrontmatterPin):
        return _observe_frontmatter(pin, anchor)
    if isinstance(pin, LsPin):
        return _observe_ls(pin, anchor)
    raise TypeError(f"unknown pin type: {type(pin).__name__}")


# -- per-class observers ----------------------------------------------------


def _observe_file(pin: FilePin, anchor: Anchor) -> Observation:
    target = resolve_path(pin.path, anchor)
    try:
        mtime = target.stat().st_mtime
    except FileNotFoundError:
        return Observation(exists=False)

    binary = _is_binary(target)

    if pin.range is not None:
        text = target.read_text(encoding="utf-8", errors="replace")
        start, end = _parse_range(pin.range, len(text.splitlines()))
        sliced = "".join(text.splitlines(keepends=True)[start - 1 : end])
        digest = hashlib.sha256(sliced.encode("utf-8")).hexdigest()
        return Observation(exists=True, mtime=mtime, hash=digest, is_binary=binary)

    if binary:
        content = target.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        return Observation(exists=True, mtime=mtime, hash=digest, is_binary=True)

    text = target.read_text(encoding="utf-8", errors="replace")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Observation(exists=True, mtime=mtime, hash=digest, is_binary=False)


def _observe_glob(pin: GlobPin, anchor: Anchor) -> Observation:
    root = anchor.ground
    pattern = pin.pattern

    # 锚点语法解析后做 glob
    if pattern.startswith("$"):
        resolved = resolve_path(pattern, anchor)
        pattern = str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved)

    matches: list[Path] = []
    for hit in sorted(root.glob(pattern)):
        if not hit.is_file():
            continue
        if _path_touches_ignore(hit, root):
            continue
        try:
            hit.relative_to(root)
        except ValueError:
            continue
        matches.append(hit)

    if not matches:
        return Observation(exists=True, mtime=None, hash=_EMPTY_HASH)

    mtimes: list[float] = []
    for m in matches:
        try:
            mtimes.append(m.stat().st_mtime)
        except FileNotFoundError:
            continue
    latest = max(mtimes) if mtimes else None

    rels = sorted(str(m.relative_to(root)) for m in matches)
    digest = hashlib.sha256("\n".join(rels).encode("utf-8")).hexdigest()
    return Observation(exists=True, mtime=latest, hash=digest)


def _observe_frontmatter(pin: FrontmatterPin, anchor: Anchor) -> Observation:
    target = resolve_path(pin.path, anchor)
    try:
        mtime = target.stat().st_mtime
        text = target.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return Observation(exists=False)

    # 提取 frontmatter 块: ---\n...\n---
    import re
    fm_match = re.match(r"\A---\s*\n(.*?)\n---", text, re.DOTALL)
    payload = fm_match.group(1) if fm_match else text
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return Observation(exists=True, mtime=mtime, hash=digest)


def _observe_ls(pin: LsPin, anchor: Anchor) -> Observation:
    root_dir = resolve_path(pin.path, anchor)
    if not root_dir.is_dir():
        return Observation(exists=False)

    entries: list[str] = []
    _walk_ls(root_dir, depth=pin.depth, prefix="", entries=entries)

    if not entries:
        return Observation(exists=True, mtime=None, hash=_EMPTY_HASH)

    digest = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
    return Observation(exists=True, mtime=None, hash=digest)


# -- helpers ----------------------------------------------------------------


def _parse_range(raw: str, total_lines: int) -> tuple[int, int]:
    """'N' or 'N-M' → (start, end) 1-indexed inclusive, clamped to file."""
    if "-" in raw:
        a, b = raw.split("-", 1)
        start, end = int(a), int(b)
    else:
        start = end = int(raw)
    return (max(1, start), min(end, total_lines))


def _walk_ls(dir_: Path, depth: int, prefix: str, entries: list[str]) -> None:
    if depth <= 0:
        return
    try:
        items = sorted(
            (e for e in dir_.iterdir() if e.name not in GLOB_IGNORE),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except OSError:
        return

    for i, entry in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        marker = "/" if entry.is_dir() else ""
        entries.append(f"{prefix}{connector}{entry.name}{marker}")
        if entry.is_dir() and depth > 1:
            sub_prefix = prefix + ("    " if is_last else "│   ")
            _walk_ls(entry, depth - 1, sub_prefix, entries)


def _is_binary(path: Path) -> bool:
    """读前 1024 bytes, 出现 null byte 判 binary."""
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(1024)
    except OSError:
        return False


def _path_touches_ignore(path: Path, root: Path) -> bool:
    """检查 path 任何一部分的 basename 是否在 GLOB_IGNORE 中."""
    for part in path.relative_to(root).parts:
        if part in GLOB_IGNORE:
            return True
    return False
