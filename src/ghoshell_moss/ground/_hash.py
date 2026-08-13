"""Pin observation — IO-light snapshot: exists + 计数 + exec payload.

observe 不读内容 — 内容读取全交 render (_render.py). 对账 (感知 digest)
由 Ground.snapshot() 承担 (hash 渲染文本全量), 不在本层.

每种子类自带观察逻辑:
- FilePin: stat → exists + size + binary 探测. 不读内容.
- GlobPin: 展开命中路径 (不读内容). 过 GLOB_IGNORE 过滤噪声目录.
- FrontmatterPin: stat → exists + 命中数. 不读内容.
- LsPin: 展开目录树 + 条目列表 (不读内容). 过 GLOB_IGNORE 过滤.
- ExecPin: observe 即执行, payload = 渲染就绪的结果文本.
- LawPin: 收集约定文件 → 命中数.

文件不存在 → Observation(exists=False).

所有发现型观察接受可选的 ``ignore: PathSpec | None`` 参数 — 场级
ignore 规则 (.gitignore 语义), 与 GLOB_IGNORE 叠层过滤.
"""

from __future__ import annotations

import asyncio
import fnmatch
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathspec import PathSpec

from ghoshell_moss.ground._addr import Anchor, resolve_path
from ghoshell_moss.ground._chain import collect_law_files
from ghoshell_moss.ground.contract import (
    ExecPin,
    FilePin,
    FrontmatterPin,
    GlobPin,
    LawPin,
    LsPin,
    Pin,
)

__all__ = [
    "Observation",
    "observe",
    "observe_sync",
    "glob_limited",
    "GLOB_IGNORE",
    "parse_range",
]

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
    """一次观察的结果 — IO-light: exists + size + (exec) payload."""

    exists: bool
    is_binary: bool = False
    payload: str | None = None
    """观察即产出内容的 verb (exec) 把结果存这里, 渲染层直接消费 —
    保证一帧只执行一次."""
    size: int | None = None
    """自然单位的规模:
    file → bytes; glob/frontmatter/ls → 命中/条目数; exec → 输出字符数.
    None = 不适用. observe 诊断展示直接消费."""
    unit: str = ""
    """size 的显示单位: 'B' / 'entries' / 'chars'."""


# -- async entry (context() 并发调用) ---------------------------------------


async def observe(
    pin: Pin, anchor: Anchor, *, ignore: PathSpec | None = None,
) -> Observation:
    """观察 pin 当前状态. async wrapper — IO 卸载到线程池."""
    return await asyncio.to_thread(observe_sync, pin, anchor, ignore=ignore)


# -- sync entry ---------------------------------------------------------------


def observe_sync(
    pin: Pin, anchor: Anchor, *, ignore: PathSpec | None = None,
) -> Observation:
    """同步观察, 按 pin 子类分发."""
    if isinstance(pin, FilePin):
        return _observe_file(pin, anchor)
    if isinstance(pin, GlobPin):
        return _observe_glob(pin, anchor, ignore=ignore)
    if isinstance(pin, FrontmatterPin):
        return _observe_frontmatter(pin, anchor, ignore=ignore)
    if isinstance(pin, LsPin):
        return _observe_ls(pin, anchor, ignore=ignore)
    if isinstance(pin, ExecPin):
        return _observe_exec(pin, anchor)
    if isinstance(pin, LawPin):
        return _observe_law(pin, anchor)
    raise TypeError(f"unknown pin type: {type(pin).__name__}")


# -- per-class observers ----------------------------------------------------


def _observe_file(pin: FilePin, anchor: Anchor) -> Observation:
    target = resolve_path(pin.arguments.path, anchor)
    try:
        st = target.stat()
    except FileNotFoundError:
        return Observation(exists=False)

    binary = _is_binary(target)
    return Observation(exists=True, is_binary=binary, size=st.st_size, unit="B")


def _observe_glob(
    pin: GlobPin, anchor: Anchor, *, ignore: PathSpec | None = None,
) -> Observation:
    root = anchor.ground
    pattern = pin.arguments.path

    if pattern.startswith("$"):
        resolved = resolve_path(pattern, anchor)
        pattern = str(resolved.relative_to(root)) if resolved.is_relative_to(root) else str(resolved)

    matches: list[Path] = []
    for hit in glob_limited(root, pattern, recursion=pin.arguments.max_depth, ignore=ignore):
        if not hit.is_file():
            continue
        if _path_touches_ignore(hit, root):
            continue
        matches.append(hit)

    if not matches:
        return Observation(exists=True, size=0, unit="entries")
    return Observation(exists=True, size=len(matches), unit="entries")


def _observe_frontmatter(
    pin: FrontmatterPin, anchor: Anchor, *, ignore: PathSpec | None = None,
) -> Observation:
    path_raw = pin.arguments.path

    # Pattern mode
    if _has_glob(path_raw):
        return _observe_frontmatter_pattern(pin, anchor, ignore=ignore)

    # Single-file mode
    target = resolve_path(path_raw, anchor)
    try:
        target.stat()
    except FileNotFoundError:
        return Observation(exists=False)
    return Observation(exists=True, size=1, unit="entries")


def _observe_frontmatter_pattern(
    pin: FrontmatterPin, anchor: Anchor, *, ignore: PathSpec | None = None,
) -> Observation:
    root = anchor.ground
    pattern = pin.arguments.path
    if pattern.startswith("$"):
        resolved = resolve_path(pattern, anchor)
        pattern = str(resolved.relative_to(root))
    hits = glob_limited(
        root, pattern,
        recursion=pin.arguments.max_depth, stop_on_match=True, ignore=ignore,
    )
    files = [h for h in hits if h.is_file() and not _path_touches_ignore(h, root)]

    if not files:
        return Observation(exists=True, size=0, unit="entries")
    return Observation(exists=True, size=len(files), unit="entries")


def _observe_ls(
    pin: LsPin, anchor: Anchor, *, ignore: PathSpec | None = None,
) -> Observation:
    root_dir = resolve_path(pin.arguments.path, anchor)
    if not root_dir.is_dir():
        return Observation(exists=False)

    effective_depth = pin.arguments.depth
    if pin.arguments.max_depth is not None:
        effective_depth = min(effective_depth, pin.arguments.max_depth)

    entries: list[str] = []
    _walk_ls(root_dir, depth=effective_depth, prefix="", entries=entries,
             ignore=ignore, ground_root=anchor.ground)

    if not entries:
        return Observation(exists=True, size=0, unit="entries")
    return Observation(exists=True, size=len(entries), unit="entries")


def _observe_exec(pin: ExecPin, anchor: Anchor) -> Observation:
    """observe 即执行. payload = 渲染就绪的结果文本.

    授权模型 = Makefile 级信任: ref 必须是场根子树内的可执行文件.
    - 相对路径, 不允许 ../ 跨场
    - 场根 (anchor.ground) 是解析基准
    - +x 位为准; 缺失 → missing (授权拒绝, 不是错误)
    - shebang 决定解释器 (sh / python / binary 一视同仁)

    失败可见: 非零退出附 [exit N] + stderr 尾部; 超时附 [timeout] 标记.
    """
    args = pin.arguments
    ref = args.ref

    # 授权检查: 拒绝绝对路径 / 跨场跳出
    if Path(ref).is_absolute() or ".." in Path(ref).parts:
        return _exec_rejected("[outside ground]")

    resolved = (anchor.ground / ref).resolve()
    try:
        resolved.relative_to(anchor.ground.resolve())
    except ValueError:
        return _exec_rejected("[outside ground]")

    if not resolved.is_file():
        return Observation(exists=False)

    if not os.access(resolved, os.X_OK):
        return _exec_rejected("[not executable]")

    env = dict(os.environ)
    env["GROUND"] = str(anchor.ground)
    env["CWD"] = str(anchor.cwd)

    try:
        proc = subprocess.run(
            [str(resolved)],
            cwd=str(anchor.ground),
            env=env,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired as e:
        partial = e.stdout if isinstance(e.stdout, str) else ""
        payload = (partial.rstrip() + "\n" if partial.strip() else "") + \
            f"[timeout after {args.timeout:g}s]"
        return Observation(exists=True, payload=payload, size=len(payload), unit="chars")
    except OSError as e:
        payload = f"error: cannot execute: {e}"
        return Observation(exists=True, payload=payload, size=len(payload), unit="chars")

    parts: list[str] = []
    if proc.stdout.strip():
        parts.append(proc.stdout.rstrip())
    if proc.returncode != 0:
        parts.append(f"[exit {proc.returncode}]")
        stderr_tail = "\n".join(proc.stderr.rstrip().splitlines()[-5:])
        if stderr_tail.strip():
            parts.append(stderr_tail)

    payload = "\n".join(parts) if parts else "(no output)"
    return Observation(exists=True, payload=payload, size=len(payload), unit="chars")


def _observe_law(pin: LawPin, anchor: Anchor) -> Observation:
    """law pin 观察 — 收集 cwd 向上到 ground root 的约定文件清单.

    内容由渲染层读取 (render-time, 与 file/glob 同构).
    """
    files = collect_law_files(anchor, pin.arguments.filename)
    if not files:
        return Observation(exists=True, size=0, unit="entries")
    return Observation(exists=True, size=len(files), unit="entries")


def _exec_rejected(message: str) -> Observation:
    """exec 授权拒绝 — 不是文件缺失, 是安全策略拒绝."""
    return Observation(exists=True, payload=message, size=0, unit="")


# -- helpers ----------------------------------------------------------------


def _has_glob(raw: str) -> bool:
    unescaped = raw.replace("\\$", "$")
    return any(c in unescaped for c in "*?[")


def glob_limited(
    root: Path,
    pattern: str,
    *,
    recursion: int | None = None,
    stop_on_match: bool = False,
    ignore: PathSpec | None = None,
) -> list[Path]:
    """Resolve a glob pattern against ``root`` by explicit recursion.

    ``pattern`` is relative to ``root`` (callers resolve anchors first).

    Two orthogonal bounds (SPEC §4.1):
    - ``recursion``: max directory levels ``**`` may cross below the
      pattern's static base. 0 = no recursion; N = N levels; None = unbounded.
      (Path component count, NOT filename-inclusive — ``recursion=1`` is
      "one layer of sub-fields", the intuitive reading.)
    - ``stop_on_match``: a directory that directly contains a match is a
      boundary — its subdirectories are not recursed. Field discovery:
      ``**/GROUND.md`` does not penetrate child grounds. The pattern's
      static base is exempt (the ground's own marker is not a stop).

    ``ignore`` is an optional PathSpec — ground-level ignore (.gitignore
    semantics), applied while walking so pruned subtrees are never entered.
    """
    parts = pattern.split("/")
    base_parts: list[str] = []
    idx = 0
    for i, seg in enumerate(parts):
        if _has_glob(seg):
            idx = i
            break
        base_parts.append(seg)
    else:
        idx = len(parts)
    base = root.joinpath(*base_parts) if base_parts else root
    if not base.is_dir():
        return []
    rel = parts[idx:]
    if not rel:
        return [base]

    matches: list[Path] = []
    _walk_glob(
        base, rel, depth=0, matches=matches, recursion=recursion,
        stop_on_match=stop_on_match, root=root, ignore=ignore, base=base,
    )
    return matches


def _walk_glob(
    dir_: Path,
    segs: list[str],
    *,
    depth: int,
    matches: list[Path],
    recursion: int | None,
    stop_on_match: bool,
    root: Path,
    ignore: PathSpec | None,
    base: Path,
) -> bool:
    """Recursively match ``segs`` under ``dir_``.

    Returns True if a match was produced at this directory level — drives
    the stop_on_match boundary (a dir that directly contains a match is a
    boundary, its subdirectories are not entered). The pattern's static
    ``base`` is exempt: its own marker must not stop discovery of children.
    """
    if not segs:
        matches.append(dir_)
        return True

    seg, rest = segs[0], segs[1:]
    local_hit = False

    if seg == "**":
        # zero dirs consumed: rest applies at dir_ itself
        if rest:
            local_hit = _walk_glob(
                dir_, rest, depth=depth, matches=matches,
                recursion=recursion, stop_on_match=stop_on_match,
                root=root, ignore=ignore, base=base,
            )
        else:
            matches.append(dir_)
            local_hit = True

        # one-or-more dirs consumed: descend, bounded by recursion +
        # stop_on_match (static base exempt)
        if recursion is None or depth < recursion:
            if not (stop_on_match and local_hit and dir_ != base):
                for child in _dir_children(dir_, root, ignore):
                    _walk_glob(
                        child, segs, depth=depth + 1, matches=matches,
                        recursion=recursion, stop_on_match=stop_on_match,
                        root=root, ignore=ignore, base=base,
                    )
    else:
        for entry in _dir_children(dir_, root, ignore):
            if not _seg_match(entry.name, seg):
                continue
            if entry.is_dir():
                if not rest:
                    matches.append(entry)
                    local_hit = True
                elif _walk_glob(
                    entry, rest, depth=depth, matches=matches,
                    recursion=recursion, stop_on_match=stop_on_match,
                    root=root, ignore=ignore, base=base,
                ):
                    local_hit = True
            elif not rest:
                matches.append(entry)
                local_hit = True

    return local_hit


def _seg_match(name: str, seg: str) -> bool:
    """Single path-segment glob match (*, ?, [...]). ``**`` is handled by _walk_glob."""
    return fnmatch.fnmatchcase(name, seg)


def _dir_children(dir_: Path, root: Path, ignore: PathSpec | None) -> list[Path]:
    """Iterable dir entries, skipping GLOB_IGNORE noise and ignore-spec pruned paths."""
    try:
        entries = list(dir_.iterdir())
    except OSError:
        return []
    out: list[Path] = []
    for e in entries:
        if e.name in GLOB_IGNORE:
            continue
        if ignore is not None:
            try:
                rel = e.relative_to(root).as_posix()
            except ValueError:
                rel = e.as_posix()
            if e.is_dir():
                if ignore.match_file(rel + "/"):
                    continue
            elif ignore.match_file(rel):
                continue
        out.append(e)
    return sorted(out, key=lambda p: p.name)


def parse_range(raw: str, total_lines: int) -> tuple[int, int]:
    """'N' or 'N-M' → (start, end) 1-indexed inclusive, clamped to [1, total_lines].

    clamp 后区间为空 (start 越过文件末尾或 descending range) 抛 ValueError.
    """
    if "-" in raw:
        a, b = raw.split("-", 1)
        start, end = int(a), int(b)
    else:
        start = end = int(raw)
    start = max(1, start)
    end = min(end, total_lines)
    if start > end:
        raise ValueError(f"invalid range {raw!r}: empty interval after clamp")
    return start, end


def _walk_ls(
    dir_: Path,
    depth: int,
    prefix: str,
    entries: list[str],
    *,
    ignore: PathSpec | None = None,
    ground_root: Path | None = None,
) -> None:
    if depth <= 0:
        return
    try:
        items = sorted(
            (e for e in dir_.iterdir() if e.name not in GLOB_IGNORE),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except OSError:
        return

    # 场级 ignore: 预过滤 — 被忽略的目录完全不出现在列表中
    visible: list[Path] = []
    for entry in items:
        if entry.is_dir() and ignore is not None and ground_root is not None:
            try:
                rel = entry.relative_to(ground_root).as_posix()
            except ValueError:
                rel = entry.as_posix()
            if ignore.match_file(rel + "/"):
                continue
        visible.append(entry)

    for i, entry in enumerate(visible):
        is_last = i == len(visible) - 1
        connector = "└── " if is_last else "├── "
        marker = "/" if entry.is_dir() else ""
        entries.append(f"{prefix}{connector}{entry.name}{marker}")
        if entry.is_dir() and depth > 1:
            sub_prefix = prefix + ("    " if is_last else "│   ")
            _walk_ls(entry, depth - 1, sub_prefix, entries,
                     ignore=ignore, ground_root=ground_root)


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
