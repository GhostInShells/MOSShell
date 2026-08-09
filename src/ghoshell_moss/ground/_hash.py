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
from ghoshell_moss.ground._chain import collect_law_files
from ghoshell_moss.ground.contract import (
    ExecPin,
    FilePin,
    FrontmatterPin,
    GlobPin,
    LawPin,
    LsPin,
    Pin,
    PathOutsideRootError,
)

__all__ = ["Observation", "PinShadow", "observe", "observe_sync", "GLOB_IGNORE", "parse_range"]

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
    payload: str | None = None
    """观察即产出内容的 verb (exec) 把结果存这里, 渲染层直接消费 —
    保证一帧只执行一次."""
    size: int | None = None
    """自然单位的规模:
    file → bytes; glob/frontmatter/ls → 命中/条目数; exec → 输出字符数.
    None = 不适用. observe 诊断展示直接消费."""
    unit: str = ""
    """size 的显示单位: 'B' / 'entries' / 'chars'."""


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
    mtime = st.st_mtime
    size = st.st_size

    binary = _is_binary(target)

    if pin.arguments.range is not None:
        text = target.read_text(encoding="utf-8", errors="replace")
        try:
            start, end = parse_range(pin.arguments.range, len(text.splitlines()))
        except ValueError:
            start, end = 0, 0  # invalid range → 空切片 hash, 确定性可对账
        sliced = "".join(text.splitlines(keepends=True)[start - 1 : end])
        digest = hashlib.sha256(sliced.encode("utf-8")).hexdigest()
        return Observation(
            exists=True, mtime=mtime, hash=digest, is_binary=binary,
            size=len(sliced.encode("utf-8")), unit="B",
        )

    if binary:
        content = target.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        return Observation(
            exists=True, mtime=mtime, hash=digest, is_binary=True,
            size=size, unit="B",
        )

    text = target.read_text(encoding="utf-8", errors="replace")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return Observation(
        exists=True, mtime=mtime, hash=digest, is_binary=False,
        size=size, unit="B",
    )


def _observe_glob(pin: GlobPin, anchor: Anchor) -> Observation:
    root = anchor.ground
    pattern = pin.arguments.path

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
        return Observation(
            exists=True, mtime=None, hash=_EMPTY_HASH,
            size=0, unit="entries",
        )

    mtimes: list[float] = []
    for m in matches:
        try:
            mtimes.append(m.stat().st_mtime)
        except FileNotFoundError:
            continue
    latest = max(mtimes) if mtimes else None

    rels = sorted(str(m.relative_to(root)) for m in matches)
    digest = hashlib.sha256("\n".join(rels).encode("utf-8")).hexdigest()
    return Observation(
        exists=True, mtime=latest, hash=digest,
        size=len(matches), unit="entries",
    )


def _observe_frontmatter(pin: FrontmatterPin, anchor: Anchor) -> Observation:
    path_raw = pin.arguments.path

    # Pattern mode
    if _has_glob(path_raw):
        return _observe_frontmatter_pattern(pin, anchor)

    # Single-file mode
    target = resolve_path(path_raw, anchor)
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
    return Observation(
        exists=True, mtime=mtime, hash=digest,
        size=1, unit="entries",
    )


def _observe_frontmatter_pattern(pin: FrontmatterPin, anchor: Anchor) -> Observation:
    root = anchor.ground
    pattern = pin.arguments.path
    if pattern.startswith("$"):
        resolved = resolve_path(pattern, anchor)
        pattern = str(resolved.relative_to(root))
    hits = sorted(root.glob(pattern))
    files = [h for h in hits if h.is_file() and not _path_touches_ignore(h, root)]

    if not files:
        return Observation(
            exists=True, mtime=None, hash=_EMPTY_HASH,
            size=0, unit="entries",
        )

    import re
    parts: list[str] = []
    latest_mtime: float | None = None
    for f in files:
        try:
            st = f.stat()
            text = f.read_text(encoding="utf-8", errors="replace")
        except (FileNotFoundError, OSError):
            continue
        if latest_mtime is None or st.st_mtime > latest_mtime:
            latest_mtime = st.st_mtime
        fm_match = re.match(r"\A---\s*\n(.*?)\n---", text, re.DOTALL)
        payload = fm_match.group(1) if fm_match else text
        rel = str(f.relative_to(root))
        parts.append(f"-- {rel}\n{payload}")

    if not parts:
        return Observation(
            exists=True, mtime=None, hash=_EMPTY_HASH,
            size=0, unit="entries",
        )

    digest = hashlib.sha256("\n\n".join(parts).encode("utf-8")).hexdigest()
    return Observation(
        exists=True, mtime=latest_mtime, hash=digest,
        size=len(parts), unit="entries",
    )


def _observe_ls(pin: LsPin, anchor: Anchor) -> Observation:
    root_dir = resolve_path(pin.arguments.path, anchor)
    if not root_dir.is_dir():
        return Observation(exists=False)

    entries: list[str] = []
    _walk_ls(root_dir, depth=pin.arguments.depth, prefix="", entries=entries)

    if not entries:
        return Observation(
            exists=True, mtime=None, hash=_EMPTY_HASH,
            size=0, unit="entries",
        )

    digest = hashlib.sha256("\n".join(entries).encode("utf-8")).hexdigest()
    return Observation(
        exists=True, mtime=None, hash=digest,
        size=len(entries), unit="entries",
    )


def _observe_exec(pin: ExecPin, anchor: Anchor) -> Observation:
    """observe 即执行. payload = 渲染就绪的结果文本, hash = sha256(payload).

    授权模型 = Makefile 级信任: ref 必须是场根子树内的可执行文件.
    - 相对路径, 不允许 ../ 跨场
    - 场根 (anchor.ground) 是解析基准
    - +x 位为准; 缺失 → missing (授权拒绝, 不是错误)
    - shebang 决定解释器 (sh / python / binary 一视同仁)

    失败可见: 非零退出附 [exit N] + stderr 尾部; 超时附 [timeout] 标记.
    """
    import os
    import subprocess

    args = pin.arguments
    ref = args.ref

    # 授权检查: 拒绝绝对路径 / 跨场跳出 (安全拒绝, 非文件缺失)
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
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return Observation(
            exists=True, hash=digest, payload=payload,
            size=len(payload), unit="chars",
        )
    except OSError as e:
        payload = f"error: cannot execute: {e}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return Observation(
            exists=True, hash=digest, payload=payload,
            size=len(payload), unit="chars",
        )

    parts: list[str] = []
    if proc.stdout.strip():
        parts.append(proc.stdout.rstrip())
    if proc.returncode != 0:
        parts.append(f"[exit {proc.returncode}]")
        stderr_tail = "\n".join(proc.stderr.rstrip().splitlines()[-5:])
        if stderr_tail.strip():
            parts.append(stderr_tail)

    payload = "\n".join(parts) if parts else "(no output)"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return Observation(
        exists=True, hash=digest, payload=payload,
        size=len(payload), unit="chars",
    )


def _observe_law(pin: LawPin, anchor: Anchor) -> Observation:
    """law pin 观察 — 收集 cwd 向上到 ground root 的约定文件清单.

    内容由渲染层读取 (render-time, 与 file/glob 同构). 观察只对
    文件集合做 hash — 位置依赖, 不参与 stale 对账 (tracks_changes=False).
    """
    files = collect_law_files(anchor, pin.arguments.filename)
    if not files:
        return Observation(
            exists=True, mtime=None, hash=_EMPTY_HASH,
            size=0, unit="entries",
        )

    rels: list[str] = []
    latest: float | None = None
    for f in files:
        rels.append(str(f.relative_to(anchor.ground)) if f.is_relative_to(anchor.ground) else str(f))
        try:
            st = f.stat()
        except OSError:
            continue
        if latest is None or st.st_mtime > latest:
            latest = st.st_mtime

    digest = hashlib.sha256("\n".join(rels).encode("utf-8")).hexdigest()
    return Observation(
        exists=True, mtime=latest, hash=digest,
        size=len(files), unit="entries",
    )


def _exec_rejected(message: str) -> Observation:
    """exec 授权拒绝 — 不是文件缺失, 是安全策略拒绝."""
    digest = hashlib.sha256(message.encode("utf-8")).hexdigest()
    return Observation(
        exists=True, hash=digest, payload=message, size=0, unit="",
    )


# -- helpers ----------------------------------------------------------------


def _has_glob(raw: str) -> bool:
    unescaped = raw.replace("\\$", "$")
    return any(c in unescaped for c in "*?[")


def parse_range(raw: str, total_lines: int) -> tuple[int, int]:
    """'N' or 'N-M' → (start, end) 1-indexed inclusive, clamped to [1, total_lines].

    clamp 后区间为空 (start 越过文件末尾或 descending range) 抛 ValueError —
    render 与 hash 共用同一实现, 避免两份 _parse_range 行为漂移.
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
