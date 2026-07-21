"""Frame rendering — SPEC §6 布局.

结构:
    ground: <label> @ <path> [(doc: <doc>)]
    [chain: <paths>]
    [$id: <id>]

    <body verbatim>

    ```@<ref>
    <ref content>
    ```

    <label>:<kind>(key="val") # <description>

    ```<label>
    <pin expansion>
    ```
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

from ghoshell_moss.ground._addr import Anchor, resolve_path
from ghoshell_moss.ground._hash import Observation, PinShadow, observe
from ghoshell_moss.ground.contract import (
    AT_BUDGET,
    AT_MAX_DEPTH,
    FilePin,
    FrontmatterPin,
    GlobPin,
    LsPin,
    Pin,
)

__all__ = ["render_context"]

# @path recognition (SPEC §6.1): @ at line-start or after whitespace,
# followed by path-start char, not in fenced block.
_AT_REF_RE = re.compile(
    r"(?:^|(?<=\s))@([a-zA-Z0-9_./$-][a-zA-Z0-9_./$-]*)",
    re.MULTILINE,
)


async def render_context(
    label: str,
    root: Path,
    doc_path: Path,
    body: str,
    pins: list[Pin],
    shadows: dict[str, PinShadow],
    anchor: Anchor,
    *,
    id_: str | None = None,
    chain_summary: str = "",
) -> str:
    """渲染一帧 — Ground.context() 的后端.

    所有 IO 密集段 (pin 观察 + 内容读取) 在此并行处理.
    """
    lines: list[str] = []

    # ---- head ----------------------------------------------------------
    doc_annotation = ""
    if doc_path != root / "GROUND.md":
        doc_annotation = f" (doc: {doc_path})"
    lines.append(f"ground: {label} @ {root}{doc_annotation}")

    if chain_summary:
        lines.append(f"chain: {chain_summary}")

    if id_:
        lines.append(f"$id: {id_}")

    lines.append("")

    # ---- body (verbatim) -----------------------------------------------
    if body.strip():
        lines.append(body.rstrip())
        lines.append("")

    # ---- @-expansion ---------------------------------------------------
    at_blocks = _expand_at_refs(body, anchor, budget=AT_BUDGET)
    for block in at_blocks:
        lines.append(block)
        lines.append("")

    # ---- observe all pins (parallel) -----------------------------------
    observations: dict[str, Observation] = {}
    if pins:
        tasks = {p.label: observe(p, anchor) for p in pins}
        results = await asyncio.gather(*tasks.values())
        observations = dict(zip(tasks.keys(), results))

    # ---- declaration block ----------------------------------------------
    decl_lines = [_render_declaration(p) for p in pins]
    if decl_lines:
        lines.extend(decl_lines)
        lines.append("")

    # ---- result blocks --------------------------------------------------
    for p in pins:
        obs = observations.get(p.label)
        shadow = shadows.get(p.label, PinShadow())

        # stale 判定
        stale = (
            shadow.hash is not None
            and obs is not None
            and obs.exists
            and obs.hash != shadow.hash
        )
        missing = obs is not None and not obs.exists

        lines.append(_render_result_block(p, obs, stale, missing, anchor))
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# -- declaration block ----------------------------------------------------


def _render_declaration(pin: Pin) -> str:
    """一行: label:kind(key="val") # description."""
    kwargs = _pin_kwargs(pin)
    desc = f" # {pin.description}" if pin.description else ""
    return f"{pin.label}:{pin.kind}({kwargs}){desc}"


def _pin_kwargs(pin: Pin) -> str:
    """Pin 子类 → kwargs 展示字符串."""
    parts: list[str] = []
    if isinstance(pin, FilePin):
        parts.append(f'path="{pin.path}"')
        if pin.range is not None:
            parts.append(f'range="{pin.range}"')
    elif isinstance(pin, GlobPin):
        parts.append(f'pattern="{pin.pattern}"')
    elif isinstance(pin, FrontmatterPin):
        parts.append(f'path="{pin.path}"')
    elif isinstance(pin, LsPin):
        parts.append(f'path="{pin.path}"')
        if pin.depth != 2:
            parts.append(f"depth={pin.depth}")
    return ", ".join(parts)


# -- result block ---------------------------------------------------------


def _render_result_block(
    pin: Pin,
    obs: Observation | None,
    stale: bool,
    missing: bool,
    anchor: Anchor,
) -> str:
    """Fenced code block with label as lang tag."""
    if missing:
        content = "[missing]"
    elif stale:
        content = _render_pin_content(pin, anchor) + "\n[changed on disk]"
    elif obs is not None and obs.exists:
        content = _render_pin_content(pin, anchor)
    elif obs is not None and not obs.exists:
        content = "[missing]"
    else:
        content = "[not yet observed]"

    return f"```{pin.label}\n{content}\n```"


def _render_pin_content(pin: Pin, anchor: Anchor) -> str:
    """按 pin 子类渲染内容."""
    if isinstance(pin, FilePin):
        return _content_file(pin, anchor)
    if isinstance(pin, GlobPin):
        return _content_glob(pin, anchor)
    if isinstance(pin, FrontmatterPin):
        return _content_frontmatter(pin, anchor)
    if isinstance(pin, LsPin):
        return _content_ls(pin, anchor)
    return f"error: unknown pin type: {type(pin).__name__}"


def _content_file(pin: FilePin, anchor: Anchor) -> str:
    try:
        target = resolve_path(pin.path, anchor)
        text = target.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return "error: cannot read file"

    lines = text.splitlines()
    if pin.range is not None:
        start, end = _parse_range(pin.range, len(lines))
        if start > len(lines):
            return "error: range beyond file end"
        return "\n".join(
            f"{i}: {lines[i - 1]}" for i in range(start, min(end, len(lines)) + 1)
        )
    return "\n".join(f"{i+1}: {ln}" for i, ln in enumerate(lines))


def _content_glob(pin: GlobPin, anchor: Anchor) -> str:
    import glob as glob_mod

    root = anchor.ground
    pattern = pin.pattern
    if pattern.startswith("$"):
        try:
            resolved = resolve_path(pattern, anchor)
            pattern = str(resolved.relative_to(root))
        except (ValueError, OSError):
            return "error: invalid glob path"

    hits = sorted(root.glob(pattern))
    files = [h for h in hits if h.is_file()]
    if not files:
        return "(no matches)"

    lines: list[str] = []
    for f in files:
        try:
            st = f.stat()
            rel = f.relative_to(root)
            lines.append(f"{rel}  ({st.st_size}B, mtime={st.st_mtime:.0f})")
        except OSError:
            continue
    return "\n".join(lines) if lines else "(all matches vanished)"


def _content_frontmatter(pin: FrontmatterPin, anchor: Anchor) -> str:
    try:
        target = resolve_path(pin.path, anchor)
        text = target.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return "error: cannot read file"

    fm = re.match(r"\A---\s*\n(.*?)\n---", text, re.DOTALL)
    if fm is None:
        return "error: no frontmatter found"
    return fm.group(1)


def _content_ls(pin: LsPin, anchor: Anchor) -> str:
    try:
        root_dir = resolve_path(pin.path, anchor)
    except (OSError, ValueError):
        return "error: invalid path"

    if not root_dir.is_dir():
        return "error: not a directory"

    entries: list[str] = []
    _walk_ls_entries(root_dir, pin.depth, "", entries)
    return "\n".join(entries) if entries else "(empty)"


# -- @-expansion ----------------------------------------------------------


def _expand_at_refs(
    body: str,
    anchor: Anchor,
    *,
    budget: int = AT_BUDGET,
) -> list[str]:
    """扫描 body 中 @path 引用, 返回 fenced expansion blocks.

    SPEC §6.1 约束: cycle 检测, depth cap, budget cap.
    """
    refs = _find_at_refs(body)
    if not refs:
        return []

    visited: set[str] = set()
    blocks: list[str] = []
    spent = 0

    for ref_path in refs:
        if ref_path in visited:
            blocks.append(f"```@{ref_path}\n(@{ref_path} already expanded above)\n```")
            continue

        visited.add(ref_path)
        content, ok = _load_at_ref(ref_path, anchor, depth=0)
        if not ok:
            blocks.append(f"```@{ref_path}\nerror: cannot load\n```")
            continue

        spent += len(content)
        if spent > budget:
            blocks.append(f"```@{ref_path}\n(@{ref_path} skipped: budget exceeded)\n```")
            continue

        blocks.append(f"```@{ref_path}\n{content}\n```")

    if spent > budget:
        blocks.insert(0, f"⚠ @-expansion over budget: {spent} > {budget}")

    return blocks


def _find_at_refs(body: str) -> list[str]:
    """从 body 中提取 @path 引用, 去重保持首次出现顺序."""
    seen: set[str] = set()
    result: list[str] = []
    for m in _AT_REF_RE.finditer(body):
        ref = m.group(1).rstrip(".-")
        if ref not in seen:
            seen.add(ref)
            result.append(ref)
    return result


def _load_at_ref(
    ref: str,
    anchor: Anchor,
    depth: int,
) -> tuple[str, bool]:
    """加载一个 @path 引用. 返回 (content, ok)."""
    if depth >= AT_MAX_DEPTH:
        return f"(@{ref} exceeds depth cap)", False

    try:
        target = resolve_path(ref, anchor)
        content = target.read_text(encoding="utf-8", errors="replace")
        return content, True
    except (OSError, ValueError):
        return f"(@{ref} not found or inaccessible)", False


# -- helpers --------------------------------------------------------------


def _parse_range(raw: str, total_lines: int) -> tuple[int, int]:
    if "-" in raw:
        a, b = raw.split("-", 1)
        return int(a), int(b)
    return int(raw), int(raw)


def _walk_ls_entries(dir_: Path, depth: int, prefix: str, entries: list[str]) -> None:
    if depth <= 0:
        return
    try:
        items = sorted(dir_.iterdir(), key=lambda p: (p.is_file(), p.name.lower()))
    except OSError:
        return

    for i, entry in enumerate(items):
        is_last = i == len(items) - 1
        connector = "└── " if is_last else "├── "
        marker = "/" if entry.is_dir() else ""
        try:
            st = entry.stat()
            size_info = f"  ({st.st_size}B)" if entry.is_file() else ""
        except OSError:
            size_info = ""
        entries.append(f"{prefix}{connector}{entry.name}{marker}{size_info}")
        if entry.is_dir() and depth > 1:
            sub_prefix = prefix + ("    " if is_last else "│   ")
            _walk_ls_entries(entry, depth - 1, sub_prefix, entries)
