"""Frame rendering — body + pin result blocks.

Frame is body verbatim followed by pin observations, delimited by
HTML comment markers.  No meta, no @-expansion, no declaration block,
no line numbers.

    <body verbatim>

    <!-- ground:pin:<label> -->
    <pin observation content>
    <!-- /ground:pin:<label> -->

Meta is a separate rendering path used by ``moss ground meta``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ghoshell_moss.ground._addr import Anchor, resolve_path
from ghoshell_moss.ground._hash import GLOB_IGNORE, Observation, PinShadow, observe
from ghoshell_moss.ground.contract import (
    FilePin,
    FrontmatterPin,
    GlobPin,
    LsPin,
    Pin,
)

__all__ = ["render_context", "render_meta"]


async def render_context(
    body: str,
    pins: list[Pin],
    shadows: dict[str, PinShadow],
    anchor: Anchor,
) -> str:
    """Render a frame: body + pin result blocks.

    All IO-heavy segments (pin observation) run in parallel.
    """
    lines: list[str] = []

    # ---- body (verbatim) -----------------------------------------------
    if body.strip():
        lines.append(body.rstrip())
        lines.append("")

    # ---- observe all pins (parallel) -----------------------------------
    if not pins:
        return "\n".join(lines).rstrip() + "\n"

    tasks = {p.label: observe(p, anchor) for p in pins}
    results = await asyncio.gather(*tasks.values())
    observations: dict[str, Observation] = dict(zip(tasks.keys(), results))

    # ---- result blocks --------------------------------------------------
    for p in pins:
        obs = observations.get(p.label)
        shadow = shadows.get(p.label, PinShadow())

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


def render_meta(
    root: Path,
    doc_path: Path,
    chain: str,
    pins: list[Pin],
    *,
    id_: str | None = None,
    label: str | None = None,
) -> str:
    """Render the meta section — ground identity + pin TOC.

    Used by ``moss ground meta``.  Separated from frame so consumers
    who don't care about ground protocol get a clean content-only frame.
    """
    lines: list[str] = []

    # location
    if doc_path.resolve() == root.resolve() / "GROUND.md":
        lines.append(f"cd {root}")
    else:
        lines.append(f"cd {root}")
        lines.append(f"ground: {doc_path}")

    # chain
    if chain:
        n = chain.count("from:")
        lines.append(f"chain: +{n}")
    else:
        lines.append("chain: +0")

    # id
    if id_:
        lines.append(f"$id: {id_}")

    # pins
    lines.append("")
    if pins:
        lines.append("pins:")
        for p in pins:
            kwargs = _pin_kwargs(p)
            desc = f"  # {p.description}" if p.description else ""
            lines.append(f"  {p.label}:{p.verb}({kwargs}){desc}")
    else:
        lines.append("pins: (none)")

    return "\n".join(lines)


# -- result block ---------------------------------------------------------


def _render_result_block(
    pin: Pin,
    obs: Observation | None,
    stale: bool,
    missing: bool,
    anchor: Anchor,
) -> str:
    """HTML-comment-delimited pin observation block."""
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

    return (
        f"<!-- ground:pin:{pin.label} -->\n"
        f"{content}\n"
        f"<!-- /ground:pin:{pin.label} -->"
    )


def _render_pin_content(pin: Pin, anchor: Anchor) -> str:
    """Dispatch per pin subclass."""
    if isinstance(pin, FilePin):
        return _content_file(pin, anchor)
    if isinstance(pin, GlobPin):
        return _content_glob(pin, anchor)
    if isinstance(pin, FrontmatterPin):
        return _content_frontmatter(pin, anchor)
    if isinstance(pin, LsPin):
        return _content_ls(pin, anchor)
    return f"error: unknown pin type: {type(pin).__name__}"


# -- per-kind content renderers -------------------------------------------


def _content_file(pin: FilePin, anchor: Anchor) -> str:
    try:
        target = resolve_path(pin.arguments.path, anchor)
    except (OSError, ValueError):
        return "error: cannot read file"

    if _is_binary(target):
        return "[binary file, not rendered]"

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return "error: cannot read file"

    if pin.arguments.range is not None:
        lines_list = text.splitlines()
        start, end = _parse_range(pin.arguments.range, len(lines_list))
        if start > len(lines_list):
            return "error: range beyond file end"
        return "\n".join(lines_list[start - 1 : min(end, len(lines_list))])

    return text


def _content_glob(pin: GlobPin, anchor: Anchor) -> str:
    root = anchor.ground
    pattern = pin.arguments.pattern
    if pattern.startswith("$"):
        try:
            resolved = resolve_path(pattern, anchor)
            pattern = str(resolved.relative_to(root))
        except (ValueError, OSError):
            return "error: invalid glob path"

    hits = sorted(root.glob(pattern))
    files = [h for h in hits if h.is_file() and not _path_touches_ignore(h, root)]
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
    import re

    try:
        target = resolve_path(pin.arguments.path, anchor)
        text = target.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return "error: cannot read file"

    fm = re.match(r"\A---\s*\n(.*?)\n---", text, re.DOTALL)
    if fm is None:
        return "error: no frontmatter found"
    return fm.group(1)


def _content_ls(pin: LsPin, anchor: Anchor) -> str:
    try:
        root_dir = resolve_path(pin.arguments.path, anchor)
    except (OSError, ValueError):
        return "error: invalid path"

    if not root_dir.is_dir():
        return "error: not a directory"

    entries: list[str] = []
    _walk_ls_entries(root_dir, pin.arguments.depth, "", entries)
    return "\n".join(entries) if entries else "(empty)"


# -- meta helpers ---------------------------------------------------------


def _pin_kwargs(pin: Pin) -> str:
    """Pin subclass → kwargs display string."""
    parts: list[str] = []
    if isinstance(pin, FilePin):
        parts.append(f'path="{pin.arguments.path}"')
        if pin.arguments.range is not None:
            parts.append(f'range="{pin.arguments.range}"')
    elif isinstance(pin, GlobPin):
        parts.append(f'pattern="{pin.arguments.pattern}"')
    elif isinstance(pin, FrontmatterPin):
        parts.append(f'path="{pin.arguments.path}"')
    elif isinstance(pin, LsPin):
        parts.append(f'path="{pin.arguments.path}"')
        if pin.arguments.depth != 2:
            parts.append(f"depth={pin.arguments.depth}")
    return ", ".join(parts)


# -- general helpers ------------------------------------------------------


def _parse_range(raw: str, total_lines: int) -> tuple[int, int]:
    if "-" in raw:
        a, b = raw.split("-", 1)
        return int(a), int(b)
    return int(raw), int(raw)


def _walk_ls_entries(dir_: Path, depth: int, prefix: str, entries: list[str]) -> None:
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
        try:
            st = entry.stat()
            size_info = f"  ({st.st_size}B)" if entry.is_file() else ""
        except OSError:
            size_info = ""
        entries.append(f"{prefix}{connector}{entry.name}{marker}{size_info}")
        if entry.is_dir() and depth > 1:
            sub_prefix = prefix + ("    " if is_last else "│   ")
            _walk_ls_entries(entry, depth - 1, sub_prefix, entries)


def _is_binary(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return b"\x00" in f.read(1024)
    except OSError:
        return False


def _path_touches_ignore(path: Path, root: Path) -> bool:
    for part in path.relative_to(root).parts:
        if part in GLOB_IGNORE:
            return True
    return False
