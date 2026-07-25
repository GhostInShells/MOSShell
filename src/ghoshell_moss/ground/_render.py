"""Frame rendering — body + pin result blocks.

Frame is body verbatim followed by pin observations, delimited by
HTML comment markers.  No meta, no @-expansion, no declaration block,
no line numbers, no raw mtime.

    <body verbatim>

    <!-- ground:pin:<label> -->
    <pin observation content>
    <!-- /ground:pin:<label> -->

Meta is a separate rendering path used by ``moss ground meta``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from ghoshell_moss.ground._addr import Anchor, anchor_kind, resolve_path
from ghoshell_moss.ground._hash import GLOB_IGNORE, Observation, PinShadow, observe
from ghoshell_moss.ground.contract import (
    ExecPin,
    FilePin,
    FrontmatterPin,
    GlobPin,
    LsPin,
    Pin,
)

__all__ = ["render_context", "render_meta", "render_walk"]


async def render_context(
    body: str,
    pins: list[Pin],
    shadows: dict[str, PinShadow],
    anchor: Anchor,
) -> str:
    """Render a frame: body + pin result blocks.

    All IO-heavy segments (pin observation) run in parallel via asyncio.gather.
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


# -- walk (场内移动) ------------------------------------------------------


def _pin_target_raw(pin: Pin) -> str:
    """Pin 的目标路径原文 (锚判定用). exec 无位置概念, 返回空."""
    if isinstance(pin, GlobPin):
        return pin.arguments.pattern
    if isinstance(pin, (FilePin, FrontmatterPin, LsPin)):
        return pin.arguments.path
    return ""


async def render_walk(
    cwd: Path,
    ground_root: Path,
    doc_path: Path,
    pins: list[Pin],
    shadows: dict[str, PinShadow],
    *,
    label: str | None = None,
) -> str:
    """场内移动视图 — cwd 无 GROUND.md, 法来自祖先场根.

    编辑权在场根, 这里只有视角:
    - 法链位置提示 (一行指回场根, 不重复 body)
    - $CWD 锚 pins 对当前目录展开 (场教的注视习惯)
    - 其余 pins 折叠为 TOC (场根的注视留在场根)

    不再有内建 ls — 若场希望站立位置有目录列表, 用 ``ls $CWD`` pin 声明.
    观感由场决定, 不由 harness 塞入.
    """
    anchor = Anchor(ground=ground_root, cwd=cwd)
    rel_doc = os.path.relpath(doc_path, cwd)
    rel_cwd = os.path.relpath(cwd, ground_root)
    display = label or ground_root.name

    lines: list[str] = []
    lines.append(f"ground: {display}  (law: {rel_doc})")
    lines.append(f"cwd: $GROUND/{rel_cwd}")
    lines.append("")

    # $CWD 锚 pins 展开; 其余折叠
    cwd_pins = [p for p in pins if anchor_kind(_pin_target_raw(p)) == "cwd"]
    folded = [p for p in pins if anchor_kind(_pin_target_raw(p)) != "cwd"]

    if cwd_pins:
        tasks = {p.label: observe(p, anchor) for p in cwd_pins}
        results = await asyncio.gather(*tasks.values())
        observations = dict(zip(tasks.keys(), results))
        for p in cwd_pins:
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

    if folded:
        lines.append(f"pins@{display} (moss ground frame {rel_doc.removesuffix('/GROUND.md') or '.'}):")
        for p in folded:
            desc = f"  # {p.description}" if p.description else ""
            lines.append(f"  {p.label}:{p.verb}({_pin_kwargs(p)}){desc}")

    return "\n".join(lines).rstrip() + "\n"


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
        content = _render_pin_content(pin, anchor, obs) + "\n[changed on disk]"
    elif obs is not None and obs.exists:
        content = _render_pin_content(pin, anchor, obs)
    elif obs is not None and not obs.exists:
        content = "[missing]"
    else:
        content = "[not yet observed]"

    return (
        f"<!-- ground:pin:{pin.label} -->\n"
        f"{content}\n"
        f"<!-- /ground:pin:{pin.label} -->"
    )


def _render_pin_content(
    pin: Pin, anchor: Anchor, obs: Observation | None = None
) -> str:
    """Dispatch per pin subclass."""
    if isinstance(pin, FilePin):
        return _content_file(pin, anchor)
    if isinstance(pin, GlobPin):
        return _content_glob(pin, anchor)
    if isinstance(pin, FrontmatterPin):
        return _content_frontmatter(pin, anchor)
    if isinstance(pin, LsPin):
        return _content_ls(pin, anchor)
    if isinstance(pin, ExecPin):
        return _content_exec(pin, obs)
    return f"error: unknown pin type: {type(pin).__name__}"


# -- per-kind content renderers -------------------------------------------


def _content_exec(pin: ExecPin, obs: Observation | None) -> str:
    """观察阶段已执行, 直接消费 payload — 一帧只跑一次进程."""
    if obs is None or obs.payload is None:
        return "[not yet observed]"
    return _apply_budget(obs.payload, pin.arguments.budget)


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
        text = "\n".join(lines_list[start - 1 : min(end, len(lines_list))])

    return _apply_budget(text, pin.arguments.budget)


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

    limit = pin.arguments.limit
    truncated = False
    if limit is not None and len(files) > limit:
        files = files[:limit]
        truncated = True

    lines: list[str] = []
    for f in files:
        try:
            st = f.stat()
            rel = f.relative_to(root)
            lines.append(f"{rel}  ({_fmt_size(st.st_size)})")
        except OSError:
            continue

    if not lines:
        return "(all matches vanished)"
    if truncated:
        lines.append(f"[showing {limit} of {len(hits)} entries]")
    return "\n".join(lines)


def _content_frontmatter(pin: FrontmatterPin, anchor: Anchor) -> str:
    import re

    path_raw = pin.arguments.path

    # Pattern mode: path contains glob characters
    if _has_glob(path_raw):
        return _content_frontmatter_pattern(pin, anchor)

    # Single-file mode
    try:
        target = resolve_path(path_raw, anchor)
        text = target.read_text(encoding="utf-8", errors="replace")
    except (OSError, ValueError):
        return "error: cannot read file"

    fm = _extract_frontmatter(text)
    if fm is None:
        return "error: no frontmatter found"
    fm = _filter_keys(fm, pin.arguments.keys)
    return _apply_budget(fm, pin.arguments.budget)


def _content_frontmatter_pattern(pin: FrontmatterPin, anchor: Anchor) -> str:
    import re

    root = anchor.ground
    pattern = pin.arguments.path
    if pattern.startswith("$"):
        try:
            resolved = resolve_path(pattern, anchor)
            pattern = str(resolved.relative_to(root))
        except (ValueError, OSError):
            return "error: invalid pattern"

    hits = sorted(root.glob(pattern))
    files = [h for h in hits if h.is_file() and h.name != ""
             and not _path_touches_ignore(h, root)]
    if not files:
        return "(no matches)"

    limit = pin.arguments.limit
    total = len(files)
    if limit is not None and len(files) > limit:
        files = files[:limit]

    blocks: list[str] = []
    total_chars = 0
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue

        fm = _extract_frontmatter(text)
        if fm is None:
            continue

        fm = _filter_keys(fm, pin.arguments.keys)
        rel = f.relative_to(root)
        block = f"-- {rel}\n{fm}"

        # per-entry budget check
        if pin.arguments.budget is not None:
            budget = pin.arguments.budget
            if total_chars + len(block) > budget:
                remaining = budget - total_chars
                if remaining > 50:
                    block = block[:remaining] + "\n..."
                else:
                    break
            total_chars += len(block)

        blocks.append(block)

    if not blocks:
        return "(no frontmatter found in matched files)"

    result = "\n\n".join(blocks)
    if limit is not None and total > limit:
        result += f"\n\n[showing {limit} of {total} entries]"
    if pin.arguments.budget is not None and total_chars >= pin.arguments.budget:
        result += f"\n[truncated at {pin.arguments.budget} chars]"
    return result


def _content_ls(pin: LsPin, anchor: Anchor) -> str:
    try:
        root_dir = resolve_path(pin.arguments.path, anchor)
    except (OSError, ValueError):
        return "error: invalid path"

    if not root_dir.is_dir():
        return "error: not a directory"

    entries: list[str] = []
    effective_depth = pin.arguments.depth
    if pin.arguments.max_depth is not None:
        effective_depth = min(effective_depth, pin.arguments.max_depth)
    _walk_ls_entries(root_dir, effective_depth, "", entries)

    limit = pin.arguments.limit
    total = len(entries)
    if limit is not None and len(entries) > limit:
        entries = entries[:limit]

    if not entries:
        return "(empty)"

    result = "\n".join(entries)
    if limit is not None and total > limit:
        result += f"\n[showing {limit} of {total} entries]"
    return result


# -- meta helpers ---------------------------------------------------------


def _pin_kwargs(pin: Pin) -> str:
    """Pin subclass → kwargs display string."""
    parts: list[str] = []
    if isinstance(pin, FilePin):
        parts.append(f'path="{pin.arguments.path}"')
        if pin.arguments.range is not None:
            parts.append(f'range="{pin.arguments.range}"')
        if pin.arguments.budget is not None:
            parts.append(f"budget={pin.arguments.budget}")
    elif isinstance(pin, GlobPin):
        parts.append(f'pattern="{pin.arguments.pattern}"')
        if pin.arguments.limit is not None:
            parts.append(f"limit={pin.arguments.limit}")
    elif isinstance(pin, FrontmatterPin):
        parts.append(f'path="{pin.arguments.path}"')
        if pin.arguments.keys:
            parts.append(f"keys={pin.arguments.keys}")
        if pin.arguments.budget is not None:
            parts.append(f"budget={pin.arguments.budget}")
        if pin.arguments.limit is not None:
            parts.append(f"limit={pin.arguments.limit}")
    elif isinstance(pin, LsPin):
        parts.append(f'path="{pin.arguments.path}"')
        if pin.arguments.depth != 2:
            parts.append(f"depth={pin.arguments.depth}")
        if pin.arguments.limit is not None:
            parts.append(f"limit={pin.arguments.limit}")
    elif isinstance(pin, ExecPin):
        parts.append(f'ref="{pin.arguments.ref}"')
        if pin.arguments.timeout != 10.0:
            parts.append(f"timeout={pin.arguments.timeout:g}")
        if pin.arguments.budget is not None:
            parts.append(f"budget={pin.arguments.budget}")
    return ", ".join(parts)


# -- format helpers -------------------------------------------------------


def _fmt_size(n_bytes: int) -> str:
    """Human-readable file size."""
    if n_bytes < 1024:
        return f"{n_bytes}B"
    if n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.0f}K"
    if n_bytes < 1024 * 1024 * 1024:
        return f"{n_bytes / (1024 * 1024):.1f}M"
    return f"{n_bytes / (1024 * 1024 * 1024):.1f}G"


def _apply_budget(text: str, budget: int | None) -> str:
    if budget is None or len(text) <= budget:
        return text
    return text[:budget] + f"\n[truncated at {budget} chars]"


def _filter_keys(fm_text: str, keys: list[str] | None) -> str:
    if keys is None:
        return fm_text
    import yaml
    try:
        fm_data = yaml.safe_load(fm_text) or {}
    except yaml.YAMLError:
        return fm_text
    filtered = {k: fm_data[k] for k in keys if k in fm_data}
    return yaml.safe_dump(filtered, allow_unicode=True, sort_keys=False).rstrip()


def _extract_frontmatter(text: str) -> str | None:
    import re
    fm = re.match(r"\A---\s*\n(.*?)\n---", text, re.DOTALL)
    return fm.group(1) if fm else None


def _has_glob(raw: str) -> bool:
    unescaped = raw.replace("\\$", "$")
    return any(c in unescaped for c in "*?[")


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
            size_info = f"  ({_fmt_size(st.st_size)})" if entry.is_file() else ""
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
