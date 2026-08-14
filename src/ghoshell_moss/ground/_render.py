"""Ground rendering — produces RenderedView (header + blocks).

RenderedView carries both structured data (--json) and self-explanatory
markdown (str/view.to_markdown()).  Layout per user format:

    ---
    name: <ground name>
    $GROUND: <path>
    $CWD: <path>            # walk only
    ---

    <body verbatim — no wrapping>

    ---
    <!-- file-greeting: welcome -->
    <pin content>

    ---
    <!-- at: @file.md -->
    <@-reference content>

Meta is a separate path (render_meta) for ``moss ground meta``.
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathspec import PathSpec

from ghoshell_moss.ground._addr import Anchor, anchor_kind, is_glob_pattern, resolve_path
from ghoshell_moss.ground._chain import collect_law_files
from ghoshell_moss.ground._hash import (
    GLOB_IGNORE,
    Observation,
    _path_touches_ignore,
    glob_limited,
    observe,
    parse_range,
)
from ghoshell_moss.ground.contract import (
    ExecPin,
    FilePin,
    FrontmatterPin,
    GlobPin,
    LawPin,
    LsPin,
    PathOutsideRootError,
    Pin,
    RenderedView,
    ViewBlock,
    ViewHeader,
)

__all__ = ["render_context", "render_meta", "render_walk"]


async def render_context(
    body: str,
    pins: list[Pin],
    anchor: Anchor,
    *,
    ground_name: str | None = None,
    ground_description: str | None = None,
    ground_id: str | None = None,
    ignore: PathSpec | None = None,
) -> RenderedView:
    """Render a ground at its root → RenderedView.

    Body 和 pin 结果各自成 ViewBlock, @-ref 展开为独立 at 块.
    观察并行 (asyncio.gather); 内容构建整体卸载到线程池, 不阻塞 loop.
    """
    header = ViewHeader(
        id=ground_id,
        name=ground_name or anchor.ground.name,
        description=ground_description,
        ground_path=str(anchor.ground),
    )

    observations: dict[str, Observation] = {}
    if pins:
        tasks = {p.label: observe(p, anchor, ignore=ignore) for p in pins}
        results = await asyncio.gather(*tasks.values())
        observations = dict(zip(tasks.keys(), results))

    blocks = await asyncio.to_thread(
        _assemble_context_blocks, body, pins, anchor, observations, ignore,
    )
    return RenderedView(header=header, blocks=blocks)


def _assemble_context_blocks(
    body: str,
    pins: list[Pin],
    anchor: Anchor,
    observations: dict[str, Observation],
    ignore: PathSpec | None,
) -> list[ViewBlock]:
    """同步组装 body + pin 块 — 含文件 IO, 由 to_thread 调用."""
    blocks: list[ViewBlock] = []

    if body.strip():
        body_content, at_children = _build_body_with_at(body, anchor.ground)
        blocks.append(ViewBlock(kind="body", label="body", content=body_content.rstrip()))
        blocks.extend(at_children)

    for p in pins:
        obs = observations.get(p.label)
        content, at_children = _pin_block_content(p, anchor, obs, ignore=ignore)

        blocks.append(ViewBlock(
            kind="pin",
            label=p.label,
            verb=p.verb,
            description=p.description or None,
            content=content.strip(),
        ))
        blocks.extend(at_children)

    return blocks


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

    Used by ``moss ground meta``.  Separated from render so consumers
    who don't need ground protocol get a clean content-only view.
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


# -- walk (场内移动) ----------------------------------------------------------


def _pin_target_raw(pin: Pin) -> str:
    """Pin 的目标路径原文 (锚判定用). exec/law 无路径概念, 返回空."""
    if isinstance(pin, GlobPin):
        return pin.arguments.path
    if isinstance(pin, (FilePin, FrontmatterPin, LsPin)):
        return pin.arguments.path
    return ""


async def render_walk(
    cwd: Path,
    ground_root: Path,
    doc_path: Path,
    pins: list[Pin],
    *,
    label: str | None = None,
    ground_id: str | None = None,
    ignore: PathSpec | None = None,
) -> RenderedView:
    """场内移动视图 → RenderedView.

    - 场根 + 站立位置进入 header ($CWD 字段)
    - $CWD 锚 pins 展开为 pin 块
    - 法链 (law) walk 时默认只列路径
    - 其余 pins 折叠为一个 folded 块 (TOC)
    """
    anchor = Anchor(ground=ground_root, cwd=cwd)
    display = label or ground_root.name

    header = ViewHeader(
        id=ground_id,
        name=display,
        ground_path=str(ground_root),
        cwd=str(cwd),
    )

    # $CWD 锚 pins 展开; 其余折叠
    cwd_pins = [
        p for p in pins
        if p.is_cwd_anchored or anchor_kind(_pin_target_raw(p)) == "cwd"
    ]
    folded = [p for p in pins if p not in cwd_pins]

    law_full = [p for p in cwd_pins if isinstance(p, LawPin) and p.always_show]
    other = [p for p in cwd_pins if not isinstance(p, LawPin)]
    full_pins = law_full + other

    observations: dict[str, Observation] = {}
    if full_pins:
        tasks = {p.label: observe(p, anchor, ignore=ignore) for p in full_pins}
        results = await asyncio.gather(*tasks.values())
        observations = dict(zip(tasks.keys(), results))

    blocks = await asyncio.to_thread(
        _assemble_walk_blocks, anchor, cwd_pins, folded, full_pins, observations, display, ignore,
    )
    return RenderedView(header=header, blocks=blocks)


def _assemble_walk_blocks(
    anchor: Anchor,
    cwd_pins: list[Pin],
    folded: list[Pin],
    full_pins: list[Pin],
    observations: dict[str, Observation],
    display: str,
    ignore: PathSpec | None,
) -> list[ViewBlock]:
    """同步组装 walk 视图块 — law_compact + full_pins + folded TOC."""
    blocks: list[ViewBlock] = []

    if cwd_pins:
        law_compact = [p for p in cwd_pins if isinstance(p, LawPin) and not p.always_show]

        for p in law_compact:
            law_files = collect_law_files(anchor, p.arguments.filename)
            if law_files:
                rels = [
                    str(f.relative_to(anchor.ground)) if f.is_relative_to(anchor.ground) else str(f)
                    for f in law_files
                ]
                content = "\n".join(rels)
            else:
                content = "(no files)"
            blocks.append(ViewBlock(
                kind="pin",
                label=p.label,
                verb=p.verb,
                description=p.description or None,
                content=content,
                meta={
                    "filename": p.arguments.filename,
                    "files": len(law_files),
                },
            ))

        for p in full_pins:
            obs = observations.get(p.label)
            content, at_children = _pin_block_content(p, anchor, obs, ignore=ignore)

            blocks.append(ViewBlock(
                kind="pin",
                label=p.label,
                verb=p.verb,
                description=p.description or None,
                content=content.strip(),
            ))
            blocks.extend(at_children)

    if folded:
        toc_lines = [f"pins at {display}:"]
        for p in folded:
            desc = f"  # {p.description}" if p.description else ""
            toc_lines.append(f"  {p.label}:{p.verb}({_pin_kwargs(p)}){desc}")
        blocks.append(ViewBlock(
            kind="folded",
            label="pins",
            content="\n".join(toc_lines),
        ))

    return blocks


def _render_pin_content(
    pin: Pin,
    anchor: Anchor,
    obs: Observation | None = None,
    *,
    ignore: PathSpec | None = None,
) -> str:
    """Dispatch per pin subclass."""
    if isinstance(pin, FilePin):
        return _content_file(pin, anchor)
    if isinstance(pin, GlobPin):
        return _content_glob(pin, anchor, ignore=ignore)
    if isinstance(pin, FrontmatterPin):
        return _content_frontmatter(pin, anchor, ignore=ignore)
    if isinstance(pin, LsPin):
        return _content_ls(pin, anchor, ignore=ignore)
    if isinstance(pin, ExecPin):
        return _content_exec(pin, obs)
    return f"error: unknown pin type: {type(pin).__name__}"


# -- per-kind content renderers -----------------------------------------------


def _content_exec(pin: ExecPin, obs: Observation | None) -> str:
    """观察阶段已执行, 直接消费 payload — 一次渲染只跑一次进程."""
    if obs is None or obs.payload is None:
        return "[not yet observed]"
    return _apply_budget(obs.payload, pin.arguments.budget)


def _content_file(pin: FilePin, anchor: Anchor) -> str:
    target = resolve_path(pin.arguments.path, anchor)
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "error: unreadable"

    if pin.arguments.range is not None:
        lines_list = text.splitlines()
        try:
            start, end = parse_range(pin.arguments.range, len(lines_list))
        except ValueError:
            return "error: invalid range (start beyond file end or descending)"
        text = "\n".join(lines_list[start - 1 : end])

    return _apply_budget(text, pin.arguments.budget)


def _content_glob(
    pin: GlobPin, anchor: Anchor, *, ignore: PathSpec | None = None,
) -> str:
    root = anchor.ground
    pattern = pin.arguments.path
    if pattern.startswith("$"):
        try:
            resolved = resolve_path(pattern, anchor)
            pattern = str(resolved.relative_to(root))
        except (ValueError, OSError):
            return "error: invalid glob path"

    hits = glob_limited(root, pattern, recursion=pin.arguments.max_depth, ignore=ignore)
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


def _content_frontmatter(
    pin: FrontmatterPin, anchor: Anchor, *, ignore: PathSpec | None = None,
) -> str:
    path_raw = pin.arguments.path

    # Pattern mode: path contains glob characters
    if is_glob_pattern(path_raw):
        return _content_frontmatter_pattern(pin, anchor, ignore=ignore)

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


def _content_frontmatter_pattern(
    pin: FrontmatterPin, anchor: Anchor, *, ignore: PathSpec | None = None,
) -> str:
    import re

    root = anchor.ground
    pattern = pin.arguments.path
    if pattern.startswith("$"):
        try:
            resolved = resolve_path(pattern, anchor)
            pattern = str(resolved.relative_to(root))
        except (ValueError, OSError):
            return "error: invalid pattern"

    hits = glob_limited(
        root, pattern,
        recursion=pin.arguments.max_depth, stop_on_match=True, ignore=ignore,
    )
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


def _content_ls(
    pin: LsPin, anchor: Anchor, *, ignore: PathSpec | None = None,
) -> str:
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
    _walk_ls_entries(root_dir, effective_depth, "", entries,
                     ignore=ignore, ground_root=anchor.ground)

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


# -- ViewBlock construction ---------------------------------------------------


def _build_body_with_at(body: str, ground_dir: Path) -> tuple[str, list[ViewBlock]]:
    """Body → (raw_content, @-children). @ref 保留在文本, 展开结果在 children."""
    children: list[ViewBlock] = []
    for at_ref in _scan_at_refs(body):
        resolved = _resolve_at_ref(at_ref, ground_dir)
        if resolved is not None:
            children.append(ViewBlock(
                kind="at",
                label=at_ref,
                content=resolved,
                meta={"from": "GROUND.md"},
            ))
    return body, children


def _pin_block_content(
    p: Pin,
    anchor: Anchor,
    obs: Observation | None,
    *,
    ignore: PathSpec | None = None,
) -> tuple[str, list[ViewBlock]]:
    """把一枚 pin 的观察结果渲染成 (content, @-children).

    observe 层的结构化状态在此统一分流 — 这是 render 与 observe 的接缝,
    "可读/不可读" 的判定只在这里发生:
    - error  → observe 层捕获的失败 (越界 / IO)
    - missing → 目标不存在
    - binary → 目标存在但不可读 (二进制)
    - 其余   → 读内容
    """
    if obs is not None and obs.error is not None:
        return f"error: {obs.error}", []
    if obs is not None and not obs.exists:
        return "[missing]", []
    if obs is not None and obs.is_binary:
        return "[binary file, not rendered]", []
    return _build_pin_content(p, anchor, obs, ignore=ignore)


def _build_pin_content(
    pin: Pin,
    anchor: Anchor,
    obs: Observation | None,
    *,
    ignore: PathSpec | None = None,
) -> tuple[str, list[ViewBlock]]:
    """Pin → (content, @-children). 按 pin 类型分发."""
    if isinstance(pin, ExecPin):
        return _content_exec(pin, obs), []
    if isinstance(pin, LawPin):
        content, children = _build_law_with_at(pin, anchor)
        if pin.arguments.lines is not None:
            content = _apply_lines_cap(content, pin.arguments.lines)
        if pin.arguments.budget is not None:
            content = _apply_budget(content, pin.arguments.budget)
        return content, children
    # file / glob / frontmatter / ls — 无 @-展开
    return _render_pin_content(pin, anchor, obs, ignore=ignore), []


def _build_law_with_at(pin: LawPin, anchor: Anchor) -> tuple[str, list[ViewBlock]]:
    """Law pin: 收集文件内容 + 每文件的 @-ref 解析为子 ViewBlock."""
    files = collect_law_files(anchor, pin.arguments.filename)
    if not files:
        return "(no files)", []

    blocks: list[str] = []
    children: list[ViewBlock] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            text = "(unreadable)"
        rel = str(f.relative_to(anchor.ground)) if f.is_relative_to(anchor.ground) else str(f)
        blocks.append(f"-- {rel}\n{text.rstrip()}")

        for at_ref in _scan_at_refs(text):
            resolved = _resolve_at_ref(at_ref, f.parent)
            if resolved is not None:
                children.append(ViewBlock(
                    kind="at",
                    label=at_ref,
                    content=resolved,
                    meta={"from": str(rel)},
                ))

    content = "\n\n".join(blocks)
    return content, children


# -- @-reference scanning -----------------------------------------------------


def _scan_at_refs(text: str) -> list[str]:
    """扫描文本中的 @-reference 路径列表 (去重, 保持出现顺序)."""
    refs: list[str] = []
    seen: set[str] = set()
    in_fence = False
    for line in text.splitlines():
        lt = line.lstrip()
        if lt.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in _AT_TOKEN_RE.finditer(line):
            ref = m.group(2)
            if ref.startswith('"') and ref.endswith('"') and len(ref) >= 2:
                ref = ref[1:-1]
            if ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs


def _resolve_at_ref(ref: str, base_dir: Path) -> str | None:
    """解析一个 @-reference → 文件内容. 找不到 / 不可读 / 越界 → None.

    用 resolve_path 施加 §8 子树约束: ``@../outside.md`` 这类逃逸
    不再被读取, 与 SPEC §6.1 "path escapes anchor subtree" 对齐.
    """
    anchor = Anchor(ground=base_dir, cwd=base_dir)
    try:
        target = resolve_path(ref, anchor)
    except PathOutsideRootError:
        return None
    if not target.is_file():
        return None
    try:
        return target.read_text(encoding="utf-8", errors="replace").rstrip()
    except OSError:
        return None


# -- pin kwargs display (meta + folded TOC) -----------------------------------


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
        parts.append(f'path="{pin.arguments.path}"')
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
    elif isinstance(pin, LawPin):
        parts.append(f'filename="{pin.arguments.filename}"')
        if pin.arguments.budget is not None:
            parts.append(f"budget={pin.arguments.budget}")
        if pin.arguments.lines is not None:
            parts.append(f"lines={pin.arguments.lines}")
    return ", ".join(parts)


# -- format helpers -----------------------------------------------------------


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


def _apply_lines_cap(text: str, lines: int | None) -> str:
    if lines is None:
        return text
    ls = text.splitlines()
    if len(ls) <= lines:
        return text
    return "\n".join(ls[:lines]) + f"\n[truncated at {lines} lines]"


# @-token regex
_AT_TOKEN_RE = re.compile(r'(^|\s)@("[^"\n]+"|[A-Za-z0-9_./-]+)')


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
    fm = re.match(r"\A---\s*\n(.*?)\n---", text, re.DOTALL)
    return fm.group(1) if fm else None


# -- general helpers ----------------------------------------------------------


def _walk_ls_entries(
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

    # ground-level ignore: pre-filter — ignored dirs excluded from listing
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
        try:
            st = entry.stat()
            size_info = f"  ({_fmt_size(st.st_size)})" if entry.is_file() else ""
        except OSError:
            size_info = ""
        entries.append(f"{prefix}{connector}{entry.name}{marker}{size_info}")
        if entry.is_dir() and depth > 1:
            sub_prefix = prefix + ("    " if is_last else "│   ")
            _walk_ls_entries(entry, depth - 1, sub_prefix, entries,
                             ignore=ignore, ground_root=ground_root)


