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
import re
from pathlib import Path

from ghoshell_moss.ground._addr import Anchor, anchor_kind, resolve_path
from ghoshell_moss.ground._chain import collect_law_files
from ghoshell_moss.ground._hash import GLOB_IGNORE, Observation, PinShadow, observe, parse_range
from ghoshell_moss.ground.contract import (
    ExecPin,
    FilePin,
    FrameItem,
    FrontmatterPin,
    GlobPin,
    LawPin,
    LsPin,
    Pin,
)

__all__ = ["render_context", "render_meta", "render_walk", "render_items"]


async def render_context(
    body: str,
    pins: list[Pin],
    shadows: dict[str, PinShadow],
    anchor: Anchor,
) -> list[FrameItem]:
    """Render a frame → list[FrameItem].

    Body 和 pin 结果各成一个 FrameItem. @-ref 展开结果作为 children
    嵌套在父 item 内. 所有 pin 观察并行 (asyncio.gather).
    """
    items: list[FrameItem] = []

    # ---- body item ------------------------------------------------------
    if body.strip():
        body_content, body_children = _build_body_with_at(body, anchor.ground)
        items.append(FrameItem(
            kind="body",
            label="body",
            content=body_content.rstrip(),
            brief=_fmt_text_brief(body_content),
            children=body_children,
        ))

    # ---- observe all pins (parallel) -----------------------------------
    if not pins:
        return items

    tasks = {p.label: observe(p, anchor) for p in pins}
    results = await asyncio.gather(*tasks.values())
    observations: dict[str, Observation] = dict(zip(tasks.keys(), results))

    # ---- pin items -----------------------------------------------------
    for p in pins:
        obs = observations.get(p.label)
        shadow = shadows.get(p.label, PinShadow())

        # 位置依赖 pin (law) 不做 stale 对账
        stale = (
            not p.is_cwd_anchored
            and shadow.hash is not None
            and obs is not None
            and obs.exists
            and obs.hash != shadow.hash
        )
        missing = obs is not None and not obs.exists

        content, at_children = _build_pin_content(p, anchor, obs)
        if stale:
            content = content + "\n[changed on disk]"
        if missing:
            content = "[missing]"

        items.append(FrameItem(
            kind=p.verb,
            label=p.label,
            content=content.rstrip() if content.strip() else content,
            brief=_pin_brief(p, obs, content),
            truncated=_pin_truncated(p, content),
            meta=_pin_meta(p, obs, anchor),
            children=at_children,
        ))

    return items


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
    shadows: dict[str, PinShadow],
    *,
    label: str | None = None,
) -> list[FrameItem]:
    """场内移动视图 → list[FrameItem].

    - 站立位置 header 作为 body item
    - $CWD 锚 pins 展开为 pin items
    - 其余 pins 折叠为 body item (TOC)
    """
    anchor = Anchor(ground=ground_root, cwd=cwd)
    rel_doc = os.path.relpath(doc_path, cwd)
    display = label or ground_root.name

    items: list[FrameItem] = []

    # walk header
    items.append(FrameItem(
        kind="body",
        label="walk",
        content=(
            f"ground: {display}  (law: {doc_path})\n"
            f"cwd: {cwd}"
        ),
        brief="",
    ))

    # $CWD 锚 pins 展开 (含位置依赖的 law); 其余折叠
    cwd_pins = [
        p for p in pins
        if p.is_cwd_anchored or anchor_kind(_pin_target_raw(p)) == "cwd"
    ]
    folded = [p for p in pins if p not in cwd_pins]

    if cwd_pins:
        tasks = {p.label: observe(p, anchor) for p in cwd_pins}
        results = await asyncio.gather(*tasks.values())
        observations = dict(zip(tasks.keys(), results))
        for p in cwd_pins:
            obs = observations.get(p.label)
            shadow = shadows.get(p.label, PinShadow())
            stale = (
                not p.is_cwd_anchored
                and shadow.hash is not None
                and obs is not None
                and obs.exists
                and obs.hash != shadow.hash
            )
            missing = obs is not None and not obs.exists

            content, at_children = _build_pin_content(p, anchor, obs)
            if stale:
                content = content + "\n[changed on disk]"
            if missing:
                content = "[missing]"

            items.append(FrameItem(
                kind=p.verb,
                label=p.label,
                content=content.rstrip() if content.strip() else content,
                brief=_pin_brief(p, obs, content),
                truncated=_pin_truncated(p, content),
                meta=_pin_meta(p, obs, anchor),
                children=at_children,
            ))

    if folded:
        toc_lines = [f"pins@{display} (moss ground frame {rel_doc.removesuffix('/GROUND.md') or '.'}):"]
        for p in folded:
            desc = f"  # {p.description}" if p.description else ""
            toc_lines.append(f"  {p.label}:{p.verb}({_pin_kwargs(p)}){desc}")
        items.append(FrameItem(
            kind="body",
            label="folded",
            content="\n".join(toc_lines),
            brief="",
        ))

    return items


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


def _content_law(pin: LawPin, anchor: Anchor) -> str:
    """law pin 渲染 — 从 cwd 向上收集约定文件, root-first 展示.

    每文件一块, ``-- {rel}`` 标注相对场根的路径. 不做 @-展开 —
    展开交给 ``_build_law_children`` 生成子 FrameItem.
    也不做截断 — 截断由上层 FrameItem 组装时统一处理.
    """
    files = collect_law_files(anchor, pin.arguments.filename)
    if not files:
        return "(no files)"

    blocks: list[str] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            text = "(unreadable)"
        rel = str(f.relative_to(anchor.ground)) if f.is_relative_to(anchor.ground) else str(f)
        blocks.append(f"-- {rel}\n{text.rstrip()}")

    return "\n\n".join(blocks)


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
        try:
            start, end = parse_range(pin.arguments.range, len(lines_list))
        except ValueError:
            return "error: invalid range (start beyond file end or descending)"
        text = "\n".join(lines_list[start - 1 : end])

    return _apply_budget(text, pin.arguments.budget)


def _content_glob(pin: GlobPin, anchor: Anchor) -> str:
    root = anchor.ground
    pattern = pin.arguments.path
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


# -- FrameItem construction ------------------------------------------------


def _build_body_with_at(body: str, ground_dir: Path) -> tuple[str, list[FrameItem]]:
    """Body → (raw_content, @-children). @ref 保留在文本, 解析结果在 children."""
    children: list[FrameItem] = []
    for at_ref in _scan_at_refs(body):
        resolved = _resolve_at_ref(at_ref, ground_dir)
        if resolved is not None:
            children.append(FrameItem(
                kind="@",
                label=at_ref,
                content=resolved,
                brief=_fmt_text_brief(resolved),
                meta={"from": "GROUND.md"},
            ))
    return body, children


def _build_pin_content(
    pin: Pin, anchor: Anchor, obs: Observation | None,
) -> tuple[str, list[FrameItem]]:
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
    return _render_pin_content(pin, anchor, obs), []


def _build_law_with_at(pin: LawPin, anchor: Anchor) -> tuple[str, list[FrameItem]]:
    """Law pin: 收集文件内容 + 每文件的 @-ref 解析为子 FrameItem.

    @ref 保留在原文不展开, 解析结果放在 children. 每文件有独立 base_dir.
    """
    files = collect_law_files(anchor, pin.arguments.filename)
    if not files:
        return "(no files)", []

    blocks: list[str] = []
    children: list[FrameItem] = []
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
                children.append(FrameItem(
                    kind="@",
                    label=at_ref,
                    content=resolved,
                    brief=_fmt_text_brief(resolved),
                    meta={"from": str(rel)},
                ))

    content = "\n\n".join(blocks)
    return content, children


# -- @-reference scanning ---------------------------------------------------


def _scan_at_refs(text: str) -> list[str]:
    """扫描文本中的 @-reference 路径列表 (去重, 保持出现顺序).

    SPEC §6.2: @ 在行首或空白后, 后跟 [A-Za-z0-9_./-] 组成的 token.
    fenced code block 内不识别. 不递归 — 返回的是本层 @ref 清单.
    """
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
    """解析一个 @-reference → 文件内容. 找不到 / 不可读 → None."""
    target = base_dir / ref
    if not target.is_file():
        return None
    try:
        return target.read_text(encoding="utf-8", errors="replace").rstrip()
    except OSError:
        return None


# -- item metadata helpers ---------------------------------------------------


def _fmt_text_brief(text: str) -> str:
    """一行内容摘要 — 字符数和行数."""
    chars = len(text)
    lines = text.count("\n") + 1
    abbrev = _fmt_size(chars).replace("B", "")
    return f"{abbrev}, {lines} lines"


def _is_status_content(content: str) -> bool:
    """非内容块 — 状态 / 错误 / 哨兵消息, 不应展示摘要."""
    c = content.strip()
    return (c.startswith("[") or c.startswith("error:") or not c)


def _pin_brief(pin: Pin, obs: Observation | None, content: str) -> str:
    """Pin 的一行摘要, 适应不同 verb. 状态/错误块返回空 — 不提供噪音摘要."""
    if _is_status_content(content):
        return ""
    if isinstance(pin, LawPin):
        n = (obs.size if obs and obs.size else 0)
        file_label = f"{n} files, " if n else ""
        return file_label + _fmt_text_brief(content)
    if obs and obs.size is not None:
        size = f"{obs.size} {obs.unit}"
        if obs.size > 0 and obs.unit == "B":
            return _fmt_text_brief(content)
        return f"{size}, " + _fmt_text_brief(content) if content else str(size)
    return _fmt_text_brief(content)


def _pin_truncated(pin: Pin, content: str) -> bool:
    """内容是否被截断 — 通过检查 truncation markers."""
    return "[truncated at" in content


def _pin_meta(pin: Pin, obs: Observation | None, anchor: Anchor) -> dict:
    """Pin 的附加上下文, 按 verb 不同."""
    if isinstance(pin, LawPin):
        n = obs.size if obs else 0
        return {
            "filename": pin.arguments.filename,
            "files": n,
            "budget": pin.arguments.budget,
            "lines": pin.arguments.lines,
        }
    if isinstance(pin, FilePin):
        return {"path": pin.arguments.path, "budget": pin.arguments.budget}
    if isinstance(pin, LsPin):
        return {"path": pin.arguments.path, "depth": pin.arguments.depth}
    if isinstance(pin, GlobPin):
        return {"path": pin.arguments.path}
    if isinstance(pin, FrontmatterPin):
        return {"path": pin.arguments.path}
    if isinstance(pin, ExecPin):
        return {"ref": pin.arguments.ref}
    return {}


# -- text serialization -----------------------------------------------------


def render_items(items: list[FrameItem], *, ground_path: str | None = None) -> str:
    """FrameItem 列表 → 文本 (``---`` + ``>`` 分隔符语法).

    每个 top-level item 以 ``---`` 开闭, 间以空行. @-children
    嵌套在父 item 的开闭区间内. 裸文本(不知道 ground 协议)和
    渲染 markdown 的读者都能识别区块边界.
    """
    out: list[str] = []
    if ground_path is not None:
        out.append(f"$GROUND: {ground_path}")
        out.append("")
    for i, item in enumerate(items):
        _render_item(out, item)
        if i < len(items) - 1:
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def _render_item(out: list[str], item: FrameItem) -> None:
    """递归渲染一个 item 及其 children."""
    # open marker
    out.append("---")
    out.append(_item_open_line(item))
    out.append("---")
    out.append("")

    # content
    if item.content.strip():
        out.append(item.content.rstrip())
        out.append("")

    # children (@-expansion sub-blocks)
    for child in item.children:
        _render_item(out, child)

    # close marker
    tail = item.brief
    if item.truncated:
        tail = f"{tail}, truncated" if tail else "truncated"
    out.append("---")
    out.append(_item_close_line(item, tail))
    out.append("---")


def _item_open_line(item: FrameItem) -> str:
    """开头 ``>`` 行."""
    if item.kind == "body":
        return f"> body:{item.label}" if item.label != "body" else "> body"
    if item.kind == "@":
        src = item.meta.get("from", "")
        from_part = f"  from:{src}" if src else ""
        return f"> @{item.label}{from_part}"
    # pin: type:label  [minimal meta]
    extra = _item_meta_hint(item)
    return f"> {item.kind}:{item.label}{extra}"


def _item_close_line(item: FrameItem, tail: str) -> str:
    """结尾 ``>`` 行."""
    if item.kind == "body":
        label = f"body:{item.label}" if item.label != "body" else "body"
        return f"> {label} end  {tail}" if tail else f"> {label} end"
    if item.kind == "@":
        return f"> @{item.label} end  {tail}" if tail else f"> @{item.label} end"
    return f"> {item.kind}:{item.label} end  {tail}" if tail else f"> {item.kind}:{item.label} end"


def _item_meta_hint(item: FrameItem) -> str:
    """Pin 开头的可选元信息, 只显示最少的 — 不加参数."""
    if item.kind == "law":
        n = item.meta.get("files", 0)
        return f"  files:{n}" if n else ""
    return ""


# -- meta helpers (legacy) --------------------------------------------------


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


def _apply_lines_cap(text: str, lines: int | None) -> str:
    if lines is None:
        return text
    ls = text.splitlines()
    if len(ls) <= lines:
        return text
    return "\n".join(ls[:lines]) + f"\n[truncated at {lines} lines]"


# @-token regex — 共用: _scan_at_refs (新) + 旧排版保留兼容.
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
    import re
    fm = re.match(r"\A---\s*\n(.*?)\n---", text, re.DOTALL)
    return fm.group(1) if fm else None


def _has_glob(raw: str) -> bool:
    unescaped = raw.replace("\\$", "$")
    return any(c in unescaped for c in "*?[")


# -- general helpers ------------------------------------------------------


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
