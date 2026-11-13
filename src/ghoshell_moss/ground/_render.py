"""Ground rendering — a position becomes one self-explaining XML face.

布局 (SPEC §6.4):

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
import time
from dataclasses import dataclass

import yaml
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathspec import PathSpec

from ghoshell_moss.ground._addr import Anchor, anchor_kind, is_glob_pattern, resolve_path
from ghoshell_moss.ground._l0 import (
    _FRONTMATTER_RE,
    DEFAULT_L0_FILENAME,
    L0Contents,
    load_l0,
)
from ghoshell_moss.ground._chain import collect_content_files
from ghoshell_moss.ground._hash import (
    GLOB_IGNORE,
    Observation,
    _entry_ignored,
    _path_touches_ignore,
    glob_limited,
    observe,
    parse_range,
)
from ghoshell_moss.ground.contract import (
    ContentPin,
    ExecPin,
    FilePin,
    FrontmatterPin,
    GlobPin,
    LsPin,
    PathOutsideRootError,
    Pin,
)

__all__ = [
    "render_meta", "render_meta_at",
    "render_position", "RenderLimits", "find_ground",
]


@dataclass
class _AtBlock:
    """@-reference 展开出来的一段内容 (渲染期的中间态, 不是契约模型)."""

    label: str
    content: str


def render_meta(
    root: Path,
    doc_path: Path,
    body: str,
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

    # content — 本场正文是否存在 (body 不向上合并)
    lines.append("content: present" if body else "content: none")

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


async def render_meta_at(root: str | Path) -> str:
    """一个目录的场身份 + pin TOC — 无驻留态, 只读盘 (给 ``moss ground meta``)."""
    target = Path(root).expanduser().resolve()
    if not (target / DEFAULT_L0_FILENAME).is_file():
        return f"no GROUND.md under {target}"
    contents = await asyncio.to_thread(load_l0, target)
    return render_meta(
        root=target,
        doc_path=target / DEFAULT_L0_FILENAME,
        body=contents.body,
        pins=contents.pins,
        id_=contents.convention.id,
    )


# -- pin targeting ------------------------------------------------------------


def _pin_target_raw(pin: Pin) -> str:
    """Pin 的目标路径原文 (锚判定用). exec/content 无路径概念, 返回空."""
    if isinstance(pin, GlobPin):
        return pin.arguments.path
    if isinstance(pin, (FilePin, FrontmatterPin, LsPin)):
        return pin.arguments.path
    return ""


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


# -- block construction ---------------------------------------------------


def _build_body_with_at(body: str, ground_dir: Path) -> tuple[str, list[_AtBlock]]:
    """Body → (raw_content, @-children). @ref 保留在文本, 展开结果在 children."""
    children: list[_AtBlock] = []
    for at_ref in _scan_at_refs(body):
        resolved = _resolve_at_ref(at_ref, ground_dir)
        if resolved is not None:
            children.append(_AtBlock(label=at_ref, content=resolved))
    return body, children


def _pin_block_content(
    p: Pin,
    anchor: Anchor,
    obs: Observation | None,
    *,
    ignore: PathSpec | None = None,
) -> tuple[str, list[_AtBlock]]:
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
) -> tuple[str, list[_AtBlock]]:
    """Pin → (content, @-children). 按 pin 类型分发."""
    if isinstance(pin, ExecPin):
        return _content_exec(pin, obs), []
    if isinstance(pin, ContentPin):
        content, children = _build_content_with_at(pin, anchor)
        if pin.arguments.lines is not None:
            content = _apply_lines_cap(content, pin.arguments.lines)
        if pin.arguments.budget is not None:
            content = _apply_budget(content, pin.arguments.budget)
        return content, children
    # file / glob / frontmatter / ls — 无 @-展开
    return _render_pin_content(pin, anchor, obs, ignore=ignore), []


def _build_content_with_at(pin: ContentPin, anchor: Anchor) -> tuple[str, list[_AtBlock]]:
    """Content pin: 收集文件内容 + 每文件的 @-ref 解析为子 ViewBlock."""
    files = collect_content_files(anchor, pin.arguments.filename)
    if not files:
        return "(no files)", []

    blocks: list[str] = []
    children: list[_AtBlock] = []
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
                children.append(_AtBlock(label=at_ref, content=resolved))

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
    elif isinstance(pin, ContentPin):
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

    # ground-level ignore: pre-filter — ignored dirs and files excluded
    visible = [e for e in items if not _entry_ignored(e, ground_root, ignore)]

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




# -- position rendering (SPEC 0.3.0) -----------------------------------------
#
# 渲染的输入是「位置」而不是「场对象」: 先由位置解析出该看哪个场 (最近的
# GROUND.md, 止于 boundary), 再决定画 field-root 面还是 walk 面. 子场由场自己
# 的 ``inline`` 点名递归渲染, 同名不自动铺开. 输出是转义过的 XML 容器, 每一层
# 形状相同 —— 这是自解释的唯一来源.

DEFAULT_RENDER_CHARS = 12000
DEFAULT_RENDER_SECONDS = 5.0
_TEMPLATE_DIR = ".grounds"


@dataclass(frozen=True)
class RenderLimits:
    """一次渲染的两个上界 (SPEC §7.2). 都是整个渲染的, 不是单 pin 的."""

    chars: int = DEFAULT_RENDER_CHARS
    seconds: float = DEFAULT_RENDER_SECONDS


class _Budget:
    def __init__(self, limits: RenderLimits) -> None:
        self.total = limits.chars
        self.used = 0
        self.deadline = time.monotonic() + limits.seconds

    @property
    def remaining_seconds(self) -> float:
        return self.deadline - time.monotonic()

    def fits(self, size: int) -> bool:
        return self.used + size <= self.total

    def charge(self, text: str) -> None:
        self.used += len(text)


def _esc(text: str) -> str:
    """内容转义 — 场内容是数据, 永远不能变成结构 (SPEC §6.4)."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _attr(text: str) -> str:
    return _esc(text).replace('"', "&quot;")


def find_ground(position: Path, boundary: Path) -> Path | None:
    """位置 → 所属场根. 向上找最近的 GROUND.md, 止于 boundary (SPEC §6.1).

    位置在 boundary 之外 → None (边界之上永不咨询).
    """
    pos = position.expanduser().resolve()
    edge = boundary.expanduser().resolve()
    if not pos.is_relative_to(edge):
        return None
    cursor = pos
    while True:
        if (cursor / DEFAULT_L0_FILENAME).is_file():
            return cursor
        if cursor == edge:
            return None
        cursor = cursor.parent


def _is_positional(pin: Pin) -> bool:
    """pin 是否跟随观看位置 (SPEC §6.3.1).

    - `content` 按定义就是从 $CWD 向上收集;
    - 其余看路径参数是否显式锚在 $CWD.
    """
    if isinstance(pin, ContentPin):
        return True
    raw = _pin_target_raw(pin)
    return bool(raw) and anchor_kind(raw) == "cwd"


def _resolve_table(template: str, base: Path, boundary: Path) -> Path | None:
    """模板名或目标 .md → 渲染表文件. 越界/缺失都抛 ValueError (渲染成 error 面)."""
    if not template:
        return None
    raw = Path(template).expanduser()
    looks_like_path = "/" in template or raw.suffix == ".md"
    if looks_like_path:
        # 路径形态按调用者 cwd 解析 (命令行语义), 再受 boundary 约束
        candidate = raw if raw.is_absolute() else raw.resolve()
    else:
        candidate = base / _TEMPLATE_DIR / f"{template}.md"
    candidate = candidate.resolve()
    if not candidate.is_relative_to(boundary.resolve()):
        raise ValueError(f"template outside boundary: {template}")
    if not candidate.is_file():
        raise ValueError(f"template not found: {template}")
    return candidate


def _discover_templates(ground_dir: Path) -> list[tuple[str, str]]:
    """``<root>/.grounds/*.md`` → [(name, description)]. 感知是渲染自带的, 不需要 pin."""
    folder = ground_dir / _TEMPLATE_DIR
    if not folder.is_dir():
        return []
    found: list[tuple[str, str]] = []
    for path in sorted(folder.glob("*.md")):
        description = ""
        try:
            head = path.read_text(encoding="utf-8", errors="replace")[:2000]
            match = _FRONTMATTER_RE.match(head)
            if match is not None:
                meta = yaml.safe_load(match.group("yaml")) or {}
                if isinstance(meta, dict):
                    description = str(meta.get("description") or "")
        except Exception:  # noqa: BLE001 — 模板描述缺失不该拖垮渲染
            description = ""
        found.append((path.stem, description))
    return found


def _error_face(position: Path, reason: str) -> str:
    return (
        f'<ground path="{_attr(str(position))}" status="error" '
        f'reason="{_attr(reason)}"/>'
    )


async def render_position(
    position: str | Path,
    *,
    boundary: str | Path,
    limits: RenderLimits | None = None,
    template: str | None = None,
    _budget: _Budget | None = None,
    _stack: tuple[Path, ...] = (),
) -> str:
    """渲染一个位置 → 自解释 XML 容器 (SPEC §6).

    失败是结果, 不是异常: 位置无场 → status="empty"; 表不可用 / 解析失败 →
    status="error". 两种情况调用方都拿到一段可读文本.
    """
    limits = limits or RenderLimits()
    budget = _budget or _Budget(limits)
    pos = Path(position).expanduser().resolve()
    edge = Path(boundary).expanduser().resolve()

    if not pos.is_relative_to(edge):
        return _error_face(pos, f"outside boundary: {edge}")

    ground = find_ground(pos, edge)
    if ground is None and not template:
        return (
            f'<ground path="{_attr(str(pos))}" status="empty" '
            f'reason="no GROUND.md up to boundary"/>'
        )

    try:
        table_path = _resolve_table(template or "", ground or pos, edge)
    except ValueError as exc:
        return _error_face(pos, str(exc))

    try:
        target = (
            await asyncio.to_thread(load_l0, ground)
            if ground is not None
            else L0Contents.empty()
        )
        source = (
            await asyncio.to_thread(load_l0, table_path.parent, table_path.name)
            if table_path is not None
            else target
        )
    except Exception as exc:  # noqa: BLE001 — SPEC §6.7: 失败是结果
        return _error_face(pos, f"{type(exc).__name__}: {exc}")

    ground_dir = ground or pos
    walk = ground is not None and pos != ground
    cwd = pos if walk else ground_dir

    convention = target.convention
    line = [f'<ground path="{_attr(str(ground_dir))}"']
    if convention.id:
        line.append(f' id="{_attr(convention.id)}"')
    name = convention.name or ground_dir.name
    line.append(f' name="{_attr(name)}"')
    if convention.description:
        line.append(f' description="{_attr(convention.description)}"')
    if table_path is not None:
        line.append(f' table="{_attr(str(table_path))}"')
    line.append(">")
    head = "".join(line)
    if not budget.fits(len(head) + 2):
        return head + "</ground>"
    budget.charge(head)
    parts = [head]

    if walk:
        node = f'  <cwd path="{_attr(str(cwd))}"/>'
        if budget.fits(len(node) + 1):
            budget.charge(node)
            parts.append(node)

    anchor = Anchor(ground=ground_dir, cwd=cwd)
    ignore = _ignore_spec_from(convention, ground_dir)

    # body — 只在场根面出现 (walk 不重发正文)
    if not walk and source.body.strip():
        body_text, at_children = await asyncio.to_thread(
            _build_body_with_at, source.body, ground_dir
        )
        body_text = "\n".join([body_text] + [b.content for b in at_children])
        node = f"  <body>{_esc(body_text.strip())}</body>"
        if budget.fits(len(node) + 1):
            budget.charge(node)
            parts.append(node)

    pins = list(source.pins)
    expanded = [p for p in pins if not walk or _is_positional(p) or p.always_show]
    folded = [p for p in pins if p not in expanded]

    for pin in expanded:
        node = await _render_pin_node(pin, anchor, ignore, budget)
        parts.append(node)

    if folded:
        rows = ["  <folded>"]
        for pin in folded:
            rows.append(
                f'    <pin verb="{_attr(pin.verb)}" label="{_attr(pin.label)}"'
                + (f' desc="{_attr(pin.description)}"' if pin.description else "")
                + "/>"
            )
        rows.append("  </folded>")
        node = "\n".join(rows)
        if budget.fits(len(node) + 1):
            budget.charge(node)
            parts.append(node)

    # imports — 本场声明要 import 的子目录 (SPEC §3.1/§6.6).
    # 只按声明走: 不发现, 不猜测; 声明里没有的目录不会出现. 子目录不必是场.
    # 记账: 子渲染自己扣预算, 这里不再重复扣 (否则同一段内容记两次).
    imports_rows: list[str] = []
    for entry in (source.convention.imports or []):
        child_dir = (ground_dir / str(entry)).resolve()
        if budget.used >= budget.total:
            imports_rows.append(
                f'  <truncated reason="budget" path="{_attr(str(child_dir))}"/>'
            )
            break
        if child_dir in _stack or child_dir == ground_dir:
            imports_rows.append(
                f'  <truncated reason="cycle" path="{_attr(str(child_dir))}"/>'
            )
            continue
        if not child_dir.is_relative_to(edge):
            imports_rows.append(
                f'  <truncated reason="outside" path="{_attr(str(child_dir))}"/>'
            )
            continue
        inner = await render_position(
            child_dir,
            boundary=edge,
            limits=limits,
            _budget=budget,
            _stack=_stack + (ground_dir,),
        )
        imports_rows.append("\n".join("  " + ln for ln in inner.splitlines()))

    if imports_rows:
        node = "  <imports>\n" + "\n".join(imports_rows) + "\n  </imports>"
        budget.charge(node)
        parts.append(node)

    templates = await asyncio.to_thread(_discover_templates, ground_dir)
    if templates:
        rows = ["  <templates>"]
        for tname, tdesc in templates:
            rows.append(
                f'    <template name="{_attr(tname)}"'
                + (f' desc="{_attr(tdesc)}"' if tdesc else "")
                + "/>"
            )
        rows.append("  </templates>")
        node = "\n".join(rows)
        if budget.fits(len(node) + 1):
            budget.charge(node)
            parts.append(node)

    parts.append("</ground>")
    return "\n".join(parts)


async def _render_pin_node(
    pin: Pin,
    anchor: Anchor,
    ignore: PathSpec | None,
    budget: _Budget,
) -> str:
    """一枚 pin → `<pin>` 节点. 墙钟耗尽 → status="timeout" (SPEC §7.2)."""
    attrs = f'verb="{_attr(pin.verb)}" label="{_attr(pin.label)}"'
    if pin.description:
        attrs += f' desc="{_attr(pin.description)}"'

    remaining = budget.remaining_seconds
    if remaining <= 0:
        return f'  <pin {attrs} status="timeout"/>'

    try:
        obs = await asyncio.wait_for(
            observe(pin, anchor, ignore=ignore), timeout=remaining
        )
    except asyncio.TimeoutError:
        return f'  <pin {attrs} status="timeout"/>'

    content, at_children = await asyncio.to_thread(
        _pin_block_content, pin, anchor, obs, ignore=ignore
    )
    text = content.strip()
    if at_children:
        text = "\n".join([text] + [b.content for b in at_children]).strip()

    body = f"  <pin {attrs}>{_esc(text)}</pin>"
    if not budget.fits(len(body) + 1):
        return f'  <pin {attrs} status="truncated"/>'
    budget.charge(body)
    return body


def _ignore_spec_from(convention, ground_dir: Path) -> PathSpec | None:
    # pathspec 只在真配了 ignore 时才需要 (TYPE_CHECKING 的导入是给注解用的)
    from pathspec import PathSpec as _PathSpec

    patterns: list[str] = list(convention.ignore or [])
    if convention.ignore_file:
        path = ground_dir / convention.ignore_file
        if path.is_file():
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                lines = []
            patterns.extend(
                ln for ln in lines if ln.strip() and not ln.strip().startswith("#")
            )
    if not patterns:
        return None
    return _PathSpec.from_lines("gitignore", patterns)
