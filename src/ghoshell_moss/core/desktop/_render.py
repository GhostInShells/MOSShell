"""Frame rendering for Ground.context().

一帧渲染的结构 (契合 K14 virtual channel 的 context_messages 消费):

    ground: <label> @ <root>   pins: <n>

    tree(depth=<d>):
      <tree>

    ⚖ <path> (instruction, unloaded)     # 若 hint_children

    ⚠ context over budget: ...            # 若超预算 (K20 报账)

    ── pin: <addr>   [changed on disk]?   [missing]?
       note: <note>
       <内容渲染: 全文 / 行区间 / glob 命中清单>

async: 顶层 render_context 是 async, 内部 tree/hints/read 用 asyncio.gather
或 to_thread 并行, 符合 ABC 的 async 契约声明.

内容截断: **不做**. K20 明说超预算只报账不动手, 由模型自主 unpin. 全文/
行区间照实渲染. 有意的形状 — 别加自动截断.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ghoshell_moss.contracts.desktop import GroundConvention, Pin
from ghoshell_moss.core.desktop._addr import (
    ParsedAddr,
    parse_addr,
    resolve_file_addr,
    resolve_glob_addr,
)
from ghoshell_moss.core.desktop._hash import Observation, observe

__all__ = ["render_context", "BUILTIN_TREE_IGNORE"]


# tree 段的 built-in 过滤集. K16 判据: 完整 gitignore 语义 (`**/`, `!`) 对 tree
# 呈现毫无价值, 且引 pathspec 依赖不值当. 这一层只做 basename 精确匹配 +
# GroundConvention.tree_ignore_extra 提供加法口. K9 未来 pin bash 承接更精细
# 过滤 (`find | grep -v ...`) 后, 这里不需要升级.
BUILTIN_TREE_IGNORE: frozenset[str] = frozenset({
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


async def render_context(
    root: Path,
    label: str,
    convention: GroundConvention,
    pins: list[Pin],
    *,
    workspace_root: Path | None = None,
    l0_file_exists: bool = False,
    l0_filename: str = "DESKTOP.md",
) -> str:
    """渲染桌面当前帧."""
    root_abs = root.resolve()

    # 观察全部 pin (并行)
    parsed_pins: list[tuple[Pin, ParsedAddr]] = [(p, parse_addr(p.addr)) for p in pins]
    observations: list[Observation] = list(
        await asyncio.gather(
            *(observe(parsed, root_abs) for _, parsed in parsed_pins)
        )
    ) if parsed_pins else []

    # 三个静态段并行 (通过 to_thread)
    ignore_names = BUILTIN_TREE_IGNORE | set(convention.tree_ignore_extra)
    tree_task = asyncio.create_task(
        asyncio.to_thread(
            _render_tree, root_abs, convention.tree_depth, ignore_names
        )
    ) if convention.tree_depth > 0 else None
    hints_task = asyncio.create_task(
        asyncio.to_thread(_find_child_hints, root_abs, convention)
    ) if convention.hint_children else None

    tree_str = await tree_task if tree_task else ""
    hints = await hints_task if hints_task else []

    # 渲染 pin blocks (同步, 观察已完成)
    pin_blocks: list[str] = []
    for (pin, parsed), obs in zip(parsed_pins, observations):
        pin_blocks.append(_render_pin(pin, parsed, obs, root_abs))

    # 报账 (K20): 只在超预算时插入警告行
    total_body = sum(len(b) for b in pin_blocks)
    budget = convention.context_budget
    over_budget = total_body > budget

    # 组装. K16 head 承担元信息: root / workspace / L0 status / pins / budget
    l0_status = "exists" if l0_file_exists else "defaults (no file)"
    pct = int(round(total_body / budget * 100)) if budget > 0 else 0
    lines: list[str] = [
        f"ground: {label} @ {root_abs}",
    ]
    if workspace_root is not None:
        lines.append(f"workspace: {workspace_root}")
    lines.append(
        f"{l0_filename}: {l0_status}   pins: {len(pins)}   budget: {pct}% "
        f"({total_body}/{budget})"
    )

    # 报账警告紧贴 head, 不埋在 tree 之后
    if over_budget:
        lines.append("")
        lines.append(_render_budget_warning(pins, pin_blocks, budget))

    if tree_str:
        lines.append("")
        lines.append(f"tree(depth={convention.tree_depth}):")
        lines.append(tree_str)

    if hints:
        lines.append("")
        for h in hints:
            lines.append(f"⚖ {h} (instruction, unloaded)")

    for block in pin_blocks:
        lines.append("")
        lines.append(block)

    return "\n".join(lines)


# ---- tree --------------------------------------------------------------


def _render_tree(
    root: Path,
    depth: int,
    ignore_names: set[str],
    prefix: str = "",
) -> str:
    if depth <= 0:
        return ""
    try:
        entries = sorted(
            (e for e in root.iterdir() if e.name not in ignore_names),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
    except OSError:
        return ""

    lines: list[str] = []
    for i, entry in enumerate(entries):
        is_last = i == len(entries) - 1
        connector = "└── " if is_last else "├── "
        marker = "/" if entry.is_dir() else ""
        lines.append(f"{prefix}{connector}{entry.name}{marker}")
        if entry.is_dir() and depth > 1:
            sub_prefix = prefix + ("    " if is_last else "│   ")
            sub = _render_tree(entry, depth - 1, ignore_names, sub_prefix)
            if sub:
                lines.append(sub)
    return "\n".join(lines)


# ---- child hints -------------------------------------------------------


def _find_child_hints(root: Path, convention: GroundConvention) -> list[str]:
    """一层深度扫子目录里的 instruction 文件, 返回相对路径列表."""
    hints: list[str] = []
    try:
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            for name in convention.instruction_files:
                target = sub / name
                if target.is_file():
                    hints.append(str(target.relative_to(root)))
                    break  # 每子目录最多一个 hint
    except OSError:
        pass
    return hints


# ---- pin block ---------------------------------------------------------


def _render_pin(
    pin: Pin, parsed: ParsedAddr, obs: Observation, root: Path
) -> str:
    """一枚 pin 的渲染块 — header + optional note + content."""
    lines: list[str] = []

    header = f"── pin: {pin.addr}"
    if not obs.exists:
        header += "   [missing]"
    elif pin.seen_hash is not None and obs.hash != pin.seen_hash:
        header += "   [changed on disk]"
    lines.append(header)

    if pin.note:
        lines.append(f"   note: {pin.note}")

    if not obs.exists:
        lines.append("   (target does not exist)")
    elif parsed.kind == "glob":
        lines.append(_render_glob_hits(parsed, root))
    elif parsed.kind == "file":
        lines.append(_render_file_content(parsed, root))
    else:
        lines.append(_render_range_content(parsed, root))

    return "\n".join(lines)


def _render_glob_hits(parsed: ParsedAddr, root: Path) -> str:
    matches = resolve_glob_addr(parsed, root)
    if not matches:
        return "   (no matches)"
    lines: list[str] = []
    for m in matches:
        try:
            stat = m.stat()
            rel = m.relative_to(root)
            lines.append(
                f"   {rel}  ({stat.st_size}B, mtime={stat.st_mtime:.0f})"
            )
        except OSError:
            continue
    return "\n".join(lines) if lines else "   (all matches vanished)"


def _render_file_content(parsed: ParsedAddr, root: Path) -> str:
    try:
        target = resolve_file_addr(parsed, root)
        content = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "   (read failed)"
    file_lines = content.splitlines()
    return "\n".join(f"   {i+1}: {ln}" for i, ln in enumerate(file_lines))


def _render_range_content(parsed: ParsedAddr, root: Path) -> str:
    assert parsed.start is not None and parsed.end is not None
    try:
        target = resolve_file_addr(parsed, root)
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "   (read failed)"
    file_lines = text.splitlines()
    if parsed.start > len(file_lines):
        return "   (range beyond file end)"
    end = min(parsed.end, len(file_lines))
    return "\n".join(
        f"   {i+1}: {file_lines[i]}" for i in range(parsed.start - 1, end)
    )


# ---- budget report ------------------------------------------------------


def _render_budget_warning(
    pins: list[Pin], blocks: list[str], budget: int
) -> str:
    """报账行. 点名最大 3 张 pin, 让模型决定撤谁."""
    total = sum(len(b) for b in blocks)
    sized = sorted(zip(pins, blocks), key=lambda pb: -len(pb[1]))[:3]
    biggest = ", ".join(f"{p.addr} ({len(b)}B)" for p, b in sized)
    return (
        f"⚠ context over budget: {total} > {budget}  "
        f"top pins: {biggest}"
    )
