"""Law chain — 向上找最近的 GROUND.md 返回其 body (单层, 不合并).

场的法 = 场自身的 body。无 GROUND.md 的目录向上找最近的场 (walk 定位),
但场与场之间不合并 body——每个子场渲染自己的根, 向上合并会造成大量重复。
只收 body, 不携带 frontmatter / pins / @-expansion.

同步 IO: 调用方用 asyncio.to_thread 卸载.
"""

from __future__ import annotations

import os
from pathlib import Path

from ghoshell_moss.ground._addr import Anchor
from ghoshell_moss.ground._l0 import load_l0

__all__ = ["collect_chain", "collect_law_files"]


def collect_chain(
    law_anchor: Path,
    *,
    boundary: Path | None = None,
) -> str:
    """从 law_anchor 向上找最近的 GROUND.md, 返回其 body (单层, 不合并).

    - law_anchor: doc 所在目录 (法锚点). 从此向上走.
    - boundary: 默认 = $HOME. None = 走到文件系统根.
    - 返回: 最近一个 GROUND.md 的 body. 空返回 "".
    """
    resolved_boundary = boundary.resolve() if boundary else _default_boundary()
    anchor = law_anchor.resolve()

    # 从 anchor 向上 (含 anchor) 找最近的非空 body, 单层返回.
    for d in _walk_upward(anchor, resolved_boundary):
        contents = load_l0(d)
        body = contents.body.strip()
        if body:
            return body

    return ""


def collect_law_files(anchor: Anchor, filename: str) -> list[Path]:
    """law pin 的收集逻辑 — 从 cwd 向上到 ground root, 收集存在的 filename 文件.

    - 边界 = ground root (不越出场, SPEC §8 subtree confinement).
    - cwd 在场根外 → 只收 cwd 一个 (与 _walk_upward 同语义).
    - 返回 root-first 顺序 (父级向下展示), 供渲染直接消费.
    """
    cwd = anchor.cwd.resolve()
    ground = anchor.ground.resolve()
    dirs = _walk_upward(cwd, ground)
    dirs.reverse()  # root-first
    return [d / filename for d in dirs if (d / filename).is_file()]


def _default_boundary() -> Path:
    home = os.environ.get("HOME")
    if home:
        return Path(home).resolve()
    return Path("/").resolve()


def _walk_upward(start: Path, boundary: Path) -> list[Path]:
    """从 start 向上走到 boundary (含), 返回目录列表 (从近到远).

    start 在 boundary 外 → 只返回 [start].
    """
    try:
        start.relative_to(boundary)
    except ValueError:
        return [start]

    dirs: list[Path] = []
    current = start
    while True:
        dirs.append(current)
        if current == boundary:
            break
        if current == current.parent:
            break
        current = current.parent
    return dirs
