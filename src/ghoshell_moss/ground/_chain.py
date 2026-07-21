"""Law chain — 祖先 GROUND.md body 收集 (K56 / SPEC §7.5).

法链 = 从 ground 的法锚点 (doc 所在目录) 向上到 $HOME, 收集每层 GROUND.md
的 body 内容. root-first (最远祖先在前, ground 自身 body 最后). 只收 body,
不携带 frontmatter / pins / @-expansion.

同步 IO: 调用方用 asyncio.to_thread 卸载.
"""

from __future__ import annotations

import os
from pathlib import Path

from ghoshell_moss.ground._l0 import DEFAULT_L0_FILENAME, load_l0

__all__ = ["collect_chain"]


def collect_chain(
    law_anchor: Path,
    *,
    boundary: Path | None = None,
) -> str:
    """从 law_anchor 向上收集祖先 GROUND.md body, root-first.

    - law_anchor: doc 所在目录 (法锚点). 从此向上走.
    - boundary: 默认 = $HOME. None = 走到文件系统根.
    - 返回: 拼接 body, 含来源标注. 空返回 "".
    """
    resolved_boundary = boundary.resolve() if boundary else _default_boundary()
    anchor = law_anchor.resolve()

    # 收集目录列表: 从 anchor 向上到 boundary (含)
    dirs = _walk_upward(anchor, resolved_boundary)
    dirs.reverse()  # root-first

    blocks: list[str] = []
    for d in dirs:
        contents = load_l0(d)
        body = contents.body.strip()
        if body:
            blocks.append(f"<!-- from: {d / DEFAULT_L0_FILENAME} -->\n\n{body}")

    return "\n\n".join(blocks)


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
