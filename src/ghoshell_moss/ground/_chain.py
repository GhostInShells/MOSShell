"""Content-file collection — the `content` pin's upward walk.

从 $CWD 向上到 ground root 收集同名的外部内容文件 (CLAUDE.md / AGENT.md...),
root-first 返回. 边界是场根: 绝不越出 (SPEC §5.6).

同步 IO: 调用方用 asyncio.to_thread 卸载.
"""

from __future__ import annotations

from pathlib import Path

from ghoshell_moss.ground._addr import Anchor

__all__ = ["collect_content_files"]


def collect_content_files(anchor: Anchor, filename: str) -> list[Path]:
    """content pin 的收集逻辑 — 从 cwd 向上到 ground root, 收集存在的 filename 文件.

    - 边界 = ground root (不越出场, SPEC §8 subtree confinement).
    - cwd 在场根外 → 只收 cwd 一个 (与 _walk_upward 同语义).
    - 返回 root-first 顺序 (父级向下展示), 供渲染直接消费.
    """
    cwd = anchor.cwd.resolve()
    ground = anchor.ground.resolve()
    dirs = _walk_upward(cwd, ground)
    dirs.reverse()  # root-first
    return [d / filename for d in dirs if (d / filename).is_file()]



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
