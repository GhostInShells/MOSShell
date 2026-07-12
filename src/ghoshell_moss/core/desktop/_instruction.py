"""Instruction chain — collect law files from ground root walking upward.

Ground.instruction() 消费本模块. 首次挂载 (Ground.load) 时一次性收集,
缓存到 Ground._instruction_cache; refresh_instruction() 重新调用刷新缓存.

行为 (契合 GroundConvention 字段语义):

- convention.upward_lookup=False: 只看 root 本层.
- convention.upward_lookup=True (default):
    从 root 向上逐层, 直到 upward_boundary (或 workspace_root 兜底,
    或 fs root). 每层收集 convention.instruction_files 命名命中的文件.

- 输出顺序: "根最先" — 最远的祖先层在前, root 本层在后. 契合 Claude Code
  的 CLAUDE.md 链语义 (outer scope 先, inner scope 覆盖).

- 场目录在 boundary 外 (罕见但合法): 只收 root 本层.

sync IO: 调用方 (Ground.load / refresh_instruction) 用 asyncio.to_thread
卸载. 测试可直接同步调用.
"""

from __future__ import annotations

from pathlib import Path

from ghoshell_moss.contracts.desktop import GroundConvention

__all__ = ["collect_instructions"]


def collect_instructions(
    root: Path,
    convention: GroundConvention,
    *,
    workspace_root: Path | None = None,
) -> str:
    """收集法链. 返回拼接后的字符串 (含每段来源标注). 空返回空串.

    - root: 场根目录.
    - workspace_root: 兜底 boundary, 当 convention.upward_boundary is None
      时用. None 表示 "没有兜底, 一路走到 fs root".
    """
    root_abs = root.resolve()
    boundary = _resolve_boundary(convention, workspace_root)

    if not convention.upward_lookup:
        dirs = [root_abs]
    else:
        dirs = _walk_upward(root_abs, boundary)

    # 根最先 (最远祖先在前)
    dirs.reverse()

    blocks: list[str] = []
    for d in dirs:
        block = _collect_from_dir(d, convention.instruction_files)
        if block:
            blocks.append(block)

    return "\n\n".join(blocks)


def _resolve_boundary(
    convention: GroundConvention,
    workspace_root: Path | None,
) -> Path | None:
    if convention.upward_boundary:
        return Path(convention.upward_boundary).resolve()
    if workspace_root is not None:
        return workspace_root.resolve()
    return None


def _walk_upward(root_abs: Path, boundary: Path | None) -> list[Path]:
    """从 root_abs 向上收集目录, 到 boundary (含) 或 fs root 为止.

    root_abs 在 boundary 外时只返回 [root_abs] (契约中 "场目录在边界外时
    只收本层" 的字面兑现).
    """
    # 越界检测
    if boundary is not None:
        try:
            root_abs.relative_to(boundary)
        except ValueError:
            return [root_abs]

    dirs: list[Path] = []
    current = root_abs
    while True:
        dirs.append(current)
        if boundary is not None and current == boundary:
            break
        if current == current.parent:
            break
        current = current.parent
    return dirs


def _collect_from_dir(dir_: Path, names: tuple[str, ...]) -> str:
    """从一个目录里收集命中 names 的 instruction 文件, 拼接为一段."""
    parts: list[str] = []
    for name in names:
        p = dir_ / name
        if p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parts.append(f"<!-- from: {p} -->\n\n{content}")
    return "\n\n".join(parts).rstrip() + "\n" if parts else ""
