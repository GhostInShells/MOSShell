"""DefaultGroundSet — 一棵场树: 一个根 + 一个边界, 按位置渲染.

不持有「打开的 Ground」: 渲染是冷读 (SPEC §7.1), 每次从盘上解析最近的那个
GROUND.md. 老的 open / close / sediment / dirty / snapshot 生命周期随 pin 的
编辑面一起退场 —— 编辑归文件工具, 渲染归位置, 两者不共享内存状态.
"""

from __future__ import annotations

from pathlib import Path

from ghoshell_moss.ground._render import (
    RenderLimits,
    find_ground,
    render_position,
)
from ghoshell_moss.ground.contract import GroundSet

__all__ = ["DefaultGroundSet"]


class DefaultGroundSet(GroundSet):
    """根 + 边界. 没有活跃场表, 没有 label 空间, 没有退出时的落盘."""

    def __init__(
        self,
        *,
        root: str | Path,
        boundary: str | Path | None = None,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._boundary = (
            Path(boundary).expanduser().resolve() if boundary is not None else self._root
        )

    @property
    def root(self) -> Path:
        return self._root

    @property
    def boundary(self) -> Path:
        return self._boundary

    def ground_at(self, position: str | Path | None = None) -> Path | None:
        """位置 → 它所属的场根. 边界之上永不咨询 (SPEC §6.1)."""
        target = Path(position).expanduser() if position is not None else self._root
        return find_ground(target, self._boundary)

    async def render(
        self,
        position: str | Path | None = None,
        *,
        template: str | None = None,
        limits: RenderLimits | None = None,
    ) -> str:
        """渲染一个位置 (None = 场根) → 自解释 XML. 失败是结果, 不是异常."""
        target = Path(position).expanduser() if position is not None else self._root
        return await render_position(
            target,
            boundary=self._boundary,
            limits=limits,
            template=template,
        )
