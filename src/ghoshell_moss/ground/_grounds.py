"""DefaultGroundSet — GroundSet ABC 的进程内实现.

每个 GroundSet 实例有独立 label 空间 — 多实例, 非单例.
不同 channel 各自创建自己的 GroundSet, label 冲突天然隔离.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ghoshell_moss.ground._ground import DefaultGround
from ghoshell_moss.ground._l0 import load_l0
from ghoshell_moss.ground.contract import Ground, GroundSet

__all__ = ["DefaultGroundSet"]


class DefaultGroundSet(GroundSet):
    """GroundSet ABC 的默认实现.

    - workspace_root: 相对路径解析基点.
    - _active: label → Ground 映射.
    - _label_by_path: abspath → label, 幂等 open 的快速查表.
    """

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
    ) -> None:
        self._workspace_root = (
            workspace_root.resolve() if workspace_root else Path.cwd().resolve()
        )
        self._active: dict[str, Ground] = {}
        self._label_by_path: dict[str, str] = {}

    # -- open/close -------------------------------------------------------

    async def open(
        self,
        dir: str | Path,
        *,
        label: str | None = None,
        doc: str | Path | None = None,
    ) -> Ground:
        dir_path = Path(dir)
        if not dir_path.is_absolute():
            dir_path = self._workspace_root / dir_path
        dir_abs = dir_path.resolve()

        # 幂等
        key = str(dir_abs)
        if key in self._label_by_path:
            return self._active[self._label_by_path[key]]

        # doc 路径
        doc_path = Path(doc).resolve() if doc else dir_abs / "GROUND.md"

        # 从 GROUND.md 加载 convention
        contents = await asyncio.to_thread(load_l0, dir_abs)
        convention = contents.convention

        # label 分配
        base = label if label else dir_abs.name
        final_label = base
        suffix = 2
        while final_label in self._active:
            final_label = f"{base}-{suffix}"
            suffix += 1

        ground = DefaultGround(
            label=final_label,
            root=dir_abs,
            doc_path=doc_path,
            convention=convention,
            workspace_root=self._workspace_root,
        )
        await ground.load()
        self._active[final_label] = ground
        self._label_by_path[key] = final_label
        return ground

    async def close(self, label: str) -> None:
        if label not in self._active:
            raise KeyError(label)
        ground = self._active[label]
        await ground.sediment()
        del self._active[label]
        for path_key, mapped in list(self._label_by_path.items()):
            if mapped == label:
                del self._label_by_path[path_key]
                break

    # -- 查询 -------------------------------------------------------------

    def active(self) -> dict[str, Ground]:
        return dict(self._active)

    def get(self, label: str) -> Ground | None:
        return self._active.get(label)

    # -- 生命周期 ---------------------------------------------------------

    async def __aenter__(self) -> "DefaultGroundSet":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        for label in list(self._active.keys()):
            try:
                await self.close(label)
            except Exception:
                pass
