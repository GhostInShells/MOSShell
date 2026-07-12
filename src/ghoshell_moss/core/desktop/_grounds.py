"""DefaultGrounds — Grounds ABC 的进程内实现. per-owner.

一个 Grounds 实例 = 一个 owner (session / process / Ghost) 的桌子. 通过
`async with` 生命周期管其下所有 Ground; owner __aexit__ 时全部 sediment
落盘, 与 subprocesses/job_supervisor 同姿态.

CTML 接触面 (K14/K21): 模型只调 Grounds 上的动词 (open/close/pin/unpin/
update/frame), 参数带 label. Grounds 内部转发到 opened[label].方法.

幂等 open (K21 决定):
- 同目录 (root abspath 相同) 再次 open → 返回已 active 的 Ground.
- 传入的 label / convention 忽略 (以已 active 的为准).
- 目录是认知单元, 不是工具 — 一个 dir 一份 pin 集与法链.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from ghoshell_moss.contracts.desktop import (
    Ground,
    GroundConvention,
    Grounds,
    Pin,
    UpdateResult,
)
from ghoshell_moss.core.desktop._ground import DefaultGround
from ghoshell_moss.core.desktop._l0 import load_l0

__all__ = ["DefaultGrounds"]


class DefaultGrounds(Grounds):

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
    ) -> None:
        self._workspace_root: Path = (
            workspace_root.resolve() if workspace_root else Path.cwd().resolve()
        )
        self._active: dict[str, Ground] = {}
        # abspath (str) → label — 幂等 open 的正查表
        self._label_by_path: dict[str, str] = {}

    async def open(
        self,
        dir: str | Path,
        *,
        label: str | None = None,
        convention: GroundConvention | None = None,
    ) -> Ground:
        dir_path = Path(dir)
        if not dir_path.is_absolute():
            dir_path = self._workspace_root / dir_path
        dir_abs = dir_path.resolve()

        # 幂等: 同目录已 active 就返回原实例
        key = str(dir_abs)
        if key in self._label_by_path:
            return self._active[self._label_by_path[key]]

        # convention 缺省时从 L0 加载
        if convention is None:
            contents = await asyncio.to_thread(load_l0, dir_abs)
            convention = contents.convention

        # label 分配: 缺省 = basename; 冲突加 -2 / -3 后缀
        base = label if label else dir_abs.name
        final_label = base
        suffix = 2
        while final_label in self._active:
            final_label = f"{base}-{suffix}"
            suffix += 1

        ground = DefaultGround(
            label=final_label,
            root=dir_abs,
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
        # 从 path 映射里清掉
        for path_key, mapped in list(self._label_by_path.items()):
            if mapped == label:
                del self._label_by_path[path_key]
                break

    # ---- 查询 -----------------------------------------------------------

    def active(self) -> dict[str, Ground]:
        return dict(self._active)

    def get(self, label: str) -> Ground | None:
        return self._active.get(label)

    # ---- 转发 (CTML 接触面) ---------------------------------------------

    def pin(self, label: str, addr: str, note: str = "") -> Pin:
        return self._require(label).pin(addr, note)

    def unpin(self, label: str, addr: str) -> None:
        self._require(label).unpin(addr)

    async def update(self, label: str, addr: str) -> UpdateResult:
        return await self._require(label).update(addr)

    async def frame(self, label: str) -> str:
        return await self._require(label).context()

    def _require(self, label: str) -> Ground:
        if label not in self._active:
            raise KeyError(label)
        return self._active[label]

    # ---- 生命周期 -------------------------------------------------------

    async def __aenter__(self) -> "DefaultGrounds":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        # sediment 全部, 尽力而为 (退出路径不 raise)
        for label in list(self._active.keys()):
            try:
                await self.close(label)
            except Exception:
                # 单个 close 失败不影响其它 close; 与 subprocesses.__aexit__
                # 的 best-effort 姿态一致.
                pass
