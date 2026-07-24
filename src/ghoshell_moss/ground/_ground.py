"""DefaultGround — Ground ABC 的进程内实现.

一个 Ground 实例 = 一个已打开的场. 由 DefaultGroundSet.open() 构造,
生命周期由 GroundSet 治理.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from pathlib import Path

from ghoshell_moss.ground._addr import Anchor
from ghoshell_moss.ground._chain import collect_chain
from ghoshell_moss.ground._hash import PinShadow, observe_sync
from ghoshell_moss.ground._l0 import DEFAULT_L0_FILENAME, dump_l0_pins, load_l0
from ghoshell_moss.ground._render import render_context
from ghoshell_moss.ground.contract import (
    Ground,
    GroundConvention,
    GlobPin,
    Pin,
    UpdateResult,
)

__all__ = ["DefaultGround"]


class DefaultGround(Ground):
    """Ground ABC 的默认实现.

    Internal state:
    - _pins: OrderedDict[label, Pin] — 最新 pin/update 在前
    - _shadows: dict[label, PinShadow] — 运行时观察影子, 不进盘
    - _body: GROUND.md body, 每次 load 时更新
    """

    def __init__(
        self,
        label: str,
        root: Path,
        doc_path: Path,
        convention: GroundConvention,
        *,
        workspace_root: Path | None = None,
    ) -> None:
        self._label = label
        self._root = root.resolve()
        self._doc_path = doc_path.resolve()
        self._convention = convention
        self._workspace_root = workspace_root
        self._pins: OrderedDict[str, Pin] = OrderedDict()
        self._shadows: dict[str, PinShadow] = {}
        self._body: str = ""
        self._dirty: bool = False

    # -- 元信息 -----------------------------------------------------------

    @property
    def label(self) -> str:
        return self._label

    @property
    def root(self) -> Path:
        return self._root

    @property
    def doc_path(self) -> Path:
        return self._doc_path

    @property
    def convention(self) -> GroundConvention:
        return self._convention

    # -- pin 管理 ---------------------------------------------------------

    def pins(self) -> list[Pin]:
        return list(self._pins.values())

    def pin(self, pin: Pin) -> Pin:
        self._pins[pin.label] = pin
        self._pins.move_to_end(pin.label, last=False)
        self._dirty = True

        # 初始观察 — 建立 shadow 基线
        anchor = self._make_anchor()
        obs = observe_sync(pin, anchor)
        self._shadows[pin.label] = PinShadow(mtime=obs.mtime, hash=obs.hash)
        return pin

    def unpin(self, label: str) -> None:
        del self._pins[label]
        self._shadows.pop(label, None)
        self._dirty = True

    # -- 对账 -------------------------------------------------------------

    async def update(self, label: str) -> UpdateResult:
        pin = self._pins[label]  # KeyError if missing
        old_shadow = self._shadows.get(label, PinShadow())

        anchor = self._make_anchor()
        obs = await asyncio.to_thread(observe_sync, pin, anchor)

        changed = old_shadow.hash != obs.hash
        self._shadows[label] = PinShadow(mtime=obs.mtime, hash=obs.hash)
        self._pins.move_to_end(label, last=False)
        self._dirty = True

        return UpdateResult(
            label=label,
            changed=changed,
            old_hash=old_shadow.hash,
            new_hash=obs.hash,
            summary=_summary(pin, obs, changed),
        )

    # -- 渲染 -------------------------------------------------------------

    async def context(self) -> str:
        return await render_context(
            body=self._body,
            pins=list(self._pins.values()),
            shadows=dict(self._shadows),
            anchor=self._make_anchor(),
        )

    async def chain_text(self) -> str:
        """返回法链 body (供 meta / instruction 使用)."""
        return await asyncio.to_thread(collect_chain, self._doc_path.parent)

    # -- 生命周期 ---------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self._dirty

    async def load(self) -> None:
        contents = await asyncio.to_thread(load_l0, self._root)
        self._body = contents.body
        self._pins = OrderedDict(
            (p.label, p) for p in contents.pins
        )
        self._shadows.clear()
        self._dirty = False

    async def sediment(self) -> None:
        await asyncio.to_thread(
            dump_l0_pins, self._root, list(self._pins.values())
        )
        self._dirty = False

    # -- internal ---------------------------------------------------------

    def _make_anchor(self) -> Anchor:
        return Anchor(
            ground=self._doc_path.parent.resolve(),
            cwd=self._root,
        )


# -- helpers --------------------------------------------------------------


def _summary(pin: Pin, obs, changed: bool) -> str:
    if not changed:
        return "no change"
    if not obs.exists:
        return "target removed"
    if isinstance(pin, GlobPin):
        return "glob hit set changed"
    return "content changed"
