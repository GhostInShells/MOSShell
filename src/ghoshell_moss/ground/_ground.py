"""DefaultGround — Ground ABC 的进程内实现.

一个 Ground 实例 = 一个已打开的场. 由 DefaultGroundSet.open() 构造,
生命周期由 GroundSet 治理.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from pathlib import Path

from pathspec import PathSpec

from ghoshell_moss.ground._addr import Anchor
from ghoshell_moss.ground._chain import collect_chain
from ghoshell_moss.ground._l0 import DEFAULT_L0_FILENAME, dump_l0_pins, load_l0
from ghoshell_moss.ground._render import render_context, render_items
from ghoshell_moss.ground.contract import (
    FrameItem,
    Ground,
    GroundConvention,
    GlobPin,
    Pin,
)

__all__ = ["DefaultGround"]


class DefaultGround(Ground):
    """Ground ABC 的默认实现.

    Internal state:
    - _pins: OrderedDict[label, Pin] — 最新 pin 在前
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
        self._body: str = ""
        self._dirty: bool = False
        self._ignore_spec: PathSpec | None = self._make_ignore_spec()

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
        return pin

    def unpin(self, label: str) -> None:
        del self._pins[label]
        self._dirty = True

    # -- 渲染 -------------------------------------------------------------

    async def context(self) -> str:
        items = await self.frame_items()
        return render_items(items, ground_path=str(self._root))

    async def frame_items(self) -> list[FrameItem]:
        """返回当前帧的 FrameItem 列表 — 供 --json / 定制渲染消费."""
        return await render_context(
            body=self._body,
            pins=list(self._pins.values()),
            anchor=self._make_anchor(),
            ignore=self._ignore_spec,
        )

    @property
    def ignore_spec(self) -> PathSpec | None:
        """场级 ignore 规则 — 从 convention 的 ignore + ignore_file 合并."""
        return self._ignore_spec

    async def chain_text(self) -> str:
        """返回法链 body (供 meta / instruction 使用)."""
        return await asyncio.to_thread(collect_chain, self._doc_path.parent)

    # -- 生命周期 ---------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self._dirty

    async def load(self) -> None:
        contents = await asyncio.to_thread(
            load_l0, self._doc_path.parent, self._doc_path.name
        )
        self._body = contents.body
        self._convention = contents.convention
        self._pins = OrderedDict(
            (p.label, p) for p in contents.pins
        )
        self._ignore_spec = self._make_ignore_spec()
        self._dirty = False

    async def sediment(self) -> None:
        # 写回法锚 doc_path — 场内移动 (doc≠root) 时沉积回场根,
        # 永不在工作场子目录创建 GROUND.md
        await asyncio.to_thread(
            dump_l0_pins,
            self._doc_path.parent,
            list(self._pins.values()),
            self._doc_path.name,
            body=self._body if not self._doc_path.is_file() else None,
        )
        self._dirty = False

    # -- internal ---------------------------------------------------------

    def _make_anchor(self) -> Anchor:
        return Anchor(
            ground=self._doc_path.parent.resolve(),
            cwd=self._root,
        )

    def _make_ignore_spec(self) -> PathSpec | None:
        """Build a merged PathSpec from convention ignore + ignore_file.

        Inline ``ignore`` list and file content are merged — both use
        .gitignore syntax.  Returns None if neither is configured.
        """
        patterns: list[str] = []

        if self._convention.ignore:
            patterns.extend(self._convention.ignore)

        if self._convention.ignore_file:
            ignore_path = self._root / self._convention.ignore_file
            if ignore_path.is_file():
                try:
                    file_patterns = ignore_path.read_text(
                        encoding="utf-8", errors="replace",
                    ).splitlines()
                    # Strip comments and blanks, but keep negation (!) lines
                    file_patterns = [
                        ln for ln in file_patterns
                        if ln.strip() and not ln.strip().startswith("#")
                    ]
                    patterns.extend(file_patterns)
                except OSError:
                    pass

        if not patterns:
            return None
        return PathSpec.from_lines("gitignore", patterns)
