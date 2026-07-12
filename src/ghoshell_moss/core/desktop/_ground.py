"""DefaultGround — Ground ABC 的进程内实现.

一个 Ground 实例 = 一个已打开的场. 由 DefaultGrounds.open() 构造, 生命
周期由 Grounds 治理 (async with).

内部状态:
- _pins: OrderedDict[addr, Pin] — 最新 pin/update 在前, 契合 pins() 顺序契约
- _instruction_cache: str — 首次 load 计算, 之后 refresh_instruction 显式刷新
- _convention: 从 L0 frontmatter 或构造参数固定, 之后不可变

sync/async 边界:
- pin / unpin / pins / instruction: sync (ABC 定义). pin 同步执行 observe_sync
  是有意的 (K17 initial 承认).
- update / refresh_instruction / context / load / sediment: async, 内部 IO 走
  to_thread + gather (符合 ABC async 契约).
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from pathlib import Path

from ghoshell_moss.contracts.desktop import (
    Ground,
    GroundConvention,
    Pin,
    UpdateResult,
)
from ghoshell_moss.core.desktop._addr import (
    ParsedAddr,
    parse_addr,
    resolve_file_addr,
)
from ghoshell_moss.core.desktop._hash import Observation, observe, observe_sync
from ghoshell_moss.core.desktop._instruction import collect_instructions
from ghoshell_moss.core.desktop._l0 import dump_l0_pins, load_l0
from ghoshell_moss.core.desktop._render import render_context

__all__ = ["DefaultGround"]


class DefaultGround(Ground):

    def __init__(
        self,
        label: str,
        root: Path,
        convention: GroundConvention,
        *,
        workspace_root: Path | None = None,
    ) -> None:
        self._label = label
        self._root = root.resolve()
        self._convention = convention
        self._workspace_root = workspace_root
        self._pins: OrderedDict[str, Pin] = OrderedDict()
        self._instruction_cache: str = ""

    # ---- 元信息 ---------------------------------------------------------

    @property
    def label(self) -> str:
        return self._label

    @property
    def root(self) -> Path:
        return self._root

    @property
    def convention(self) -> GroundConvention:
        return self._convention

    # ---- pin 管理 -------------------------------------------------------

    def pins(self) -> list[Pin]:
        return list(self._pins.values())

    def pin(self, addr: str, note: str = "") -> Pin:
        parsed = parse_addr(addr)
        if parsed.kind != "glob":
            resolve_file_addr(parsed, self._root)  # boundary check
        obs = observe_sync(parsed, self._root)
        existing = self._pins.get(addr)
        # note 处理: 显式传入优先, 否则继承 (幂等重 pin 时不丢注记)
        final_note = note if note else (existing.note if existing else "")
        new_pin = Pin(
            addr=addr,
            note=final_note,
            pinned_at=time.time(),
            seen_mtime=obs.mtime,
            seen_hash=obs.hash,
        )
        self._insert_pin_at_front(new_pin)
        return new_pin

    def unpin(self, addr: str) -> None:
        del self._pins[addr]  # KeyError 契约

    async def update(self, addr: str) -> UpdateResult:
        if addr not in self._pins:
            raise KeyError(addr)
        old = self._pins[addr]
        parsed = parse_addr(addr)
        if parsed.kind != "glob":
            resolve_file_addr(parsed, self._root)
        obs = await observe(parsed, self._root)
        changed = old.seen_hash != obs.hash
        new_pin = Pin(
            addr=old.addr,
            note=old.note,
            pinned_at=old.pinned_at,
            seen_mtime=obs.mtime,
            seen_hash=obs.hash,
        )
        self._insert_pin_at_front(new_pin)
        return UpdateResult(
            addr=addr,
            changed=changed,
            old_mtime=old.seen_mtime,
            new_mtime=obs.mtime,
            diff_preview=_diff_preview(old, obs, parsed),
        )

    def _insert_pin_at_front(self, pin: Pin) -> None:
        """有则替换, 无则插入; 结果都 move-to-front (最新在前)."""
        addr = pin.addr
        if addr in self._pins:
            del self._pins[addr]
        self._pins[addr] = pin
        self._pins.move_to_end(addr, last=False)

    # ---- 渲染 -----------------------------------------------------------

    def instruction(self) -> str:
        return self._instruction_cache

    async def refresh_instruction(self) -> None:
        self._instruction_cache = await asyncio.to_thread(
            collect_instructions,
            self._root,
            self._convention,
            workspace_root=self._workspace_root,
        )

    async def context(self) -> str:
        return await render_context(
            self._root, self._label, self._convention, self.pins()
        )

    # ---- 生命周期 -------------------------------------------------------

    async def load(self) -> None:
        contents = await asyncio.to_thread(load_l0, self._root)
        # convention 已在构造时确定; 这里只装载 pins + instruction 缓存
        self._pins = OrderedDict((p.addr, p) for p in contents.pins)
        await self.refresh_instruction()

    async def sediment(self) -> None:
        await asyncio.to_thread(dump_l0_pins, self._root, self.pins())


def _diff_preview(old_pin: Pin, obs: Observation, parsed: ParsedAddr) -> str:
    """bounded diff 摘要 — UpdateResult.diff_preview 载体, 避免历史洪泛."""
    if old_pin.seen_hash == obs.hash:
        return "no change"
    if old_pin.seen_hash is None:
        return "initial observation"
    if not obs.exists:
        return "target removed"
    if parsed.kind == "glob":
        return "glob hit set changed"
    if parsed.kind == "range":
        return f"range {parsed.start}-{parsed.end} content changed"
    return "file content changed"
