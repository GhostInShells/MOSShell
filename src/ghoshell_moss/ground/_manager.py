"""GroundsManager — one ghost governing several homes at once.

D1 of `ghost-ground-governance`: plurality lives *here*, not in `GroundSet`.
A `GroundSet` is "one root and the tree it grows"; a ghost lives in three homes
(ghost / mode / project), and three trees is what it actually has. So the manager
holds N single-rooted sets and answers the questions a channel asks of them: which
slots exist, which of them actually have a ground, what to do when one does not.

Two rules this module exists to keep:

- **Existence is the switch.** A slot whose home has no `GROUND.md` is not mounted
  and says so (`reason`). It is never inherited from an ancestor — that inheritance
  is what `collect_chain` does for directories *inside* a ground, and it is exactly
  wrong for a home, whose neighbours are other homes.
- **Addresses come from the caller.** A home's absolute path is a fact of the launch
  (the matrix knows it), never something a `GROUND.md` derives with `../..`. The
  manager takes paths; it does not discover them.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ghoshell_moss.ground.contract import GroundSet

__all__ = ["GroundSlot", "GroundsManager"]

L0_FILENAME = "GROUND.md"


@dataclass
class GroundSlot:
    """One home a ghost may govern. ``groundset`` is None when it is not mounted.

    ``reason`` is what a reader needs in order to tell "there is no ground here"
    (absent) from "there is one and it failed to load" (broken) — the two must never
    look alike in a notice.
    """

    key: str
    home: Path
    groundset: GroundSet | None = None
    reason: str = ""

    @property
    def mounted(self) -> bool:
        return self.groundset is not None

    @property
    def root(self):
        return self.groundset.root if self.groundset is not None else None

    def status_line(self) -> str:
        if self.mounted:
            return f"{self.key}: ok"
        return f"{self.key}: {self.reason or 'absent'}"


class GroundsManager:
    """Ordered slots; the first mounted one is the primary.

    Order is the caller's (ghost -> mode -> project) and means precedence when two
    layers speak about the same thing. It is *not* a hierarchy: the three homes are
    peers, and nothing here nests one inside another.
    """

    def __init__(
        self,
        slots: list[GroundSlot],
        *,
        logger: logging.Logger | None = None,
    ) -> None:
        self._slots = list(slots)
        self._logger = logger or logging.getLogger("moss")

    # -- lifecycle ---------------------------------------------------------

    async def mount(
        self,
        groundset_factory: Callable[["GroundSlot"], "GroundSet"],
    ) -> "GroundsManager":
        """Make every slot that can be mounted hold a `GroundSet`.

        **This method orchestrates; it does not create.** Who owns a set builds it —
        the channel factory hands in `groundset_factory`, and a slot that already
        carries a set keeps it untouched. That keeps one creation path: earlier both
        `mount()` and the channel factory built sets, so two sets could describe the
        same home and only one of them would be the one being read.

        The factory receives the whole slot rather than just its path: the slot key is
        the home's identity, and the set it builds should carry that identity instead
        of deriving a second name from the directory basename.

        Sync IO (a GROUND.md parse plus a `.grounds/` template scan per set) runs off
        the event loop; a slot that fails keeps its reason instead of taking the
        whole manager down — one bad home must not blind the other two.
        """
        for slot in self._slots:
            if slot.groundset is not None:
                slot.reason = ""
                continue
            if not slot.home.is_dir():
                slot.reason = "absent (no such directory)"
                continue
            if not (slot.home / L0_FILENAME).is_file():
                slot.reason = f"absent (no {L0_FILENAME})"
                continue
            try:
                slot.groundset = await asyncio.to_thread(
                    groundset_factory, slot
                )
            except Exception as exc:  # noqa: BLE001 — reported, never hidden
                slot.reason = f"broken ({exc.__class__.__name__}: {exc})"
                self._logger.warning("[grounds] %s failed to mount: %r", slot.key, exc)
                continue
            slot.reason = ""
        return self

    async def close(self) -> None:
        for slot in self._slots:
            if slot.groundset is None:
                continue
            try:
                await slot.groundset.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001 — one bad sediment must not block the rest
                self._logger.exception("[grounds] closing %s failed", slot.key)
            slot.groundset = None

    # -- queries -----------------------------------------------------------

    def slots(self) -> list[GroundSlot]:
        return list(self._slots)

    def mounted(self) -> list[GroundSlot]:
        return [s for s in self._slots if s.mounted]

    def absent(self) -> list[GroundSlot]:
        return [s for s in self._slots if not s.mounted]

    @property
    def primary(self) -> GroundSlot | None:
        """The first mounted slot — where an unqualified path or open() lands."""
        return next(iter(self.mounted()), None)

    def get(self, key: str) -> GroundSlot | None:
        return next((s for s in self._slots if s.key == key), None)

    def require(self, key: str) -> GroundSlot:
        slot = self.get(key)
        if slot is None:
            known = ", ".join(s.key for s in self._slots) or "(none)"
            raise KeyError(f"no ground slot {key!r} — known: {known}")
        return slot

    def status_lines(self) -> list[str]:
        """One line per slot, for the channel's own notice — absence is content."""
        return [s.status_line() for s in self._slots]
