"""ScreenModel — the single source of truth for the screen body's window state.

One store, two faces: the channel drives it from the model side, the web surface from
the human side, and neither talks to the other directly.

Items live in a materialization pool, each in exactly one group. A group is an ordered
list of item ids; the active group lays its items out on the stage. The store owns the
invariants — an item is in at most one group, and an empty group vanishes — so neither
face has to restate them.

The layout shape is a pure function of ``(n, family, dir)``. The model never says *how
big* anything is; it gives an ordered id list and the family, and the shape falls out.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

__all__ = [
    "Item",
    "ScreenModel",
    "Layout",
    "Cell",
    "LayoutError",
    "compute_layout",
    "grid_template",
]

FAMILIES = ("grid", "stack")
DIRS = ("lr", "tb")


class LayoutError(ValueError):
    """A proposed arrangement cannot be laid out (bad family/dir, wrong item count)."""


@dataclass
class Item:
    """A windowable thing in the materialization pool."""

    id: str
    url: str
    label: str = ""
    group: str = ""


@dataclass(frozen=True)
class Cell:
    """One slot's grid coordinates (1-based, CSS grid line numbers)."""

    r: int
    c: int
    rs: int
    cs: int


@dataclass(frozen=True)
class Layout:
    """The resolved shape for one arrangement."""

    family: str
    dir: str
    cols: int
    rows: int
    cells: list[Cell]


def compute_layout(n: int, family: str, dir: str) -> Layout:
    """Resolve ``(n, family, dir)`` into grid coordinates.

    ``grid`` splits equally; ``stack`` gives one master (index 0) plus a strip of the
    rest. ``dir`` is the axis: ``lr`` fills a row first, ``tb`` a column first.
    """
    if family not in FAMILIES:
        raise LayoutError(f"unknown family {family!r} (grid | stack)")
    if dir not in DIRS:
        raise LayoutError(f"unknown dir {dir!r} (lr | tb)")
    if n <= 0:
        raise LayoutError("need at least one item to arrange")

    if family == "stack":
        if n < 2:
            raise LayoutError("stack needs at least 2 items (one master + a strip)")
        m = n - 1
        if dir == "lr":
            cells = [Cell(1, 1, m, 1)]
            cells += [Cell(1 + k, 2, 1, 1) for k in range(m)]
            return Layout("stack", "lr", 2, m, cells)
        cells = [Cell(1, 1, 1, m)]
        cells += [Cell(2, 1 + k, 1, 1) for k in range(m)]
        return Layout("stack", "tb", m, 2, cells)

    # grid — cols = ceil(sqrt(n)) for n >= 4; n in {2, 3} lay out in one band along dir.
    if n == 1:
        cols, rows = 1, 1
    elif n == 2:
        cols, rows = (2, 1) if dir == "lr" else (1, 2)
    elif n == 3:
        cols, rows = (3, 1) if dir == "lr" else (1, 3)
    else:
        cols = ceil(n ** 0.5)
        rows = ceil(n / cols)
    cells = [Cell(k // cols + 1, k % cols + 1, 1, 1) for k in range(n)]
    return Layout("grid", dir, cols, rows, cells)


def grid_template(layout: Layout) -> tuple[str, str]:
    """The CSS ``grid-template-columns`` / ``grid-template-rows`` for a layout.

    Stack gives the master axis a 2:1 share; everything else is even.
    """
    if layout.family == "stack":
        if layout.dir == "lr":
            return ("2fr 1fr", f"repeat({layout.rows}, 1fr)")
        return (f"repeat({layout.cols}, 1fr)", "2fr 1fr")
    return (f"repeat({layout.cols}, 1fr)", f"repeat({layout.rows}, 1fr)")


class ScreenModel:
    """Window state: the pool, the groups, the active group, fullscreen, layout."""

    def __init__(self) -> None:
        self._items: dict[str, Item] = {}
        self._groups: dict[str, list[str]] = {}
        self._active: str = ""
        self._fullscreen: str | None = None
        self._family = "grid"
        self._dir = "lr"

    # -- queries -----------------------------------------------------------

    def items(self) -> list[Item]:
        return list(self._items.values())

    def get(self, item_id: str) -> Item | None:
        return self._items.get(item_id)

    def groups(self) -> list[str]:
        """Group names in creation order."""
        return list(self._groups.keys())

    def group_items(self, group: str) -> list[str]:
        """The ordered id list of a group (empty if the group does not exist)."""
        return list(self._groups.get(group, []))

    def active(self) -> str:
        return self._active

    def active_items(self) -> list[str]:
        return self.group_items(self._active)

    def fullscreen(self) -> str | None:
        return self._fullscreen

    def family(self) -> str:
        return self._family

    def dir(self) -> str:
        return self._dir

    def layout(self) -> Layout:
        """The active group's resolved layout, from the declared family/dir and its size."""
        return compute_layout(len(self.active_items()), self._family, self._dir)

    def group_of(self, item_id: str) -> str:
        item = self._items.get(item_id)
        return item.group if item is not None else ""

    # -- mutations ---------------------------------------------------------

    def open(self, item_id: str, url: str, *, label: str = "", group: str = "") -> Item:
        """Materialize a window into a group (default: the active group).

        An item lives in exactly one group. Opening into a fresh group creates it.
        """
        if not item_id:
            raise ValueError("item id is required")
        if item_id in self._items:
            raise ValueError(f"item {item_id!r} already exists")
        if not url:
            raise ValueError("url is required")
        target = group or self._active
        if not target:
            raise ValueError(
                f"no active group and none given — activate() a group first"
            )
        item = Item(id=item_id, url=url, label=label or item_id, group=target)
        self._items[item_id] = item
        self._groups.setdefault(target, []).append(item_id)
        return item

    def close(self, item_id: str) -> Item:
        """Remove a window. An emptied group is deleted automatically."""
        item = self._require(item_id)
        del self._items[item_id]
        group = self._groups[item.group]
        group.remove(item_id)
        if not group:
            del self._groups[item.group]
            if self._active == item.group:
                self._active = ""
        if self._fullscreen == item_id:
            self._fullscreen = None
        return item

    def activate(self, group: str) -> None:
        """Switch the active group (and clear a stale fullscreen)."""
        if group not in self._groups:
            raise ValueError(f"no group {group!r} — groups(): {self.groups()}")
        self._active = group
        self._fullscreen = None

    def arrange(self, ids: list[str], *, family: str, dir: str) -> Layout:
        """Set the active group's order and layout.

        ``ids`` must be exactly the active group's items, in the order the model wants.
        """
        if not self._active:
            raise ValueError("no active group — activate() first")
        current = self._groups[self._active]
        if sorted(ids) != sorted(current):
            raise ValueError(
                f"ids must be exactly the active group's items "
                f"({', '.join(current)}) — got {', '.join(ids)}"
            )
        layout = compute_layout(len(ids), family, dir)
        self._groups[self._active] = list(ids)
        self._family = family
        self._dir = dir
        return layout

    def set_fullscreen(self, item_id: str | None) -> None:
        if item_id is not None:
            self._require(item_id)
        self._fullscreen = item_id

    def _require(self, item_id: str) -> Item:
        item = self._items.get(item_id)
        if item is None:
            raise KeyError(f"no item {item_id!r} — items(): {[i.id for i in self.items()]}")
        return item
