"""ScreenModel — the single source of truth for the screen body's window state.

One store, two faces: the channel drives it from the model side, the web surface
from the human side, and neither talks to the other directly.

Model
-----
There is one materialization pool. Every item lives in it. An item is
*additionally* in at most one group; an item in no group is **on the desktop** —
it is still there, still in the pool, just not arranged. The desktop is not a
group: it has no name, no layout, and cannot be arranged. It is the resting
state of the pool.

The screen shows exactly one thing at a time — one group, or the desktop. That
is ``arena()``. Switching the arena never moves an item between groups; it
changes which arrangement is presented.

Groups own their own layout. ``arrange`` sets one group's order and its
family/dir; switching away and back returns to what that group was told, in the
shape it was told. Fullscreen is per-group for the same reason: a group remembers
whether it was filling the stage.

The layout shape is a pure function of ``(n, family, dir)``. The model never says
*how big* anything is; it gives an ordered id list and the family, and the shape
falls out.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
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

DEFAULT_LAYOUT = ("grid", "lr")
"""A group that was never arranged: equal split, left-to-right."""


class LayoutError(ValueError):
    """A proposed arrangement cannot be laid out (bad family/dir, wrong item count)."""


@dataclass
class Item:
    """A windowable thing in the materialization pool.

    ``group`` empty means the item is on the desktop. ``service`` is the mesh
    address this item was adopted from, when it came from a webview service —
    empty for items the model opened by hand.
    """

    id: str
    url: str
    label: str = ""
    group: str = ""
    service: str = ""
    icon: str | None = None


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


@dataclass
class Group:
    """An arrangement: an ordered id list plus the layout it was told to take."""

    name: str
    ids: list[str] = field(default_factory=list)
    family: str = DEFAULT_LAYOUT[0]
    dir: str = DEFAULT_LAYOUT[1]
    fullscreen: str | None = None


def compute_layout(n: int, family: str, dir: str) -> Layout:
    """Resolve ``(n, family, dir)`` into grid coordinates.

    ``grid`` splits equally; ``stack`` gives one master (index 0) plus a strip of
    the rest. ``dir`` is the axis: ``lr`` fills a row first, ``tb`` a column first.
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
    """Window state: the pool, the groups, the active arena, desktop, fullscreen."""

    DESKTOP = ""
    """The desktop arena — no group. ``arena()`` returns this when the desktop shows."""

    def __init__(self) -> None:
        self._items: dict[str, Item] = {}
        self._groups: dict[str, Group] = {}
        self._arena: str = self.DESKTOP
        self._dismissed: dict[str, float] = {}
        """service address → the item id a human/model dismissed; a tombstone so
        auto-adoption does not resurrect it until the service stops and returns."""
        self._next_index: int = 0
        """Monotonic item handle. Never recycled — an id must stay stable so a
        handle the model holds does not drift onto a later item after a destroy."""

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
        g = self._groups.get(group)
        return list(g.ids) if g is not None else []

    def arena(self) -> str:
        """What the screen shows: a group name, or ``DESKTOP`` (empty string)."""
        return self._arena

    def arena_items(self) -> list[str]:
        """The ids laid out on the stage — the active group's, or empty on the desktop."""
        return self.group_items(self._arena) if self._arena else []

    def active_group(self) -> Group | None:
        """The active group object, or None when the desktop is showing."""
        return self._groups.get(self._arena) if self._arena else None

    def desktop_items(self) -> list[str]:
        """Ids on the desktop — in the pool but in no group."""
        return [i.id for i in self._items.values() if not i.group]

    def fullscreen(self) -> str | None:
        """The fullscreen item of the active group (per-group state)."""
        g = self.active_group()
        return g.fullscreen if g is not None else None

    def family(self) -> str:
        g = self.active_group()
        return g.family if g is not None else DEFAULT_LAYOUT[0]

    def dir(self) -> str:
        g = self.active_group()
        return g.dir if g is not None else DEFAULT_LAYOUT[1]

    def layout(self) -> Layout | None:
        """The active group's resolved layout, or None on the desktop."""
        g = self.active_group()
        if g is None or not g.ids:
            return None
        return compute_layout(len(g.ids), g.family, g.dir)

    def group_of(self, item_id: str) -> str:
        item = self._items.get(item_id)
        return item.group if item is not None else ""

    def service_of(self, item_id: str) -> str:
        item = self._items.get(item_id)
        return item.service if item is not None else ""

    def item_of_service(self, address: str) -> Item | None:
        for item in self._items.values():
            if item.service and item.service == address:
                return item
        return None

    def adopted(self) -> dict[str, str]:
        """service address → item id, for every item that came from a service."""
        return {i.service: i.id for i in self._items.values() if i.service}

    def dismissed(self) -> set[str]:
        return set(self._dismissed)

    # -- materialization ---------------------------------------------------

    def open(
        self,
        url: str,
        *,
        label: str = "",
        group: str = "",
        service: str = "",
        icon: str | None = None,
    ) -> Item:
        """Materialize a window into the pool; the store assigns its handle.

        The returned item's ``id`` is a monotonic index the caller uses as the
        handle for every later command — the model never invents a name. Default
        resting place is the desktop. Pass ``group`` to land it directly in a
        group (creating the group on first use). ``service`` records the mesh
        address it was adopted from, when it came from a webview service.
        """
        if not url:
            raise ValueError("url is required")
        if group and group not in self._groups:
            self._groups[group] = Group(name=group)
        item_id = str(self._next_index)
        self._next_index += 1
        item = Item(
            id=item_id, url=url, label=label or item_id, group=group,
            service=service, icon=icon,
        )
        self._items[item_id] = item
        if group:
            self._groups[group].ids.append(item_id)
        # A hand-opened item is never tombstoned — it is a deliberate act.
        if service:
            self._dismissed.pop(service, None)
        return item

    def navigate(self, item_id: str, url: str) -> Item:
        """Point a hand-opened item at a new url. Rejected for adopted views.

        An adopted item's url is owned by its node — repointing it here would
        make the screen lie about what that node is. Only hand-opened items
        (``service`` empty) may be re-pointed; a ``page.goto``-style move.
        """
        item = self._require(item_id)
        if item.service:
            raise ValueError(
                f"#{item_id} is a webview window — its url belongs to its node, "
                f"not the screen"
            )
        if not url:
            raise ValueError("url is required")
        item.url = url
        return item

    def adopt(
        self,
        address: str,
        url: str,
        *,
        label: str,
        icon: str | None = None,
    ) -> Item | None:
        """Materialize a window discovered on the mesh (webview service).

        Lands on the desktop, never in a group — a discoverer that quietly
        arranged things would be choosing layout, which is the model's job.
        Returns None when the address is tombstoned or already present; the
        caller treats that as "nothing to do".
        """
        if address in self._dismissed:
            return None
        if self.item_of_service(address) is not None:
            return None
        return self.open(url, label=label, service=address, icon=icon)

    def release(self, address: str) -> Item | None:
        """Drop the item adopted from a service that went away.

        A non-tombstone removal — the service is gone, not dismissed, so the
        tombstone is cleared and a later re-announce is adopted afresh.
        """
        item = self.item_of_service(address)
        self._dismissed.pop(address, None)
        if item is None:
            return None
        return self._remove(item.id)

    def dismiss(self, item_id: str) -> Item | None:
        """Send an item back to the desktop — off any group, still in the pool.

        A soft move: the item stays (and keeps its service), so it remains visible
        on the desktop. It is not a tombstone — only ``destroy`` is.
        """
        item = self._items.get(item_id)
        if item is None or not item.group:
            return None
        self._detach(item_id, item.group)
        item.group = ""
        self._prune_empty_groups()
        return item

    def destroy(self, item_id: str) -> Item | None:
        """Remove an item from the pool entirely.

        A hand-removal, not a service going away: if the item was adopted, its
        tombstone is raised so auto-adoption does not bring it straight back.
        """
        item = self._items.get(item_id)
        if item is None:
            return None
        if item.service:
            self._dismissed[item.service] = time.time()
        return self._remove(item_id)

    # -- grouping ----------------------------------------------------------

    def arrange(
        self,
        group: str,
        ids: list[str],
        *,
        family: str,
        dir: str,
    ) -> Layout:
        """Put ``ids`` into ``group``, in that order, with that layout, and show it.

        This is the model's whole move: pick items (any subset of the pool), name
        the arrangement, and it exists. An item may be in only one group, so any
        listed item currently elsewhere in the pool is moved here. The group is
        created on first use; arranging it never deletes other groups.
        """
        if not group:
            raise ValueError("group name is required")
        if not ids:
            raise ValueError("arrange needs at least one item")
        unknown = [i for i in ids if i not in self._items]
        if unknown:
            raise ValueError(
                f"unknown item(s): {', '.join(unknown)} — items() lists them"
            )
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate ids in arrange: {', '.join(ids)}")
        layout = compute_layout(len(ids), family, dir)
        wanted = set(ids)

        # Every listed item may currently be elsewhere (another group or the
        # desktop); detach it there before it lands here.
        for item_id in ids:
            item = self._items[item_id]
            if item.group and item.group != group:
                self._detach(item_id, item.group)

        target = self._groups.get(group)
        if target is None:
            target = Group(name=group)
            self._groups[group] = target
        # Items pulled out of this group by an earlier move must not linger.
        target.ids = [i for i in target.ids if i in wanted] + [
            i for i in ids if i not in target.ids
        ]
        target.family = family
        target.dir = dir
        if target.fullscreen and target.fullscreen not in wanted:
            target.fullscreen = None
        for item_id in ids:
            self._items[item_id].group = group
        self._prune_empty_groups()
        self._arena = group
        return layout

    def activate(self, group: str) -> None:
        """Show a group, or the desktop (``""``)."""
        if group == self.DESKTOP:
            self._arena = self.DESKTOP
            return
        if group not in self._groups:
            raise ValueError(f"no group {group!r} — groups(): {self.groups()}")
        self._arena = group

    def float_all(self) -> list[str]:
        """Send the active group's items back to the desktop; show the desktop."""
        freed = self.group_items(self._arena)
        for item_id in freed:
            item = self._items.get(item_id)
            if item is not None:
                item.group = ""
        g = self._groups.get(self._arena)
        if g is not None:
            g.ids = []
            g.fullscreen = None
        self._prune_empty_groups()
        self._arena = self.DESKTOP
        return freed

    def set_fullscreen(self, item_id: str | None) -> None:
        """Set the active group's fullscreen item (None exits)."""
        g = self.active_group()
        if g is None:
            raise ValueError("no active group — fullscreen needs a group on stage")
        if item_id is not None:
            self._require(item_id)
            if item_id not in g.ids:
                raise ValueError(
                    f"#{item_id} is not in the active group #{g.name}"
                )
        g.fullscreen = item_id

    # -- internals ---------------------------------------------------------

    def _detach(self, item_id: str, group: str) -> None:
        """Pull an item out of ``group``'s order (and its fullscreen claim)."""
        g = self._groups.get(group)
        if g is None:
            return
        if item_id in g.ids:
            g.ids.remove(item_id)
        if g.fullscreen == item_id:
            g.fullscreen = None

    def _prune_empty_groups(self) -> None:
        for name in [n for n, g in self._groups.items() if not g.ids]:
            del self._groups[name]
            if self._arena == name:
                self._arena = self.DESKTOP

    def _remove(self, item_id: str) -> Item | None:
        item = self._items.pop(item_id, None)
        if item is None:
            return None
        if item.group:
            g = self._groups.get(item.group)
            if g is not None and item_id in g.ids:
                g.ids.remove(item_id)
                if g.fullscreen == item_id:
                    g.fullscreen = None
        self._prune_empty_groups()
        return item

    def _require(self, item_id: str) -> Item:
        item = self._items.get(item_id)
        if item is None:
            raise KeyError(f"no item {item_id!r} — items(): {[i.id for i in self.items()]}")
        return item
