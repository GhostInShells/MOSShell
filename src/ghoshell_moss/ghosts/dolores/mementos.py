"""Memento owners of one ghost process, addressed by name.

A ghost home holds several owners — the ego's line, a dreamer's, a field's. They share one
memento home and differ by owner name, and the name is what a coordinate is scoped to: it is
passed in, never guessed from a path segment.

The manager holds only weak references. An owner lives while a caller holds it, which keeps a
long-running process from pinning every line it ever read; a writable gate is released when
its last holder goes away, the same way the filesystem registry behaves.
"""

from __future__ import annotations

import weakref
from abc import ABC, abstractmethod
from pathlib import Path

from ghoshell_container import INSTANCE, IoCContainer, Provider

from ghoshell_moss.core.blueprint.ghost import GhostHome
from ghoshell_moss.memento import Memento
from ghoshell_moss.memento._fs_memento import FsMemento, new_local_memento, owner_dir_for

__all__ = [
    "DOLORES_MEMENTO_ROOT",
    "DoloresMementos",
    "FsDoloresMementos",
    "FsDoloresMementosProvider",
]

# The memento home under a ghost home: one directory per owner name.
DOLORES_MEMENTO_ROOT = ".memento"


class DoloresMementos(ABC):
    """Resolve an owner name to its memento, inside one memento home."""

    @abstractmethod
    def get(self, owner: str, *, or_create: bool = False, writable: bool = True) -> Memento:
        """Open (or return) one owner.

        :raise FileNotFoundError: if owner does not exist and not create a one.
        """
        ...


class FsDoloresMementos(DoloresMementos):
    """Owners under one filesystem home, weakly held."""

    def __init__(self, home: Path) -> None:
        self._home = Path(home)
        self._owners: weakref.WeakValueDictionary[str, FsMemento] = weakref.WeakValueDictionary()

    @property
    def home(self) -> Path:
        """The directory every owner lives under."""
        return self._home

    def get(self, owner: str, *, or_create: bool = False, writable: bool = True) -> Memento:
        existing = self._owners.get(owner)
        if existing is not None:
            if writable:
                existing.open_for_write()
            return existing
        owner_dir = owner_dir_for(self._home, owner)
        if or_create:
            owner_dir.mkdir(parents=True, exist_ok=True)
        elif not owner_dir.is_dir():
            raise FileNotFoundError(f"memento owner {owner!r} does not exist under {self._home}")
        opened = new_local_memento(owner_dir, owner=owner, writable=writable)
        self._owners[owner] = opened
        return opened


class FsDoloresMementosProvider(Provider[DoloresMementos]):
    """Provide the ghost's memento owners as an IoC singleton.

    Cheap at boot: only the manager is built. An owner is opened when something asks for it
    by name, so a ghost that never touches its trajectory pays nothing.
    """

    def singleton(self) -> bool:
        return True

    def factory(self, con: IoCContainer) -> INSTANCE:
        ghost_home = con.force_fetch(GhostHome)
        return FsDoloresMementos(ghost_home.home() / DOLORES_MEMENTO_ROOT)
