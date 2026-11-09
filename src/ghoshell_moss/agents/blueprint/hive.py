"""The hive — the place agents are made in: where they may live, and what they may be.

A hive answers two questions, and both are policy:

- **where** — the declared places (``Workspace``) an agent may be created inside, bounded by
  ``root``;
- **what** — the templates (``AgentPreset``) it may be composed from.

Anything with a lifetime — who exists, what they are doing, how they end — belongs to ``Agents``,
which is what this place hands out.

A concrete hive governs **one engine generation** and reads the engine's own declarations; it keeps
no roster of its own. A hive with no collection is legal and says so by raising from ``agents()``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ghoshell_moss.memento.abcd import Memento
from .agent import AgentPreset, Agents, Workspace

__all__ = ["Hive", "MementoAgentHive"]


class Hive(ABC):
    """One place: an engine generation, its declared workspaces, its declared presets."""

    # -- identity --------------------------------------------------------------- #

    @property
    @abstractmethod
    def name(self) -> str:
        """The hive's name. Addresses are resolved inside it, never globally."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """The engine family this hive drives, e.g. ``dsh``. Paired with ``version``, never packed
        into one string: the family and the version change for different reasons."""

    @property
    @abstractmethod
    def version(self) -> str:
        """The engine version this hive speaks, e.g. ``0.2.0-rc.2``."""

    @property
    @abstractmethod
    def root(self) -> Path:
        """**The maximum boundary** — the directory this hive may hand out places from.

        A place outside it is refused, never clamped: silently narrowing a path would hand the
        caller a boundary it did not ask for.
        """

    # -- the policy: places and templates ----------------------------------------- #

    @abstractmethod
    def workspaces(self) -> list[Workspace]:
        """The declared places an agent may be created inside, read from the engine — never a
        registry of our own, so there is one source of truth for "which directories exist". Only
        places inside ``root`` are listed."""

    @abstractmethod
    def presets(self) -> list[AgentPreset]:
        """The templates this hive can compose an agent from — also a read, since presets are
        declared by the deployment. No filter argument: the caller reads the list."""

    @abstractmethod
    def preset(self, ref: str) -> AgentPreset:
        """One preset by id or name. Raises when it does not exist — never ``None``."""

    @abstractmethod
    async def create_workspace(self, path: str, name: str = "") -> Workspace:
        """Declare a new place inside ``root`` and return it. Nothing is predefined.

        ``path`` is resolved **against ``root``** and must stay inside it; outside is refused, never
        clamped. ``name`` is what callers address it by from then on; empty means the engine names
        it.
        """

    # -- the runtime -------------------------------------------------------------- #

    @abstractmethod
    def agents(self) -> Agents:
        """The collection this place holds.

        A method, not a property: a hive may have to do work to hand one out. An implementation with
        nothing to hold raises ``NotImplementedError``.
        """

class MementoAgentHive(Hive, ABC):
    """A hive that also keeps a trajectory — the store the agents' branches live in.

    Only the store is added; places, presets and the collection are inherited.
    """

    @property
    @abstractmethod
    def memento(self) -> Memento:
        """The store this hive's agents write their trajectory into."""
