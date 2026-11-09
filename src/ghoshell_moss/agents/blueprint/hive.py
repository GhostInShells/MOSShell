"""The hive — the place agents are made in: where they may live, and what they may be.

A hive answers exactly two questions, and both are policy:

- **where** — the declared places (``Workspace``) an agent may be created inside;
- **what** — the templates (``AgentPreset``) it may be composed from.

Neither changes because an agent did something. That is the line this file draws: anything with a
lifetime — who exists, what they are doing, how they end — lives in ``Agents``, which is what this
place hands out. A verb that answers "what is happening" does not belong here, and a verb here that
needs to know an agent exists is a sign the verb was put on the wrong side.

A concrete hive is a shell's governance surface for **one engine generation** (``DshHive`` for dsh
0.2.0, say), not a runtime the shell wraps around an engine: it reads the engine's own declarations
rather than keeping a roster of its own.

A hive without a collection is legitimate — a read-only surface, a listing, a probe — and says so
by raising from ``agents()`` instead of returning an empty collection.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ghoshell_moss.memento.abcd import Memento
from ghoshell_moss.core.concepts.channel import Channel
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
        """The engine family this hive drives, e.g. ``dsh``.

        Paired with ``version`` rather than packed into one string: the family and the version
        change for different reasons, and a consumer that routes on one should not have to split a
        string to read the other.
        """

    @property
    @abstractmethod
    def version(self) -> str:
        """The engine version this hive speaks, e.g. ``0.2.0-rc.2``."""

    @property
    @abstractmethod
    def root(self) -> Path:
        """**The maximum boundary** — the directory this hive may hand out places from.

        A workspace is a permission boundary an agent acts inside; if a hive could declare any
        directory on the machine, the boundary would be whatever someone typed. So the hive has an
        outer edge of its own (the project home, for a node), and a place outside it is refused
        rather than clamped: silently narrowing a path would hand the caller a boundary it did not
        ask for.
        """

    # -- the policy: places and templates ----------------------------------------- #

    @abstractmethod
    def workspaces(self) -> list[Workspace]:
        """The declared places an agent may be created inside.

        A read over the engine's own list, never a registry of ours: two sources of truth for
        "which directories exist" would drift the first time somebody mounted one by hand. A
        workspace that carries no name cannot be handed out, because a caller addresses it by name.
        """

    @abstractmethod
    def presets(self) -> list[AgentPreset]:
        """The templates this hive can compose an agent from.

        Also a read — presets are declared by the deployment, not authored here. Hence no filter
        argument: with a handful of entries the caller reads the roster.
        """

    @abstractmethod
    def preset(self, ref: str) -> AgentPreset:
        """One preset by id or name. Raises when it does not exist — never ``None``."""

    @abstractmethod
    async def create_workspace(self, path: str, name: str = "") -> Workspace:
        """Declare a new place, inside ``root``, and return it.

        **Nothing is predefined.** The environment does not hand a fresh node a workspace, and
        creating one is a runtime act rather than an install step — the caller asks for the place
        it wants, when it wants it.

        ``path`` is resolved **against ``root``** and must stay inside it: a declared place outside
        the hive's boundary is a permission the hive does not have, so it raises instead of
        clamping. ``name`` is what callers will address it by from then on; empty means the engine
        names it.

        Workspaces are the industry-level constraint mechanism, not a MOSS invention: an engine
        groups sessions by directory, and the directory is the sandbox's writable root. This verb
        is how a shell hands that constraint out deliberately.
        """

    # -- the runtime -------------------------------------------------------------- #

    @abstractmethod
    def agents(self) -> Agents:
        """The collection this place holds — the default product of a hive.

        A method rather than a property because it is a product, not a field: a hive may have to do
        work to hand one out. An implementation with nothing to hold raises ``NotImplementedError``
        here; that is a legitimate hive, not a broken one.
        """

    def as_channel(self) -> Channel | None:
        """Render this place as its own channel, or answer ``None``.

        The place has something to say that the collection does not: which workspaces may be handed
        out, which presets may compose an agent — the policy a caller reads before asking for
        anything. An implementation that has nothing to add answers ``None``; that is legal, and it
        only means the ghost learns the policy from somewhere else.
        """
        return None


class MementoAgentHive(Hive, ABC):
    """A hive that also keeps a trajectory — the store the agents' branches live in.

    Only the store is added here. Everything about places, presets and the collection is inherited,
    because the trajectory is the one thing this layer has that the plain one does not.
    """

    @property
    @abstractmethod
    def memento(self) -> Memento:
        """The store this hive's agents write their trajectory into."""
