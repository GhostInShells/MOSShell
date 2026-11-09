"""The hive — where delegations are born, and where the whole store is read from.

One hive is one memento store, one preset roster, and every ``MementoAgent`` spawned from
here. It is a shell-level blueprint rather than part of the agent contract: the agent owns
its own branch and writes the trace; the hive owns the address space and reads across
branches.

**The reading surface lives here, not on the agent.** Resolving a coordinate means resolving
a branch index inside the whole store; an agent knows only its own branch, and giving every
agent a global index would be the store leaking into every delegation. So:

- writing goes through the agent (it owns its branch);
- reading goes through the hive (it owns the address space).

A concrete hive is the shell's own governance surface for one engine generation (``DshHive``
for dsh 0.2.0, say), not a runtime the shell wraps around an engine. This file states only
what every hive must hold.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from ghoshell_moss.memento.abcd import CommitView, Memento
from ghoshell_moss.message import Message

from .memento_agent import AgentPreset, Mandate, MementoAgent

__all__ = ["Hive"]


class Hive(ABC):
    """One memento store, a set of presets, and every delegation spawned from here.

    **The reading surface lives here, not on the agent.** Reading a coordinate means resolving
    a branch index inside the whole store; an agent knows only its own branch, and giving every
    agent a global index would be the store leaking into every delegation. So:

    - writing goes through the agent (it owns its branch);
    - reading goes through the hive (it owns the address space).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The hive's name — presets are addressed inside it."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        ...

    @property
    @abstractmethod
    def memento(self) -> Memento:
        """The store. One hive, one store; nothing is shared across hives."""

    # -- presets --------------------------------------------------------------- #

    @abstractmethod
    def presets(self, *, capability: str = "") -> list[AgentPreset]:
        """Available templates; ``capability`` filters by what they can do."""

    @abstractmethod
    def preset(self, ref: str) -> AgentPreset:
        """Take one preset by id. Raises when it does not exist — never None."""

    # -- delegations ----------------------------------------------------------- #

    @abstractmethod
    async def spawn(
            self,
            *,
            mandate: Mandate,
            preset: str = "",
            workspace: str,
            title: str = "",
    ) -> MementoAgent:
        """Create one delegation: branch, declaration commit, first carrier.

        - ``workspace`` is the permission boundary and is **required**; a delegation without
          one would be a process with an unbounded sandbox, which is never what was meant.
        - ``title`` is for humans; addressing is always the branch name.
        - The declaration is written by **the delegator**, before any carrier exists. That is
          the one exception to "only the carrier writes the trace", and it is deliberate: the
          mandate is the delegator's act, and a branch whose first commit is missing has no
          answer to "what is this delegation".
        """

    @abstractmethod
    def find(self, name: str) -> MementoAgent | None:
        """Find one delegation by name — after a restart, this is how a ghost claims its work back."""

    @abstractmethod
    def agents(self, *, include_archived: bool = False) -> list[MementoAgent]:
        """The delegations this hive holds. Archived ones are hidden by default."""

    @abstractmethod
    async def prompt(
            self,
            to: str,
            message: Message,
            *,
            mode: Literal["queue", "steer"] = "queue",
    ) -> str:
        """Send one message to a delegation by name; unknown names raise, never silently drop."""

    @abstractmethod
    async def broadcast(
            self,
            message: Message,
            *,
            to: list[str] | None = None,
    ) -> dict[str, str]:
        """Send the same message to many delegations; returns one receipt per name.

        A fan-out here is a *read of who exists*, not a tree: the hive addresses a flat set.
        Hierarchical fan-out (an agent spawning agents) is not modelled this iteration.
        """

    # -- reading (the global address space) ------------------------------------ #

    @abstractmethod
    async def read(self, coord: str) -> CommitView:
        """One commit anywhere in the store, by coordinate (``{branch_index}-{seq}``).

        This is layer 1 of reading: the commit view, its note, and the references it carries.
        It works on an archived branch too — archiving removes the name pointer, not the data.
        """

    @abstractmethod
    async def transcript(self, coord: str, *, limit: int = 0) -> str:
        """Layer 2: the **model behaviour** of the span this commit belongs to, as a reading.

        Reads the engine session's events through the agreed rendering — user lines, model
        lines, tool calls by name and arguments; tool results, reasoning and mechanism frames
        are dropped on purpose. ``limit`` caps how many turns back from that commit; 0 = the
        session's own default.
        """

    @abstractmethod
    async def chat(self, coord: str, message: Message) -> Message:
        """Ask one question standing on that span — off the record.

        Nothing is appended, no session is created, nothing reaches the disk: the answer comes
        back and the usage is returned with it. Cheap when the span's session is still alive
        (its prefix is already assembled); expensive when it is not (a throwaway session is
        seeded from the log). See ``Not in scope`` for the anchor variant.
        """

    # -- workspaces ------------------------------------------------------------ #

    @abstractmethod
    def workspaces(self) -> list[str]:
        """The permission boundaries this hive can hand out.

        Modeled and displayed here because a workspace is a *place*, and places are the hive's
        business: an agent is created *inside* one and never owns one.
        """

# --------------------------------------------------------------------------- #
# Not in scope for this iteration, and not to be lost
# --------------------------------------------------------------------------- #
#
# - ``Hive.chat(anchor, message)``: the same question as ``chat``, addressed by a point read
#   from a file, with no live session and no coordinate resolution. This is what lets a ghost
#   carry a question about a dead span across processes.
