"""The agent layer — what an agent is, who holds it, and what it can be asked to do.

Three types, one question each:

- ``AgentSession`` — one carrier: the thing that gets driven. ``send`` / ``cancel`` /
  ``configure``, a cheap state read, and two observation faces. It ends with its session.
- ``Agents`` — the collection: who exists now, how one joins (``open`` / ``watch``), how one leaves
  (``retire``), and the channel that shows it. It holds no driving verbs: ask it for a session and
  drive that.
- ``Hive`` (``hive.py``) — the place: which workspaces may be handed out, which presets may compose
  an agent.

Two identity types, never interchangeable:

- ``SessionName`` — the address inside a collection. The caller gives it; every collection-level
  verb addresses by it.
- ``SessionId`` — the engine's handle for one carrier. It appears when taking up an existing session
  (``watch``) and in events; it is not an address.

Contracts on this surface:

- ``send`` returns nothing and never waits; an answer arrives through ``on_turn_end``.
- A preset is held as a **snapshot taken at creation**; ``state().preset`` keeps the id for a caller
  that wants the current declaration.
- ``open`` never reuses a held name; ``retire`` stops the runtime before dropping the name.
- This module does not import the trajectory layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.message import Message

__all__ = [
    "AgentPhase",
    "AgentPreset",
    "AgentSession",
    "AgentSessionEvent",
    "AgentState",
    "Agents",
    "EventHook",
    "MessageMode",
    "SessionId",
    "SessionName",
    "Workspace",
    "WorkspaceName",
]


# --------------------------------------------------------------------------- #
# Identities — named, so they cannot be swapped by accident
# --------------------------------------------------------------------------- #

SessionId = str
"""One carrier's identity, assigned by the engine. Appears only when taking up an existing session."""

SessionName = str
"""One agent's address inside a collection, assigned at ``open``/``watch``. This layer's word."""

WorkspaceName = str
"""A declared place's name — an id or a title from ``Hive.workspaces()``. Never a path."""


# --------------------------------------------------------------------------- #
# Places and templates
# --------------------------------------------------------------------------- #


class Workspace(BaseModel):
    """A place an agent can be created inside.

    A workspace is the engine's grouping of sessions by directory; the **permission boundary** is
    the session's ``cwd``. A session may move between workspaces — its ``cwd`` is what decides what
    it may touch.

    ``id`` is what the engine addresses it by, ``name`` is what callers address it by, and ``path``
    is data about the place, never a way to name it. An engine that gives no title leaves ``name``
    empty; the hive then derives one from the path.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(default="")
    name: str = Field(default="", description="the display name callers may use as a WorkspaceName.")
    path: str = Field(
        default="",
        description="an **absolute** host path — the directory this workspace groups. This layer "
        "never resolves a relative path: an engine that accepts one resolves it, and the value "
        "read back is always absolute.",
    )
    title: str = Field(default="", description="display title; the engine defaults it to the basename.")


class AgentPreset(BaseModel):
    """A template — what an agent is composed of, not who it is.

    ``id`` is the identity an agent stores. ``name`` is a display label and is often absent, so it
    can never be the identity.

    What the template can do is its composition, which the engine states as readable content; this
    layer states no capability list of its own. ``config_json_schema`` describes the patch
    ``configure`` accepts and is filled by the concrete implementation — its pair is
    ``AgentState.config``, which holds the values currently in force.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="stable identifier; also the label's fallback.")
    name: str = Field(default="", description="display label; empty when the deployment gave none.")
    description: str = Field(default="", description="one sentence on what this preset is for.")
    manual: str = Field(
        default="",
        description=(
            "how to use it well, in the hive's own words. The engine states a composition, not a "
            "usage; this is where a hive teaches one."
        ),
    )
    config_json_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object"},
        description=(
            "the JSON schema of the patch ``configure`` accepts, filled by the concrete "
            "implementation. The default states no constraint."
        ),
    )


# --------------------------------------------------------------------------- #
# State — read from the engine, never our own vocabulary
# --------------------------------------------------------------------------- #


class AgentPhase(str, Enum):
    """A lifecycle position — one event per position, never one per token.

    ``reasoning`` and ``tool_use`` refine ``running``; an engine that cannot split them emits
    ``running``.
    """

    created = "created"
    """The carrier exists. The first position a subscriber can see."""

    input = "input"
    """Something went in: a message reached the carrier (accepted, claimed, or rejected)."""

    reasoning = "reasoning"
    """The carrier is thinking — a refinement of ``running``."""

    tool_use = "tool_use"
    """The carrier called a tool — a refinement of ``running``."""

    running = "running"
    """The carrier is working, unspecified. ``event_name`` says more, in the engine's words."""

    output = "output"
    """Something came out. ``description`` carries what a human reads; ``payload`` the raw data."""

    idle = "idle"
    """The carrier stopped working and is waiting."""

    closed = "closed"
    """The carrier is finished. **The last event of a session**: nothing follows it, and a
    subscriber that misses it has no other way to tell "ended" from "went quiet"."""


class AgentState(BaseModel):
    """The cheap read — what a driver needs to decide, without touching the engine.

    Two kinds of data: the structural facts fixed at open (identity, where it acts, which preset
    composed it), and ``config`` — the values ``configure`` has put in force.
    """

    model_config = ConfigDict(extra="allow")

    id: SessionId = Field(description="the carrier's engine id.")
    name: SessionName = Field(default="", description="the name this agent is addressed by in its collection.")
    title: str = ""
    preset: str = Field(default="", description="the preset's **id**, not the preset.")
    cwd: str = Field(
        default="",
        description="an **absolute** path: the sandbox's writable root, i.e. the permission boundary.",
    )
    workspace_id: str = Field(
        default="",
        description="the grouping the session belongs to — an **id, never a path**. Not a permission "
        "boundary; resolve it through the hive's workspace list.",
    )
    phase: AgentPhase = Field(
        default=AgentPhase.created,
        description="the last lifecycle position seen for this carrier. The event stream is the "
        "truth; this is its latest reading, so a listing never has to replay it.",
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="the runtime configuration match the agent preset config json schema.",
    )


# --------------------------------------------------------------------------- #
# The event protocol — lifecycle observation, not a chunk stream
# --------------------------------------------------------------------------- #


class AgentSessionEvent(BaseModel):
    """One lifecycle event from one carrier — what an observability surface consumes.

    Three fields, three jobs:

    - ``event_name`` — the engine's own word, untouched; this layer invents no event vocabulary.
    - ``description`` — what a human reads.
    - ``payload`` — the concrete data behind the event, for debugging. Optional by contract: a
      consumer that requires it is reading an engine's shapes.
    """

    model_config = ConfigDict(extra="allow")

    hive: str = Field(description="the hive that owns this carrier.")
    engine: str = Field(description="which engine produced this event, as ``name@version``.")
    session_id: SessionId = Field(description="the carrier this event came from.")
    session_name: SessionName = Field(default="", description="the name this carrier is addressed by.")
    preset: str = Field(default="", description="the preset's id, as the carrier reports it.")
    event_name: str = Field(
        default="",
        description="the engine's own name for the event, passed through untouched — this layer "
        "invents no event vocabulary.",
    )
    description: str = Field(default="", description="what a human reads.")
    phase: AgentPhase
    payload: dict[str, Any] = Field(default_factory=dict, description="the concrete agent event data.")


EventHook = Callable[[AgentSessionEvent], Awaitable[None] | None]


# --------------------------------------------------------------------------- #
# The carrier — one session, driven
# --------------------------------------------------------------------------- #


class MessageMode(str, Enum):
    """Where an accepted message lands.

    The names are this layer's; the mapping onto an engine's verbs is the implementation's.
    """

    inject = "inject"
    """Inserted immediately, and the carrier is **not** woken: it waits for something else."""

    enqueue = "enqueue"
    """Queued, and the carrier **is** woken: it runs after the current turn."""

    steer = "steer"
    """The turn in flight is stopped, and this message is answered now."""


class AgentSession(ABC):
    """One carrier session — the living half of a delegation.

    It ends with its session. An identity that outlives a carrier belongs to the trajectory layer
    and is not reachable from here — which is why there is no ``reboot``.
    """

    @property
    @abstractmethod
    def name(self) -> SessionName:
        """The name the runtime gave this carrier — the address its collection knows it by."""

    @property
    def id(self) -> SessionId:
        """The carrier's engine id. **Not** an address: the collection addresses by ``name``."""
        return self.state().id

    @property
    def title(self) -> str:
        """A human-facing label. Display only: addressing never goes through it."""
        return self.state().title

    @property
    @abstractmethod
    def preset(self) -> AgentPreset:
        """The preset as it was when this carrier was created — a snapshot, never a lookup.

        A deployment may replace a preset declaration at any moment while this carrier keeps
        running what it was created with, so the snapshot is what this agent is composed of.
        ``state().preset`` carries the id for a caller that wants the current declaration.
        """

    @property
    def cwd(self) -> str:
        """An **absolute** path: where this agent acts, and therefore what it may act on.

        Fixed at open. This layer never hands out a relative path and never guesses one.
        """
        return self.state().cwd

    @property
    def workspace_id(self) -> str:
        """The workspace this carrier is grouped under — an **id, not a path**.

        Grouping is not permission: ``cwd`` decides what the agent may touch, a workspace decides
        which list it appears in.
        """
        return self.state().workspace_id

    @abstractmethod
    def state(self) -> AgentState:
        """The cheap read. Never asks the engine for what it already mirrors."""

    @abstractmethod
    async def send(
        self,
        *messages: Message,
        mode: MessageMode = MessageMode.enqueue,
    ) -> None:
        """Hand one or more messages to the carrier. **Returns nothing, and never waits.**

        Acceptance is the only fact this call reports, carried by the absence of an exception. The
        answer arrives through ``on_turn_end``, never as a return value.
        """

    @abstractmethod
    async def cancel(self, reason: str = "") -> None:
        """Stop the turn in flight. Work already done is untouched.

        ``reason``, when given, reaches the carrier itself, so an intended stop is distinguishable
        from a failure at both ends.
        """

    @abstractmethod
    async def configure(self, patch: dict[str, Any]) -> str:
        """Patch the carrier's settings; the change applies to subsequent turns.

        The patch must match ``AgentPreset.config_json_schema``. Absent fields keep their current
        value — nothing is reset by omission. Returns a short description of what is now in force.
        """

    @abstractmethod
    def on_session_event(self, hook: EventHook) -> Callable[[], None]:
        """Observe this carrier's lifecycle events; returns the disposer.

        The thin protocol: a ``phase``, the engine's ``event_name``, a displayable ``description``
        and ``payload``. An implementation may add richer hooks of its own; they are not part of
        this contract.
        """

    @abstractmethod
    def on_turn_end(
        self,
        callback: Callable[[str, AgentSessionEvent], Awaitable[None] | None],
    ) -> Callable[[], None]:
        """Observe the end of a turn: ``(final_answer, event)``, **once per turn**.

        A turn that uses tools produces several assistant messages — the preamble, then the calls —
        and only the last one is the answer. Process data belongs to ``on_session_event``; this
        face promises exactly one thing per turn, as **plain text**, so a caller never has to guess
        which message counted. A turn that ends without an answer emits nothing here.
        """


# --------------------------------------------------------------------------- #
# The collection — who exists, and the channel that shows them
# --------------------------------------------------------------------------- #


class Agents(ABC):
    """The agents one hive holds — the roster, the join/leave verbs, and its channel.

    It holds no driving verbs: ``send`` / ``cancel`` / ``configure`` belong to a session, so a
    caller asks for one (``get``) and talks to that.

    ``as_channel()`` renders the collection as a child channel of its owner's, never as a root of
    its own. The roster reaches the ghost as **named notices** — one fragment per held agent, so an
    agent that disappears can say so.

    A hive with no collection is legal: ``Hive.agents`` raises ``NotImplementedError``.
    """

    # -- discovery: data, not handles -------------------------------------------- #

    @abstractmethod
    def list_sessions(self) -> list[AgentSession]:
        """Every agent currently held, in a stable order. The enumeration a notice renders from."""

    @abstractmethod
    def get(self, name: SessionName) -> AgentSession:
        """The session this collection holds under ``name``.

        :raise NameError: nothing is held under that name.
        """

    @abstractmethod
    def on_event(self, callback: EventHook) -> Callable[[], None]:
        """Observe every held session through one subscription; returns the disposer.

        Events carry ``session_id`` and ``session_name``, so one subscription covers a surface that
        watches several agents at once.
        """

    # -- lifecycle ----------------------------------------------------------------- #

    @abstractmethod
    async def open(
        self,
        name: SessionName,
        *,
        workspace: WorkspaceName,
        preset: str = "",
        readonly: bool = False,
    ) -> AgentSession:
        """Create one agent inside a declared workspace, under ``name``.

        - ``name``: the address the caller wants. A name already held raises rather than rebinding.
        - ``workspace``: a place from ``Hive.workspaces()`` — an id or a title, never a path. An
          unknown name raises.
        - ``preset``: the composition; empty means the deployment's default.
        - ``readonly``: a **creation-time** property, not a later patch. An implementation that
          cannot express it at creation must set the policy before the first message is accepted.
          The invariant: no turn runs under a permission nobody asked for.
        """

    @abstractmethod
    async def watch(self, name: SessionName, session_id: SessionId) -> AgentSession:
        """Take up a session that exists but is not held here, under ``name``.

        The only way back in, and an explicit act: a collection promises nothing across a restart.
        What a session id can still reach is the implementation's answer.

        :raise KeyError: no such session exists to watch.
        """

    @abstractmethod
    async def retire(self, name: SessionName, reason: str = "") -> None:
        """End an agent's membership: stop its carrier, then drop it from the roster.

        The order is part of the contract: cancel the turn (with ``reason``) → dispose the
        subscriptions → close the carrier → drop it from the roster **last**, so a listing taken
        mid-teardown shows something winding down rather than something that vanished. Nothing the
        carrier emits after this call may reach a caller.

        An orderly end emits a terminal event; a process that dies emits nothing.
        """

    @abstractmethod
    async def clear(self) -> None:
        """Retire everything held. The collection ends; the place it lived in does not."""

    # -- presentation ----------------------------------------------------------------- #

    def as_channel(self) -> Channel | None:
        """Render this collection as its own child channel — the control surface for it.

        The collection produces its channel so that every consumer describes the same roster in the
        same words. An implementation may bind extra detail onto the same object.

        ``None`` is legal: a collection without a channel is one the ghost cannot drive.
        """
        return None
