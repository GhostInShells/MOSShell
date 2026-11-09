"""The agent layer — three layers, two identities, one direction.

## The three layers

- **``AgentSession``** — one carrier: the thing that actually gets driven. It has verbs
  (``send`` / ``cancel`` / ``configure``) and two observation faces, and it dies with its session.
- **``Agents``** — the collection: who exists right now, how one joins, how one leaves. It owns the
  roster and the channel, and it never drives anything itself — you ask it for a session and then
  talk to that.
- **``Hive``** (``hive.py``) — the place: which workspaces may be handed out, which presets may
  compose an agent. Policy, stable, and not this file.

The cut is by **ownership and lifetime**, not by feature. Anything with a lifetime lives in
``Agents``; anything that changes the world lives on ``AgentSession``; anything that is a rule
lives in ``Hive``. A verb that appears in two of them is a verb whose owner has not been decided.

## The two identities

``SessionName`` and ``SessionId`` are different things with different owners, and confusing them is
how this layer rots:

- **the name** belongs to the runtime: the caller gives it at ``open``/``watch``, the collection
  keys its roster by it, and every collection-level verb addresses by it.
- **the session id** belongs to the engine: it identifies one carrier, appears only when a session
  that already exists is being taken up (``watch``), and is carried in every event.

Neither converts into the other by convention. A name that outlives its carrier is a trajectory
concept; this layer has no such word.

## Decisions this file has already taken

**Async control, unconditionally.** A synchronous prompt that returns a final answer is a blocking
command in CTML, and a driver holding many agents could only drive one at a time. So ``send``
accepts and returns; what came out arrives as events. The synchronous convenience — send, then wait
— is a caller-side loop over the same stream, never a primitive here.

**Two observation faces, and they are not the same thing.** *Lifecycle events* feed the
observability surface: one event per position, never a chunk delta, with the engine's own
``event_name`` passed through untouched and the layer's own ``phase`` as the only vocabulary it
owns. *The output face* (``on_output``) is the precise field a ghost needs: the turn's final answer
as plain text. Conflating them is how a lifecycle stream gets mistaken for a return value.

**What is deliberately absent.** No ops, no commits (nothing parses a reply's first line); no
anchors or coordinates (ways of pointing at a trajectory); no fan-out (an agent cannot spawn an
agent — the collection opens, and that cut keeps a team an explicit act); no live relation objects
(an agent carries its preset as a **snapshot taken at creation**, plus the id in its state for a
caller that wants to join to the current declaration — a live lookup would quietly re-bind a
running agent to a template it never ran).

The rule that keeps this file honest: **it must not import the trajectory.** If a capability needs
a branch to exist, it does not live here.
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

    Two different facts wear this name and must not be conflated: a workspace is the engine's
    **grouping** of sessions by directory, while the **permission boundary** is the session's
    ``cwd`` — the sandbox's writable root. A session can move between workspaces; its ``cwd`` is
    what decides what it may touch.

    ``path`` is data about the place, never a way to name it: callers address a place by
    ``WorkspaceName``, so a workspace without one cannot be handed out.
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
    """A template — what an agent is *composed of*, not who it is.

    ``id`` is the whole relationship: an agent stores the id it was created with. ``name`` is a
    display label and is often absent (measured on dsh 0.2.0-rc.2: four of the five shipped presets
    carry no name at all), so it can never be the identity.

    There is no capability tag list here on purpose. What a preset can do is its **composition** —
    the plugins and tools it mounts — which the engine states as readable content. A parallel tag
    vocabulary would have to be invented by us, could never be satisfied by an adapter, and would
    be a second source of truth for a fact the engine already owns.

    The patch surface ``configure`` accepts **is** here, but as an opening rather than as content.
    ``config_json_schema`` is a field a concrete implementation fills with its engine's schema;
    this file never states what is in it. Its pair lives on the state: schema on the template,
    current values on the instance.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(description="stable identifier; also the label's fallback.")
    name: str = Field(default="", description="display label; empty when the deployment gave none.")
    description: str = Field(default="", description="one sentence on what this preset is for.")
    manual: str = Field(
        default="",
        description=(
            "how to use it well, in the hive's own words. Unfilled for now: the engine states a "
            "composition, not a usage — this is the place a hive will teach later, kept from the "
            "earlier contract so the intent is not lost."
        ),
    )
    config_json_schema: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object"},
        description=(
            "the JSON schema of the patch ``configure`` accepts, filled by the concrete "
            "implementation. The default is the empty schema — 'an object, no constraint stated' — "
            "which is honest: this layer knows nothing about it."
        ),
    )


# --------------------------------------------------------------------------- #
# State — read from the engine, never our own vocabulary
# --------------------------------------------------------------------------- #


class AgentPhase(str, Enum):
    """Where in the carrier's life this event happened.

    These are **lifecycle positions**, not stream deltas: one event per position, not one per
    token. That is what makes them drawable — a surface can place them on a timeline without
    knowing any engine.

    ``reasoning`` and ``tool_use`` are the two sub-phases engines commonly distinguish while
    working; an engine that cannot split them emits ``running`` and loses nothing.
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
    """The cheap read: everything a driver needs to decide, without touching the engine.

    Two kinds of data live here: the structural facts fixed when the carrier opened (who it is,
    where it acts, which preset composed it), and ``config`` — the values ``configure`` has put in
    force, shaped by the preset's ``config_json_schema``.
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
    """One lifecycle event from one carrier — the shape an observability surface consumes.

    Deliberately generic, because its first consumer draws pictures: a graph places these on a
    timeline without knowing any engine's vocabulary.

    Three fields, three jobs, and they must not be conflated:

    - ``event_name`` — the engine's own word, untouched. This layer invents no event vocabulary.
    - ``description`` — what a human reads. **The displayable field.**
    - ``payload`` — the concrete, already-filtered data behind the event. Debug-oriented: a
      broadcast link may drop it, and a consumer that requires it is reading an engine's shapes.
      It is not a chunk delta — this is a lifecycle event, one per position, not one per token.
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
    """Where an accepted message lands. Three boundaries — two cannot express the set.

    The names are ours and the mapping is the implementation's: an engine with two verbs covers
    these three by composing them, and a caller never has to know which.
    """

    inject = "inject"
    """Inserted immediately, and the carrier is **not** woken: it waits for something else."""

    enqueue = "enqueue"
    """Queued, and the carrier **is** woken: it runs after the current turn."""

    steer = "steer"
    """The turn in flight is stopped, and this message is answered now."""


class AgentSession(ABC):
    """One carrier session — the living half of a delegation.

    The other half is an identity that outlives this carrier, and it belongs to the layer that owns
    a trajectory. This object dies with its session, by design: that is why it has no ``reboot``.
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
        """The preset **as it was when this carrier was created** — a snapshot, never a lookup.

        A deployment may replace a preset declaration at any moment, while this carrier keeps
        running the composition it was actually created with. So the snapshot is the only honest
        answer to "what is this agent composed of"; ``state().preset`` carries the id for a caller
        that wants to join to the current declaration itself.
        """

    @property
    def cwd(self) -> str:
        """An **absolute** path: where this agent acts, and therefore what it may act on.

        Fixed when the carrier was opened. This layer never hands out a relative path and never
        guesses what one would have meant.
        """
        return self.state().cwd

    @property
    def workspace_id(self) -> str:
        """The workspace this carrier is grouped under — an **id, not a path**.

        Grouping and permission are two facts, and this is the weaker one: ``cwd`` decides what the
        agent may touch, a workspace only decides which list it shows up in.
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

        There is no return value, and that is not an oversight: "accepted" is the only fact this
        call can honestly report, and it is already carried by the absence of an exception. The
        answer is not returned here — it arrives as an ``output`` event. Returning it would be the
        synchronous protocol wearing an async signature.
        """

    @abstractmethod
    async def cancel(self, reason: str = "") -> None:
        """Stop the turn in flight. Work already done is untouched.

        ``reason`` travels with the cancellation so the carrier — and whoever reads the trace later
        — can tell an intended stop from a failure. It is optional because "stop" is a complete
        instruction on its own.
        """

    @abstractmethod
    async def configure(self, patch: dict[str, Any]) -> str:
        """Patch the carrier's settings, applied at the next turn boundary.

        The patch must match ``AgentPreset.config_json_schema``. That field is the opening; this
        docstring states nothing about its content, because the shape is the engine's. Absent
        fields keep their current value — nothing is reset by omission. Returns a short description
        of what is now in force.
        """

    @abstractmethod
    def on_session_event(self, hook: EventHook) -> Callable[[], None]:
        """Observe this carrier's lifecycle events; returns the disposer.

        The thin protocol: a ``phase``, the engine's own ``event_name``, a displayable
        ``description``, and ``payload`` for whoever wants the raw data. An implementation that
        wants to hand out its own richer structures adds its own hooks — they are not this
        contract's business.
        """

    @abstractmethod
    def on_output(
        self,
        callback: Callable[[str, AgentSessionEvent], Awaitable[None] | None],
    ) -> Callable[[], None]:
        """Observe finished answers: ``(final_answer, event)`` per output.

        The one precise field a ghost needs — the turn's final answer as **plain text** — paired
        with the event that carried it, so a caller can tell where the text came out without
        reading ``payload`` and without knowing the engine.
        """


# --------------------------------------------------------------------------- #
# The collection — who exists, and the channel that shows them
# --------------------------------------------------------------------------- #


class Agents(ABC):
    """The agents one hive holds — the roster, the join/leave verbs, and the channel.

    **Why this is not the hive.** A hive answers *where may an agent live* and *what may it be
    composed of*: policy, stable, owned by whoever governs the place. This answers *who exists
    right now, and how does one come and go*: a runtime collection with a lifetime.

    **Why the driving verbs are not here.** ``send`` / ``cancel`` / ``configure`` belong to a
    carrier, and a collection is not one. Ask for the session (``get``) and talk to it. Duplicating
    a verb on both layers is how a protocol loses track of who owns the action.

    **The channel is a child.** ``as_channel()`` renders this collection as a channel mounted under
    its owner's main channel, never as a root of its own: the node is the address, this is an organ.
    The roster reaches the ghost as **named notices** — one fragment per held agent, so an agent
    that disappears can say so, which a single folded notice cannot.

    A hive with no collection at all is legitimate (a read-only surface, a listing). It raises
    ``NotImplementedError`` from ``Hive.agents`` rather than returning an empty collection that
    would look like "nobody is running" when the truth is "this hive cannot run anything".
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

        The events already carry ``session_id`` and ``session_name``, which is why one collection
        subscription is enough for a surface that watches several agents at once.
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

        - ``name`` is the address the caller wants; the collection takes it and holds the agent by
          it. Asking for a name that is already held raises rather than silently rebinding.
        - ``workspace`` names a place from ``Hive.workspaces()`` — an id or a title, never a path.
          An unknown name raises: a permission boundary is a declared place, not something typed.
        - ``preset`` is the composition; empty means the deployment's default.
        - ``readonly`` is a **creation-time** property, not a later patch. An implementation that
          cannot express it at creation must set the policy before the first message is accepted,
          and say so; the invariant is that no turn runs under a permission nobody asked for.
        """

    @abstractmethod
    async def watch(self, name: SessionName, session_id: SessionId) -> AgentSession:
        """Take up a session that exists but is not held here, under ``name``.

        The only way back in, and deliberately an explicit act: a plain collection promises nothing
        across a restart, so anything "restored" automatically would be a guess dressed as a fact.
        What a session id can still reach is the implementation's answer.

        :raise KeyError: no such session exists to watch.
        """

    @abstractmethod
    async def retire(self, name: SessionName, reason: str = "") -> None:
        """End an agent's membership: stop its carrier, then drop it from the roster.

        Order is the contract, because each step loses something if moved: stop the producer →
        emit the terminal event → wait for that dispatch to return → dispose the hooks → close the
        carrier → drop it from the roster **last**, so a listing taken mid-teardown shows something
        winding down rather than something that vanished.

        The terminal event is promised for an orderly shutdown only. A process that dies emits
        nothing, and that gap is the reader's watch, not this protocol's promise.
        """

    @abstractmethod
    async def clear(self) -> None:
        """Retire everything held. The collection ends; the place it lived in does not."""

    # -- presentation ----------------------------------------------------------------- #

    def as_channel(self) -> Channel | None:
        """Render this collection as its own child channel — the ghost's control surface for it.

        Here rather than inside a builder, because a channel *is* the wiring: if the collection did
        not produce it, every consumer would write its own, and the second one would describe the
        same roster in different words. An implementation may bind whatever extra detail its
        channel needs onto the same object; this face only promises the entry point.

        ``None`` is a legal answer, and it is not a defect: a collection without a channel is one the
        ghost cannot drive — a fixture, a probe, a runtime still being assembled. Saying so beats
        returning a channel that describes nothing.
        """
        return None
