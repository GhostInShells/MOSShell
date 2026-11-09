"""The agent layer — one carrier session, driven asynchronously.

## The decision this file takes, first

**Async control, unconditionally.** A synchronous ``prompt`` that returns a final answer is the
other industry shape, and taking it would have two consequences this project cannot accept:

- in CTML a synchronous prompt *is* a blocking command — the ghost's whole channel would stand
  still while one delegate thinks;
- a driver holding many agents could only ever drive one at a time, which turns a swarm back
  into a list.

So nothing here waits. ``prompt`` accepts and returns; what came out arrives as **events**. The
opposite of this is not "async as an extra": a sync abstraction that also carries a queue is two
protocols in one signature, and there is no way to tell which one a caller meant.

The synchronous convenience — send, then wait for one answer — is a **caller-side loop** built on
the same event stream. It is never the primitive, and it does not belong in this contract.

## Two observation faces, and they are not the same thing

**Lifecycle events** are for the observability surface: one event per position in the carrier's
life — never a chunk delta — carrying only what every engine can state:

``hive`` · ``engine`` · ``session_id`` · ``preset`` · ``event_name`` · ``description`` ·
``phase`` · ``payload``

``phase`` is the one vocabulary this layer owns; ``event_name`` stays the engine's, untouched.
``description`` is what a human reads; ``payload`` is the concrete data behind the event,
debug-oriented and droppable on a broadcast link. **An event is not an answer.**

**The output face** is the precise field a ghost needs: ``on_output(final_answer, event)`` — the
turn's final answer as plain text, paired with the event that carried it. Text is the promise;
the rest of the event is context. Conflating the two is how a lifecycle stream gets mistaken for
a return value.

The richer thing — a "turn", its tool calls, its reasoning, all structured — is exactly what we
avoid modelling: it would be the engine's shape in disguise. So the abstraction exposes thin
hooks and an implementation adds whatever thicker ones it likes beside them.

The same event protocol is read at two altitudes, which is why it carries ``hive`` and
``session_id`` even on a single session: a session's hook is already filtered to one carrier,
while the hive (next blueprint) hands the same events unfiltered.

## Why the hive lives in this file

``AgentSession`` and ``Hive`` are one layer — a carrier and the place carriers are kept — so they
share a module for now. ``hive.py`` still holds the memento face of the hive; when that layer is
handled, its class becomes ``MementoAgentHive`` and the base moves over. Nothing imports either
yet, so this costs no one anything today.

## What is deliberately absent, and where it went instead

- **No session topology.** "One conversation over several sessions", "a ref pointing at a
  historical session's coordinate" — those are MOSS concepts. An agent here has exactly one
  carrier and no coordinate of its own.
- **No ops, no commits.** Nothing parses a reply's first line.
- **No names.** Identity is the carrier's id. A stable name that outlives its carrier is what a
  branch buys, so it is not this layer's word to spend.
- **No anchors, coordinates or reboots.** They are all ways of pointing at a trajectory.
- **No fan-out.** An agent cannot spawn an agent. The hive spawns, and that cut is what keeps a
  team an explicit act instead of an accident.
- **No relation objects.** An agent carries its preset's **id**, not the preset; the caller joins
  through ``Hive.preset(id)``. A preset is a declaration on disk that a deployment can replace at
  any time, while a live agent keeps running the composition it was created with. Handing out the
  object would quietly re-bind a running agent to a template it never ran.

The rule that keeps this file honest: **it must not import the trajectory.** If a capability
needs a branch to exist, it does not live here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ghoshell_moss.core.blueprint.channel_builder import MutableChannel
from ghoshell_moss.message import Message

__all__ = [
    "AgentPhase",
    "AgentPreset",
    "AgentSession",
    "AgentSessionEvent",
    "AgentState",
    "EventHook",
    "Hive",
    "MessageMode",
    "Workspace",
]


# --------------------------------------------------------------------------- #
# Places and templates
# --------------------------------------------------------------------------- #


class Workspace(BaseModel):
    """A place an agent can be created inside.

    Two different facts wear this name and must not be conflated: a workspace is the engine's
    **grouping** of sessions by directory, while the **permission boundary** is the session's
    ``cwd`` — the sandbox's writable root. A session can move between workspaces; its ``cwd`` is
    what decides what it may touch.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(default="")
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
    display label and is often absent (measured on dsh 0.2.0-rc.2: four of the five shipped
    presets carry no name at all), so it can never be the identity.

    There is no capability tag list here on purpose. What a preset can do is its **composition**
    — the plugins and tools it mounts — which the engine states as readable content. A parallel
    tag vocabulary would have to be invented by us, could never be satisfied by an adapter, and
    would be a second source of truth for a fact the engine already owns.

    The patch surface ``configure`` accepts **is** here — but as an opening rather than as
    content. ``config_json_schema`` is a field a concrete implementation fills with its engine's
    schema; this file never states what is in it, because the moment it did, an adapter would owe
    a guess. Its pair lives on the state: schema on the template, current values on the instance.
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
            "implementation. The default is the empty schema — 'an object, no constraint stated' "
            "— which is honest: this layer knows nothing about it."
        ),
    )


# --------------------------------------------------------------------------- #
# State — read from the engine, never our own vocabulary
# --------------------------------------------------------------------------- #


class AgentState(BaseModel):
    """The cheap read: everything a driver needs to decide, without touching the engine.

    Two kinds of data live here: the structural facts fixed when the carrier opened (who it is,
    where it acts, which preset composed it), and ``config`` — the values ``configure`` has put in
    force, shaped by the preset's ``config_json_schema``.
    """

    model_config = ConfigDict(extra="allow")

    id: str
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
    config: dict[str, Any] = Field(
        default_factory=dict,
        description="the runtime configuration match the agent preset config json schema.",
    )


# --------------------------------------------------------------------------- #
# The event protocol — lifecycle observation, not a chunk stream
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
    session_id: str = Field(description="the carrier this event came from.")
    preset: str = Field(default="", description="the preset's id, as the carrier reports it.")
    event_name: str = Field(
        default="",
        description="the engine's own name for the event, passed through untouched — this layer "
        "invents no event vocabulary.",
    )
    description: str = Field(
        default="",
        description="optional description of the event",
    )
    phase: AgentPhase
    payload: dict[str, Any] = Field(
        default_factory=dict,
        description="the concrete agent event data",
    )


EventHook = Callable[[AgentSessionEvent], Awaitable[None] | None]


# --------------------------------------------------------------------------- #
# The agent — one carrier, one delegation
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

    The other half is an identity that outlives this carrier, and it belongs to the layer that
    owns a trajectory. This object dies with its session, by design: that is why it has no
    ``reboot``.
    """

    @property
    def id(self) -> str:
        """The carrier's identity, and therefore the agent's. Asking for another id is asking for
        another agent — this layer has no second identity to give."""
        return self.state().id

    @property
    def title(self) -> str:
        """A human-facing name. Display only: addressing never goes through it."""
        return self.state().title

    @property
    def preset(self) -> str:
        """The **id** of the preset this carrier was composed from — not the preset itself."""
        return self.state().preset

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

        Grouping and permission are two facts, and this is the weaker one: ``cwd`` decides what
        the agent may touch, a workspace only decides which list it shows up in.
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

        ``reason`` travels with the cancellation so the carrier — and whoever reads the trace
        later — can tell an intended stop from a failure. It is optional because "stop" is a
        complete instruction on its own.
        """

    @abstractmethod
    async def configure(self, patch: dict[str, Any]) -> str:
        """Patch the carrier's settings, applied at the next turn boundary.

        The patch must match ``AgentPreset.config_json_schema``. That field is the opening; this
        docstring states nothing about its content, because the shape is the engine's. Absent
        fields keep their current value — nothing is reset by omission. Returns a short
        description of what is now in force.
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
# The hive — the address space the sessions live in
# --------------------------------------------------------------------------- #


class Hive(ABC):
    """Where sessions are opened and held — the address space, the presets, the places.

    **The baseline rule: a non-memento hive holds runtime, in-memory sessions only.** Nothing it
    holds is promised to outlive the process. Coming back to a session is therefore a separate,
    explicit act, addressed by its id (``resume``) — never something a roster does implicitly on
    restart. A hive that keeps a trajectory makes a stronger promise, and that belongs above this
    layer.

    **Closing belongs to the holder.** A session does not close itself: an object that can end its
    own address is a process responsible for something it does not own. The hive closes what it
    holds, in a fixed order (see ``close`` and ``retire``).

    **Two altitudes of one event protocol.** A session's hooks push and are already filtered to
    one carrier; ``events()`` is the pull side over every carrier the hive holds. That asymmetry
    is exactly why ``AgentSessionEvent`` carries ``session_id`` even though a session's own hook
    never needs it.

    A concrete hive is the shell's governance surface for **one engine generation** (``DshHive``
    for dsh 0.2.0, say), not a runtime the shell wraps around an engine.
    """

    # -- identity --------------------------------------------------------------- #

    @property
    @abstractmethod
    def name(self) -> str:
        """The hive's name. Presets and sessions are addressed inside it, never globally."""

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """The engine family this hive drives, e.g. ``dsh``.

        Paired with ``version`` rather than packed into one string: the family and the version
        change for different reasons, and a consumer that routes on one should not have to split
        a string to read the other.
        """

    @property
    @abstractmethod
    def version(self) -> str:
        """The engine version this hive speaks, e.g. ``0.2.0-rc.2``."""

    # -- discovery: data, not handles -------------------------------------------- #

    @abstractmethod
    def workspaces(self) -> list[Workspace]:
        """The places a session can be opened inside — a read over the engine's own list, never a
        registry of ours."""

    @abstractmethod
    def presets(self) -> list[AgentPreset]:
        """The templates this hive can open a session from.

        Also a read: presets are declared by the deployment, not authored here. Hence no filter
        argument — with a handful of entries the caller reads the roster.
        """

    @abstractmethod
    def preset(self, ref: str) -> AgentPreset:
        """One preset by id. Raises when it does not exist — never ``None``."""

    @abstractmethod
    def sessions(self) -> list[AgentState]:
        """The roster as data: which sessions this hive holds right now, and how each one stands."""

    @abstractmethod
    def session(self, session_id: str) -> AgentSession | None:
        """The handle for a session this hive currently holds, or ``None``.

        Deliberately separate from ``sessions()``: a listing is a snapshot anybody may take, while
        a handle is a claim that something is drivable *now*.
        """

    # -- lifecycle --------------------------------------------------------------- #

    @abstractmethod
    async def open(
        self,
        *,
        preset: str = "",
        cwd: str = "",
        title: str = "",
    ) -> AgentSession:
        """Open one carrier and hand it back as a session.

        **There is no instruction argument.** Opening a session is not briefing it: the engine
        takes a preset, a directory and an id, and nothing else (measured on dsh 0.2.0:
        ``session/create`` is ``{workspaceId?, cwd?, sessionId?, agentPreset?}``). The first
        message is a ``send``.

        ``cwd`` is the permission boundary — **absolute**; empty means the engine's default. An
        empty ``preset`` means the deployment's default, which the engine decides.
        """

    @abstractmethod
    async def resume(self, session_id: str) -> AgentSession:
        """Take up a session this hive no longer holds, addressed by its id.

        The only way back in, and deliberately an explicit act: the baseline hive promises nothing
        about its sessions outliving the process, so anything "restored" automatically would be a
        guess dressed as a fact. What an id can still reach is the engine's answer; an id it
        cannot reach raises.
        """

    @abstractmethod
    async def retire(self, to: str, reason: str = "") -> None:
        """Close one held session and drop it — ``close``'s ordered teardown, for a single carrier."""

    @abstractmethod
    async def close(self) -> None:
        """Close every session this hive holds, then the hive itself.

        The order per session is the contract, because each step loses something if moved:

        1. stop the producer — cancel the turn in flight;
        2. emit the terminal event (``AgentPhase.closed``), so a surface can tell "ended" from
           "went quiet";
        3. wait for that dispatch to return, **then** dispose the hooks — reversed, the terminal
           event is lost;
        4. close the carrier;
        5. drop it from the roster last, so a listing taken mid-teardown shows something winding
           down rather than something that vanished.

        The terminal event is promised for an orderly shutdown only. A process that dies emits
        nothing, and that gap is the reader's watch — liveness and timeouts — not something this
        protocol can pretend to cover.
        """

    # -- driving ----------------------------------------------------------------- #

    @abstractmethod
    async def send(
        self,
        to: str,
        *messages: Message,
        mode: MessageMode = MessageMode.enqueue,
    ) -> None:
        """Send messages to a held session by id. An unknown id raises — it never drops silently."""

    @abstractmethod
    async def cancel(self, to: str, reason: str = "") -> None:
        """Stop the turn in flight on one session; the queue is kept."""

    # -- observation -------------------------------------------------------------- #

    @abstractmethod
    def events(self, since: int = 0) -> tuple[list[AgentSessionEvent], int]:
        """Read events across every carrier this hive holds — the pull side of the protocol.

        ``since`` is an opaque cursor; 0 starts from the earliest event the hive still keeps.
        Returns the page and the cursor to pass next time. A surface that connects late catches up
        here and then switches to push, so nothing has to have been listening from the start — and
        a surface that is closing can drain the tail before it goes away.
        """

    # -- presentation -------------------------------------------------------------- #

    @abstractmethod
    def as_channel(self) -> MutableChannel:
        """Render this hive as a channel — the shell's control surface for it.

        Here rather than inside a builder, because a channel *is* the wiring: if the hive did not
        produce it, every consumer would write its own, and the second one would describe the same
        roster in different words. An implementation may bind whatever extra detail its channel
        needs onto the same object; this face only promises the entry point.
        """
