"""MementoAgent — one delegation, carried by a memento branch.

This is the layer a Ghost faces when it hands work to another intelligence. It is not
another agent framework: it answers one question — **when a ghost delegates, what must
persist and what may disappear.**

## Three words, three boundaries

- **Agent** (``MementoAgent``): one delegation. Its identity is a memento branch: the ghost
  holds it, addresses it by name, and finds it again after a restart by that name.
- **Session** (``MementoAgentSession``): this delegation's living carrier — one engine
  session (dsh today). It dies, is replaced, and is rebuilt at will; the delegation does not
  move. **The agent is the long-lived one; the session is the short-lived one.**
- **Branch** (memento): the vehicle. Every trace of the delegation lives in it, and there is
  no second source of truth.

## The interaction is asynchronous, and that is not a detail

A ghost may hold dozens of delegations. If talking to an agent meant waiting for its turn,
the ghost would be doing one thing at a time — the swarm would be a list, not a swarm.

So the contract has exactly two halves:

- **sending**: ``prompt`` enqueues and returns a receipt. It never waits for a turn.
- **observing**: ``on_commit`` reports each finished step; ``state`` is the cheap read.

Synchronous waiting exists only where a caller genuinely needs one answer (a tool call, a
test). It is built on the same two halves; it is never the primitive.

## Where the runtime ends and the trajectory begins

The engine owns the turn: one ``prompt`` becomes a turn, a turn ends with a final answer.
The agent layer owns what that answer *means*: it parses the op, writes one commit, and the
branch records it. Everything the engine does in between — streaming, tool calls, retries —
is invisible here.

**The trajectory is the only constant.** A session can be thrown away and rebuilt from the
commit trail; nothing that exists only in memory is expected to survive. That is why there
is no ``close``: closing would leave the delegation pointing at nothing. There is
``reboot``, which starts a *new* session located by the last commit — so the pointer is
never empty.

## What this layer deliberately does not do

- **No fan-out.** The hive spawns; an agent cannot spawn. Hierarchical modelling is a
  product decision, and cutting the edge here is what keeps the tree explicit. If the engine
  grows subagents of its own, they are the engine's business and stay inside its session.
- **No reading tools bound into the running session.** A live agent does not get to read its
  own trajectory; the ghost (and the hive) read it. See the hive blueprint's ``Hive.read``.
- **No briefing format.** ``Mandate.instruction`` is a default rendering, not the shape.
  How a session is primed — declaration, proposition, folded history, absorbed references —
  is the manager's business, and it is replaceable there.

## Verbs lean on dsh

``prompt`` / ``cancel`` / ``reboot`` / ``configure`` mirror the engine's own vocabulary:
``prompt`` carries dsh's ``mode`` (``queue`` = the next turn, ``steer`` = the current one),
because inventing a second name for the same boundary would only add a place to drift.

## Not in scope, and must not be forgotten

- ``MementoAgentSession.export_anchor(note=...)`` — fold a span into a point and keep the
  note. Not built now; losing it from the map is not allowed.
- A raw, unfiltered read (every engine event). Debug only, never on this surface.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, Field

from ghoshell_moss.core.blueprint.mindflow import Priority
from ghoshell_moss.memento.abcd import (
    Branch,
    BranchAddr,
    Child,
    CommitView,
    Confluence,
)
from ghoshell_moss.message import Message

__all__ = [
    "Follow",
    "AgentOp",
    "Mandate",
    "AgentCommit",
    "AgentStatus",
    "AgentState",
    "AgentPreset",
    "CommitCallback",
    "MementoAgentSession",
    "MementoAgent",
]


# --------------------------------------------------------------------------- #
# The declaration — what the delegator hands over
# --------------------------------------------------------------------------- #


class Follow(str, Enum):
    """Who pushes the next step, once this one lands.

    The three values are exclusive, so there is no separate ``terminal`` flag: a terminal op
    is simply ``end``. ``goon=True`` next to ``terminal=True`` becomes unrepresentable.
    """

    wait = "wait"
    """Stop. The delegator says what happens next. (Default: the safest starting point.)"""

    go = "go"
    """Keep going on its own. While the delegator sleeps, this is the only thing moving."""

    end = "end"
    """This delegation is finished. Terminal ops (done / canceled / error) use it."""


class AgentOp(BaseModel):
    """One operation the model is allowed to reach for — a control word.

    An op is not a tool and not a phase: it is **the first line of every reply**. It therefore
    does three things at once — constrains the shape of the reply, decides who pushes next,
    and says how loudly the result reaches the delegator.

    The op table is written by the delegator (the ghost), because it is the delegator's
    intent. A preset may suggest a table; that is sugar, and the ghost may replace it whole.
    """

    op: str = Field(description="The control word itself, e.g. `implement`.")
    instruction: str = Field(description="When to use it, and what to write when you do.")
    follow: Follow = Field(default=Follow.wait, description="Who pushes the next step.")
    priority: Priority | None = Field(
        default=None,
        description=(
            "How loudly this step reaches the delegator: a mindflow signal tier, or None = "
            "record it in the trajectory and make no signal. Reuses the project's ladder "
            "rather than inventing a second one."
        ),
    )


class Mandate(BaseModel):
    """One delegation's terms — the proposition plus the operations allowed to serve it.

    Written at spawn, landing in the branch's first commit, read-only afterwards. **To change
    the proposition, fork a new branch**: "what this delegation is" must have exactly one
    answer for its whole life.

    ``instruction`` is the default rendering, not the only one: a manager may render the same
    mandate differently (another language, another level of detail) without touching the data.
    """

    proposition: str = Field(
        default="",
        description=(
            "The proposition: what the world looks like when this is done, where the boundary "
            "is, what counts as finished. Plain language."
        ),
    )
    ops: list[AgentOp] = Field(
        default_factory=list,
        description="The allowed operations. An empty table means: report only, do not act.",
    )

    def op(self, name: str) -> AgentOp | None:
        """Look up a control word; unknown returns None (callers fall back to wait + silence)."""
        for item in self.ops:
            if item.op == name:
                return item
        return None

    def opens(self) -> list[str]:
        return [item.op for item in self.ops]

    def instruction(self) -> str:
        """Render the declaration as the instruction a session is primed with.

        Deliberately plain: it says what the reply shape is and what each op means. Anything
        else the manager wants to say belongs to the manager's own briefing.
        """
        parts = [
            "# Delegation",
            "",
            "## Proposition",
            "",
            (self.proposition.strip() or "(no proposition stated)"),
            "",
            "## Protocol",
            "",
            "Every reply is exactly **one step**, written in this form:",
            "",
            "    <op>(<target>): <title>",
            "    <body>",
            "",
            "`<target>` is optional. `<title>` is one line. `<body>` says what this step",
            "produced: what you did, what you learned, what is still open. One step per",
            "reply — never narrate several steps at once, and never answer in prose without",
            "an op line.",
        ]
        if self.ops:
            parts += ["", "## Operations", ""]
            for item in self.ops:
                parts.append(f"- `{item.op}` — {item.instruction.strip()}")
                parts.append(f"  · after this step: {item.follow.value}")
        else:
            parts += ["", "No operation is defined yet: report only, and do not act."]
        return "\n".join(parts)


# --------------------------------------------------------------------------- #
# The trace — what the branch hands back
# --------------------------------------------------------------------------- #

_STEP_LINE = re.compile(
    r"^(?P<op>[A-Za-z][A-Za-z0-9_-]*)(?:\("
    r"(?P<target>[^)]*)\))?\s*[:：]\s*(?P<title>.*)$"
)


def _strip_fence(text: str) -> str:
    """Drop an overall markdown fence — the shell models like to wrap replies in."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


class AgentCommit(BaseModel):
    """One step, read out of a memento commit.

    Not a second storage: a semantic reading of ``CommitView``. The first line of the message
    is the op declaration; the rest is the body.

    A reply that does not parse **does not raise**: ``violation=True``, the op stays empty and
    the whole text falls into the body. A step that broke the protocol is still a step, and a
    trajectory that hides it is a trajectory that lies about what happened.
    """

    coord: str = Field(description="{branch_index}-{seq}: the address people cite.")
    seq: int = Field(description="In-branch commit ordinal.")
    op: str = Field(default="", description="Control word; empty = this step did not parse.")
    target: str = Field(default="", description="What the op acted on, optional.")
    title: str = Field(default="", description="The rest of the op line.")
    body: str = Field(default="", description="This step's body.")
    created: AwareDatetime = Field(description="When the commit was written (UTC).")
    broken: bool = Field(default=False, description="A broken note placeholder, not a protocol violation.")
    violation: bool = Field(default=False, description="No op line could be parsed.")
    recipients: list[BranchAddr] = Field(
        default_factory=list, description="Branches this step's message was addressed to."
    )
    confluences: list[Confluence] = Field(
        default_factory=list, description="References this step had received before it was written."
    )
    children: list[Child] = Field(
        default_factory=list, description="Branches created from this one before this step."
    )

    @property
    def headline(self) -> str:
        """``op(target): title`` on one line — folding, listings and logs all use it."""
        if not self.op:
            return self.title
        head = f"{self.op}({self.target})" if self.target else self.op
        return f"{head}: {self.title}" if self.title else head

    @classmethod
    def from_view(cls, view: CommitView) -> "AgentCommit":
        """Read one commit as one step. **This is the only upwards seam.**"""
        message = _strip_fence(view.message)
        first, _, rest = message.partition("\n")
        match = _STEP_LINE.match(first.strip())
        shared = dict(
            coord=view.coord,
            seq=view.seq,
            created=view.created,
            recipients=list(view.recipients),
            confluences=list(view.confluences),
            children=list(view.children),
        )
        if match is None:
            return cls(
                title=first.strip(),
                body=rest.strip("\n"),
                broken=view.is_broken,
                violation=bool(message),
                **shared,
            )
        return cls(
            op=match.group("op"),
            target=(match.group("target") or "").strip(),
            title=match.group("title").strip(),
            body=rest.strip("\n"),
            broken=view.is_broken,
            **shared,
        )


# --------------------------------------------------------------------------- #
# State — derived from two facts, never stored
# --------------------------------------------------------------------------- #


class AgentStatus(str, Enum):
    """Where this delegation stands, derived from two facts.

    The facts: does the branch still have a name pointer, and is there a live carrier. Words
    like "waiting for a decision" or "stuck" are not states — they are the op of some step.
    """

    running = "running"
    """A live carrier exists. A prompt now enters its next turn."""

    idle = "idle"
    """No live carrier. The last commit still locates where it stopped; reboot resumes there."""

    archived = "archived"
    """The name pointer is gone. The trace is on disk, but it is out of the active set."""


class AgentState(BaseModel):
    """The cheap read: everything a delegator needs to decide, without touching the engine."""

    name: str
    address: BranchAddr
    status: AgentStatus
    pending: int = Field(default=0, description="Prompts queued and not yet consumed.")
    last: str = Field(default="", description="Coordinate of the most recent commit; '' = empty branch.")
    session: str = Field(default="", description="Engine session id when running; '' otherwise.")


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #


class AgentPreset(BaseModel):
    """A template — a capability, not an identity. Chosen by ``capabilities``, not by name.

    A preset exposes one thing beside its description: the **schema of what can be configured
    at runtime**. There is no init-time config: an agent is created with the preset's defaults
    and patched afterwards through ``MementoAgent.configure``. That keeps creation cheap and
    keeps "what this agent is running with" a runtime fact rather than a constructor argument.
    """

    hive: str = Field(description="The hive this preset belongs to.")
    id: str = Field(description="Preset id inside the hive.")
    capabilities: list[str] = Field(
        default_factory=list,
        description="What it can do, e.g. `code` / `research` / `long-context` / `vision`. Match on these.",
    )
    description: str = Field(default="", description="One line, for reading details.")
    manual: str = Field(default="", description="How to use it well — the only text a delegator needs.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Readable defaults, e.g. model.")

    def config_schema(self) -> dict[str, Any]:
        """The JSON schema of the patch ``configure`` accepts.

        Static for the preset's lifetime, which is why it is safe to render into a command
        docstring. Current values are not here — they are state, and state belongs in a
        notice, not in a prompt that re-sends whenever it moves.
        """
        return _BASE_CONFIG_SCHEMA


_BASE_CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "A patch. Absent fields keep their current value; nothing is reset by omission.",
    "properties": {
        "llm": {
            "type": "object",
            "description": "Model route for subsequent turns.",
            "properties": {
                "provider": {"type": "string", "description": "Route key, as reported by the catalog."},
                "model": {"type": "string", "description": "Model id within that route."},
                "reasoningEffort": {
                    "type": "string",
                    "description": "Thinking depth; `off` disables thinking — it is an instruction, not 'unset'.",
                },
                "maxTokens": {"type": "integer", "description": "Hard ceiling for one answer."},
                "temperature": {"type": "number"},
                "stop": {"type": "array", "items": {"type": "string"}},
            },
        },
        "sandbox": {
            "type": "string",
            "enum": ["read-only", "workspace-write"],
            "description": "Where tools may act.",
        },
        "approval": {
            "type": "string",
            "enum": ["ask", "never"],
            "description": "Whether a human must approve a command.",
        },
    },
}


# --------------------------------------------------------------------------- #
# The carrier — short-lived
# --------------------------------------------------------------------------- #

TurnHook = Callable[[str], Awaitable[None] | None]
"""Called once per finished turn with the final answer text — the only thing this layer sees."""


class MementoAgentSession(ABC):
    """A delegation's living carrier: one engine session.

    It does three things: take prompts, end turns, and go away. **It does not write the
    trajectory** — it reports the final answer of each turn, and the agent layer above it
    decides what that answer becomes. That keeps this the only place that knows an engine
    exists.

    Its life is shorter than the delegation's: after it ends, the delegation is ``idle`` and
    ``reboot`` builds a new one from the commit trail.
    """

    @property
    @abstractmethod
    def session_id(self) -> str:
        """The engine's own identifier for this session — data, not an address."""

    @property
    @abstractmethod
    def alive(self) -> bool:
        """False once the engine session is gone; a prompt must fail then, not queue forever."""

    @abstractmethod
    async def prompt(
        self,
        message: Message,
        *,
        mode: Literal["queue", "steer"] = "queue",
    ) -> str:
        """Hand one message to the engine; returns a receipt and does not wait for the turn.

        ``queue`` takes the next turn; ``steer`` lands inside the one in flight. Mirrors the
        engine's own boundary instead of inventing a second name for it.
        """

    @abstractmethod
    async def cancel(self) -> None:
        """Stop the turn in flight. Work already committed is untouched."""

    @abstractmethod
    def on_turn(self, hook: TurnHook) -> Callable[[], None]:
        """Subscribe to finished turns; returns an unsubscribe.

        The hook receives the final answer text. It is called once per turn, after the
        engine has closed it, and it must not raise into the engine: the agent layer wraps it.
        """

    @abstractmethod
    async def configure(self, patch: dict[str, Any]) -> str:
        """Apply a config patch to this session; returns a short receipt.

        Patch semantics: absent field = keep. Takes effect at the next turn boundary, the same
        rule as model selection — a turn in flight is never rewritten underneath itself.
        """

    @abstractmethod
    async def close(self) -> None:
        """Tear the engine session down. Mechanical: called by ``reboot``, never by a caller.

        Nothing here is expected to survive; whatever must survive is already a commit.
        """


CommitCallback = Callable[[AgentCommit], Awaitable[None] | None]
"""Called once per finished step. Exceptions are the consumer's bug; they never touch the trace."""


# --------------------------------------------------------------------------- #
# The delegation — long-lived
# --------------------------------------------------------------------------- #


class MementoAgent(ABC):
    """One delegation. **Identity = one branch.**

    Everything durable about it lives in ``branch``, so a process restart, a machine change or
    an engine swap leaves the same agent. The carrier may die a hundred times; the trace does
    not move, and a rebuild is "fold the branch and start a carrier" — not "find the old
    session".
    """

    # -- identity (almost never changes) -------------------------------------- #

    @property
    @abstractmethod
    def hive(self) -> str:
        """The hive this delegation belongs to."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Address inside the hive — stable for the delegation's life."""

    @property
    @abstractmethod
    def address(self) -> BranchAddr:
        """The durable address: what goes into recipients, confluences and children."""

    @property
    @abstractmethod
    def branch(self) -> Branch:
        """The vehicle itself; reads, views and node spaces are reached from here."""

    @property
    @abstractmethod
    def mandate(self) -> Mandate:
        """The declaration written at spawn (first commit), read-only."""

    @property
    @abstractmethod
    def preset(self) -> AgentPreset:
        """The template that shaped it; a reboot picks the carrier again by this."""

    @property
    @abstractmethod
    def workspace(self) -> str:
        """The permission boundary: where this delegation may act. Set by the hive at spawn."""

    # -- state (cheap, frequent) ---------------------------------------------- #

    @abstractmethod
    def state(self) -> AgentState:
        """Two facts, derived; safe to call as often as a notice refreshes."""

    # -- sending (asynchronous by definition) ---------------------------------- #

    @abstractmethod
    async def prompt(
        self,
        message: Message,
        *,
        mode: Literal["queue", "steer"] = "queue",
    ) -> str:
        """Send one message; returns a receipt. **Never waits for a turn.**

        With no live carrier the message is held until ``reboot``; that is the whole reason
        the trajectory exists — a queued message that a crash drops is still visible in the
        rebuilt context as an absorbed reference.
        """

    @abstractmethod
    async def cancel(self) -> None:
        """Stop the turn in flight; the queue is kept."""

    @abstractmethod
    async def reboot(self) -> None:
        """Replace the carrier. **The only way a carrier gets replaced.**

        Reads the last commit, folds a briefing, starts a new engine session from it, and
        writes one commit recording that start (engine, session id, effective config). So the
        next reboot has a locator: **there is never a delegation without a pointer.**
        """

    @abstractmethod
    async def configure(self, patch: dict[str, Any]) -> str:
        """Patch the running config; absent fields keep their value.

        Applied at the next turn boundary. The schema is ``preset.config_schema()``. If a
        concrete implementation needs the values to outlive a crash, it stores them in the
        branch directory and reads them back on reboot — memento itself keeps no config.
        """

    # -- observing (the other half of the async contract) ---------------------- #

    @abstractmethod
    def on_commit(self, callback: CommitCallback) -> Callable[[], None]:
        """Subscribe to finished steps; returns an unsubscribe."""

    # -- lifecycle ------------------------------------------------------------- #

    @abstractmethod
    async def archive(self) -> None:
        """Archive this delegation and everything under it (memento does the subtree).

        The name pointer goes; the trace stays. Nothing in the live runtime is kept alive —
        a carrier still running against an archived branch is a bug at the caller's level.
        """


# --------------------------------------------------------------------------- #
# Not in scope for this iteration, and not to be lost
# --------------------------------------------------------------------------- #
#
# - ``MementoAgentSession.export_anchor(note=...)``: fold a span into a point and keep the
#   note next to it. A span folded into a point is lossy by construction — the left end is
#   gone — so the conversion must stay explicit and must refuse to guess.
# - Raw event reads (layer 3). Debug only; never on this surface.
