"""Case — one stateful task: a branch, a protocol, and agents that come and go.

This module is the outer layer of the agent stack.

- **Inner** (``agent.py`` / ``hive.py``): how a carrier is created, driven and observed.
- **Outer** (this module): what a task is once a carrier can be thrown away.

A case is **spawned and starts running**. It is not a session: the carrier is thrown away,
rebuilt, compacted; the case does not move. Its identity is one memento branch, and the
trajectory is the only truth.

## Two halves

- **``Case``** — the durable half: a branch directory, a prompt document, a control document.
  It survives everything.
- **``CaseAgent``** — the living half: one carrier and the loop that drives it. It can die at
  any moment and be rebuilt from the case.

The split is deliberate, and it is measurable: ``Case`` answers "what is this task", ``CaseAgent``
answers "who is working on it right now". A case without an agent is idle, not broken.

## Lifecycle: draft → bind → autopilot

- **``draft``** creates the branch and writes the two documents, then returns the path of the
  prompt document. Authoring the case is editing that markdown file — with the shared file editor,
  by a model or a human — not passing a large payload through a command.
- **``bind``** defines a **new** carrier: it reads the prompt document, composes the priming
  context, opens the carrier, and commits one **init** commit recording which session was bound
  and with which configuration. It is the explicit redefinition — the special path, used when the
  session is gone or a different composition is wanted.
- **``autopilot``** continues the case: with no live carrier it reattaches to the session the last
  commit names (a *reboot*), then dispatches the loop. Nothing is returned as an answer: every
  finished step arrives through ``on_commit``. The loop runs until it breaks.

The default start is therefore a resume, not a rebuild: the session keeps its real history, and
the carrier layer only marks the restart boundary before handing the floor back. A session whose
last commit is archived cannot be continued — its log stays readable, but no turn can run — and
that is where ``bind`` defines a new one.

## Two stops, and they are not the same thing

- **``pause``** — the issuer stops the case. An external act, recorded in the control document.
- **``break``** — the loop stops itself: a protocol violation, an exhausted budget, or an operator
  whose ``next`` is ``BREAK``. An internal fact, recorded in the step's operation.

Both end the run; who did it is what tells them apart.

## The op belongs to the executor

An op is the **first line of every reply** and it names the action that just ended. It is the
executor's vocabulary, never shared with the issuer: if both spoke the same words, the trace
could no longer separate "what I asked for" from "what it did".

Two consequences:

- **Not an action, not an op.** A budget notice, a compaction, a pause — none of them ends an
  action, so none of them is a word. They live in the control document or arrive through
  ``inject``.
- **Derivable, not reported.** Progress, remaining budget and "did the re-orientation land" are
  computed from the trace; asking a model to report them only adds a place to be wrong.

## The branch directory

| path | face | who writes |
|---|---|---|
| ``CASE.md`` | **prompt** — frontmatter (name, description, protocol, owner, created) plus the brief as body | written at draft, then edited by its author |
| ``case.meta.json`` | **control** — state, owner, reason, address, timestamps | the manager, in the issuer's name |
| ``tmp/`` | the durable mirror of the agent's scratch directory | the manager, at every commit |

## What lands in the trajectory

| commit | metatype | metadata |
|---|---|---|
| one finished step | ``""`` | ``operation`` (the parsed op) and ``usage`` (tokens of the turn) |
| a carrier was bound | ``init`` | ``session_ref``, ``preset``, ``workspace``, ``tmp_dir`` |
| a run started | ``run.start`` | ``run`` — started, budget |
| a run ended | ``run.end`` | ``run`` — started, ended, budget, stop |

Step commits carry an operation; system commits do not. A reader filters by ``metatype`` and
never tries to parse a system commit as a step.

## Budgets are runs

A budget is **one run's grant**, not a property of the case. It is written into the commit
metadata when the run starts, and each step records what its turn consumed. What is left is
therefore derived: the grant minus the usage of the steps since ``run.start``. Nothing is
counted twice and nothing is stored in a second place.

Context compaction is not here: the engine's own policy (pressure / overflow) and its
summarizer own it. A case layer that compacted by itself would be a second, worse summarizer.

## The priming chain

A new carrier is primed in this order, before its first prompt:

1. the **preset's** own prompt (who it is) — the carrier's, not ours;
2. the **protocol introduction** — what an op line is, one step per reply;
3. the **brief** — the task as its author wrote it;
4. the **op table** — which words this case may speak, and what follows each;
5. the **file manifest** — what is already in the scratch directory.

2–4 come from ``CASE.md``; 5 tells a reborn carrier that its files are still there.

## Boundaries

- **No fan-out.** The hive holds cases; a case does not hold cases.
- **The live agent does not read its own trajectory.** The manager reads and chats; a running
  carrier only sees what was injected.
- **No second config store.** Runtime configuration belongs to the carrier; the init commit
  records what it was, not what it should be.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import AwareDatetime, BaseModel, Field

from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.memento.abcd import (
    Branch,
    BranchAddr,
    CommitRef,
    CommitView,
    Memento,
)

from .agent import AgentSession, WorkspaceName
from .hive import Hive

__all__ = [
    "AutopilotRequest",
    "BindRequest",
    "Budget",
    "CASE_NAME_PATTERN",
    "Case",
    "CaseAgent",
    "CaseAgentHive",
    "CaseAgentManager",
    "CaseBrief",
    "CaseCommit",
    "CaseDefinition",
    "CaseMeta",
    "CaseProtocol",
    "CaseState",
    "CommitCallback",
    "Level",
    "Next",
    "Operation",
    "Operator",
    "ProtocolTemplate",
    "Run",
    "Usage",
]


CASE_NAME_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
"""Case names become branch names and directory names: lowercase letters, digits, `-`, `_`."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# The protocol — the words a case may speak
# --------------------------------------------------------------------------- #


class Next(str, Enum):
    """What the driving loop does with a step after it has been read.

    Two values, not three: an ended case is finished by a word whose ``next`` is ``BREAK``, and
    the case's own state says whether it was resolved or cancelled. A third "end" value would be
    a second way to say the same thing.
    """

    CONTINUE = "continue"
    """The loop drives the next turn without waiting for anyone."""

    BREAK = "break"
    """The loop stops and waits for the issuer. A violation always breaks: an unreadable step is
    not a step, and continuing past it would build a trajectory on top of a lie."""


class Level(str, Enum):
    """How much attention a step deserves from the issuer.

    This is the case author's intent, not a runtime mechanism: the channel maps it onto the
    project's signal ladder when it renders the surface.
    """

    BACKGROUND = "background"
    """Nobody needs to be woken. Read on inspection."""

    INFO = "info"
    """Ordinary progress; a surface may show it, nobody is interrupted."""

    NOTICE = "notice"
    """The issuer should see this soon — a checkpoint, a change of plan."""

    WARNING = "warning"
    """The issuer is needed: something is blocked, failing, or asking for a decision."""


class Operator(BaseModel):
    """One word the case may speak, and everything its use implies.

    Words are declared by whoever authors the case. A word does three things at once: it constrains
    the shape of the reply, it decides what the loop does next, and it says how loudly the step
    should reach the issuer.
    """

    name: str = Field(
        description="The word itself, written as the first token of a reply, e.g. `plan`."
    )
    instruction: str = Field(
        description=(
            "When to use this word, and what the body must say when it is used. Written for the "
            "executor: plain, imperative, one paragraph at most."
        )
    )
    next: Next = Field(
        default=Next.CONTINUE,
        description="What the driving loop does after a step spoken with this word.",
    )
    level: Level | None = Field(
        default=None,
        description=(
            "How loudly this step should reach the issuer. None means the step is recorded and "
            "surfaced on inspection only — no signal is raised."
        ),
    )


class CaseProtocol(BaseModel):
    """A case's interaction language: a reusable introduction plus its words."""

    introduction: str = Field(
        default="",
        description=(
            "The text every case built from this protocol receives first: what an op line is, one "
            "step per reply, never prose without an op. It is the same for every case that uses "
            "the protocol, which is why it is not repeated per word."
        ),
    )
    ops: list[Operator] = Field(
        default_factory=list,
        description="The words this case may speak. An empty table means: report only, do not act.",
    )

    def op(self, word: str) -> Operator | None:
        """Look up a word; unknown returns None (callers fall back to breaking the loop)."""
        for item in self.ops:
            if item.name == word:
                return item
        return None

    def words(self) -> list[str]:
        """The words this protocol opened, in declaration order."""
        return [item.name for item in self.ops]

    def instruction(self) -> str:
        """Render the protocol as the priming text: introduction, then the word table."""
        parts = [self.introduction.strip()] if self.introduction.strip() else []
        if self.ops:
            parts += ["## Operations", ""]
            for item in self.ops:
                parts.append(f"- `{item.name}` — {item.instruction.strip()}")
                parts.append(f"  · after this step: {item.next.value}")
        else:
            parts += ["No operation is defined yet: report only, and do not act."]
        return "\n".join(parts)


class ProtocolTemplate(CaseProtocol):
    """A named protocol — the reusable half of a case definition.

    Templates are stored by the hive so a case can be created by naming one instead of carrying
    a full table in the command payload.
    """

    name: str = Field(description="Stable template name; the case definition references it by this.")
    description: str = Field(
        description="One line on when this protocol fits, for choosing between templates."
    )


# --------------------------------------------------------------------------- #
# Runs and budgets — written into the trajectory, never into a second store
# --------------------------------------------------------------------------- #


class Budget(BaseModel):
    """One run's grant. Not a property of the case: a case may run many times.

    A budget is a ceiling, not an estimate. Both fields are optional; a missing field means that
    dimension is unbounded, which is a legitimate choice for a supervised parallel run.
    """

    seconds: float | None = Field(
        default=None,
        description="Wall-clock ceiling for this run, in seconds. The clock starts when the run starts.",
    )
    tokens: int | None = Field(
        default=None,
        description=(
            "Token ceiling for this run: the sum of the steps' engine-reported usage since the run "
            "started."
        ),
    )


class Run(BaseModel):
    """One run's record — stored in commit metadata, never in the control document.

    A run is a span of the trajectory, so the trajectory is where it belongs: ``run.start`` opens
    it, ``run.end`` closes it, and everything between those two commits is its content.
    """

    started: AwareDatetime = Field(description="When the loop was dispatched.")
    ended: AwareDatetime | None = Field(
        default=None, description="When the loop stopped; None on the start record."
    )
    budget: Budget = Field(default_factory=Budget, description="The grant this run was given.")
    stop: str = Field(
        default="",
        description=(
            "Why the loop stopped: `paused`, `exhausted`, `violation`, `ended`, a `break` word "
            "from the protocol, or `interrupted` when the process died before the loop could "
            "close the run. Empty while the run is open."
        ),
    )


class Usage(BaseModel):
    """What one finished turn consumed — recorded on that step's commit."""

    tokens: int = Field(
        default=0,
        description="Total tokens the engine reported for this turn (input plus output).",
    )


# --------------------------------------------------------------------------- #
# The documents — prompt face and control face
# --------------------------------------------------------------------------- #


class CaseState(str, Enum):
    """Where a case stands, as its issuer declares it.

    This is not the runtime state: whether a carrier is running right now is ``CaseAgent``'s
    business, and a case is normally open while nothing is running.
    """

    draft = "draft"
    """The documents exist; the case has not been handed to an agent yet."""

    open = "open"
    """The case is live: it may be running now, or waiting for its next run."""

    resolved = "resolved"
    """The goal was reached. The trajectory is kept and the case may be reopened by its issuer."""

    cancelled = "cancelled"
    """The case was abandoned. Its branch is archived and its name stops resolving."""


class CaseMeta(BaseModel):
    """The control face of a case — written by the manager, always in the issuer's name.

    Nothing on this document describes work; it describes who owns the case and where it stands.
    Every action that changes it is attributable to ``owner``, which is the only reason the
    control face can be trusted at all.
    """

    state: CaseState = Field(description="The lifecycle position declared by the issuer.")
    owner: str = Field(
        description=(
            "The issuer's identity — the one in whose name every control-side action is taken. "
            "A case with no owner cannot be paused, closed or resolved."
        )
    )
    reason: str = Field(
        default="",
        description=(
            "Why the state last changed; free text, written by the owner. A pause is not a state "
            "change and does not write here — it ends a run and reaches the carrier itself."
        ),
    )
    address: BranchAddr = Field(description="The branch this case writes into.")
    created: AwareDatetime = Field(description="When the branch was created.")
    updated: AwareDatetime = Field(description="When the control face last changed.")


class CaseBrief(BaseModel):
    """The prompt face of a case, as its document reads.

    Frontmatter plus body of ``CASE.md``. The protocol here is **resolved**: the case carries the
    words it was created with, so changing a template later cannot rewrite an old case.
    """

    name: str = Field(description="The case's name; also the branch's name.")
    description: str = Field(default="", description="One line on what this case is.")
    content: str = Field(
        default="",
        description=(
            "The brief itself, verbatim from the document's body: the task, its acceptance, its "
            "boundary, and anything the agent must know before it starts."
        ),
    )
    protocol: CaseProtocol = Field(description="The protocol this case speaks.")


class CaseDefinition(BaseModel):
    """What it takes to draft a case — the input schema of the draft command.

    ``protocol`` may name a stored template or carry a full table; the draft resolves a name into
    the protocol and writes the resolved copy into the document.
    """

    name: str = Field(
        pattern=CASE_NAME_PATTERN,
        description=(
            "The case's name and, at the same time, its branch name: lowercase letters, digits, "
            "`-` and `_`, at most 64 characters. Must be unique in the hive; a taken name is "
            "refused with a NameError rather than rebound."
        ),
    )
    description: str = Field(default="", description="One line on what this case is, for listings.")
    content: str = Field(
        default="",
        description=(
            "The brief: what the world looks like when this is done, how it will be judged, and "
            "what the agent must not touch. Written in plain language. This is only the seed of "
            "`CASE.md` — the case is authored by editing that file afterwards."
        ),
    )
    protocol: CaseProtocol | str = Field(
        description=(
            "The words this case may speak: a ``ProtocolTemplate`` name, or the full table when "
            "the case needs a protocol of its own."
        )
    )
    owner: str = Field(
        description=(
            "The issuer's identity, written as `ghost/<name>`. Every control-side action on this "
            "case is taken in this name, including pause and close. The hive fills it in when the "
            "command comes from a ghost."
        )
    )


# --------------------------------------------------------------------------- #
# The response — one step, read out of one commit
# --------------------------------------------------------------------------- #


class Operation(BaseModel):
    """The parsed op line of one step, as it is stored in the commit metadata.

    This is the only place a step's control meaning is written: the commit's message keeps the
    raw text, the metadata keeps the reading. A reader never re-parses prose to find out what a
    step did.
    """

    op: str = Field(default="", description="The word; empty when the line did not parse.")
    target: str = Field(default="", description="What the word acted on; empty when there is none.")
    title: str = Field(default="", description="The rest of the op line.")
    next: Next = Field(
        default=Next.CONTINUE,
        description="What the loop does next. A violation is always `break`.",
    )
    violation: bool = Field(
        default=False,
        description=(
            "True when no op line could be read. The step is still recorded — a broken step is a "
            "step, and a trajectory that hides it would be lying."
        ),
    )

    @property
    def headline(self) -> str:
        """``op(target): title`` on one line — listings, folding and logs all use it."""
        if not self.op:
            return self.title
        head = f"{self.op}({self.target})" if self.target else self.op
        return f"{head}: {self.title}" if self.title else head

    @classmethod
    def from_commit_ref(cls, ref: CommitRef) -> "Operation | None":
        """Read the operation a commit carries; ``None`` for a system commit."""
        data = ref.metadata.get("operation")
        if not isinstance(data, dict):
            return None
        return cls(**data)


SESSION_REF = TypeVar("SESSION_REF")


class CaseCommit(Generic[SESSION_REF], ABC):
    """One commit in a case's trajectory — a step, or a system record.

    The same type covers both because they live in the same stream and readers must be able to
    walk it linearly. ``operation`` tells them apart: a step has one, a system record does not.
    """

    @property
    @abstractmethod
    def view(self) -> CommitView:
        """The memento view this commit was read from."""

    @property
    @abstractmethod
    def operation(self) -> Operation | None:
        """The parsed step, or ``None`` when this commit is a system record."""

    @property
    @abstractmethod
    def session_ref(self) -> SESSION_REF | None:
        """The carrier that produced this commit; ``None`` for records written by the manager."""

    @property
    def metatype(self) -> str:
        """The commit's kind: empty for a step, otherwise ``init`` / ``run.start`` / ``run.end``."""
        return self.view.ref.metatype

    @property
    def message(self) -> str:
        """The raw message, exactly as it was committed."""
        return self.view.message

    @property
    def coord(self) -> str:
        """The address people cite, e.g. ``1-42``."""
        return self.view.coord

    @property
    def usage(self) -> Usage:
        """What this commit's turn consumed; zero for system records."""
        data = self.view.ref.metadata.get("usage")
        return Usage(**data) if isinstance(data, dict) else Usage()

    @property
    def headline(self) -> str:
        """One line for listings: the step's op line, or the system record's kind."""
        operation = self.operation
        if operation is not None:
            return operation.headline
        return self.metatype or self.view.title


CommitCallback = Callable[[CaseCommit], Awaitable[None] | None]
"""Called once per commit. Exceptions are the consumer's bug; they never touch the trace."""


# --------------------------------------------------------------------------- #
# The durable half
# --------------------------------------------------------------------------- #


class Case(ABC):
    """One case's durable face: its documents, its metadata, and its trajectory.

    Holding a ``Case`` never implies that anything is running. Reading it never touches a carrier,
    which is why a week-old case is as readable as a fresh one.
    """

    @property
    @abstractmethod
    def directory(self) -> Path:
        """The branch directory: ``CASE.md``, ``case.meta.json`` and ``tmp/`` live here."""

    @property
    @abstractmethod
    def document(self) -> Path:
        """The prompt document — ``<directory>/CASE.md`` — the file its author edits."""

    @property
    @abstractmethod
    def tmp(self) -> Path:
        """The durable mirror of the agent's scratch directory."""

    @property
    @abstractmethod
    def brief(self) -> CaseBrief:
        """The prompt face, parsed from ``CASE.md``."""

    @property
    @abstractmethod
    def meta(self) -> CaseMeta:
        """The control face, parsed from ``case.meta.json``."""

    @property
    @abstractmethod
    def branch(self) -> Branch:
        """The memento branch this case writes into."""

    @abstractmethod
    def commits(self, limit: int = 50) -> list[CaseCommit]:
        """The most recent commits, newest last — steps and system records alike."""


# --------------------------------------------------------------------------- #
# The living half
# --------------------------------------------------------------------------- #


class CaseAgent(Generic[SESSION_REF], ABC):
    """One case's living face: a carrier, and the loop that drives it.

    It is not an ``AgentSession``: a carrier ends, a case does not. It can be created, destroyed
    and rebuilt from the case at any moment.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The case name this agent serves."""

    @property
    @abstractmethod
    def case(self) -> Case:
        """The durable half this agent is working on."""

    @property
    @abstractmethod
    def session(self) -> AgentSession | None:
        """The carrier right now, or ``None`` when none is attached."""

    @abstractmethod
    def is_running(self) -> bool:
        """Whether the loop is driving right now — the runtime fact, not the case's state.

        It becomes false the moment the loop stops, by either stop: an exhausted budget, a break
        step or an external pause.
        """

    @abstractmethod
    def on_commit(self, callback: CommitCallback) -> Callable[[], None]:
        """Subscribe to this case's commits; returns the disposer.

        This is the only result face: ``autopilot`` dispatches and returns, and everything that
        happens afterwards arrives here.
        """


# --------------------------------------------------------------------------- #
# The collection
# --------------------------------------------------------------------------- #


class BindRequest(BaseModel):
    """What it takes to define a case's carrier — the input schema of the bind command.

    Every field except the case name is optional: a bind with nothing filled in reuses the form
    the last init commit recorded (workspace, preset, config, scratch) but still creates a **new**
    session. That is the point of the verb: it is the explicit redefinition. Continuing a case
    after a restart does not call it — ``autopilot`` reattaches to the session the trace names.
    """

    case_name: str = Field(description="The case to bind. It must already exist.")
    workspace: WorkspaceName | None = Field(
        default=None,
        description=(
            "The place the agent may act in — an id or a title from the hive's workspace list, "
            "never a path. None = reuse the last init commit's workspace."
        ),
    )
    preset: str = Field(
        default="", description="The composition of the carrier. Empty = reuse, or the hive default."
    )
    readonly: bool = Field(
        default=False,
        description=(
            "Creation-time permission: the carrier may read but not write. Not a later patch — the "
            "invariant is that no turn runs under a permission nobody asked for."
        ),
    )
    tmp_dir: str = Field(
        default="",
        description=(
            "Where the agent's scratch directory is created inside the workspace. Empty = a "
            "directory derived from the case name."
        ),
    )
    config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Runtime configuration patch applied at creation (model, sandbox, approval). None = "
            "reuse the last init commit's configuration."
        ),
    )


class AutopilotRequest(BaseModel):
    """What it takes to dispatch a run — the input schema of the autopilot command."""

    case_name: str = Field(
        description=(
            "The case to drive. It must already exist; a bound carrier is not required — the loop "
            "reattaches to the session the last commit names, and fails only when that session is "
            "archived or gone."
        )
    )
    budget: Budget = Field(
        default_factory=Budget,
        description=(
            "This run's grant. A new grant is written for every dispatch; nothing carries over "
            "from the previous run."
        ),
    )


class CaseAgentManager(Generic[SESSION_REF], ABC):
    """The cases one hive holds, and the verbs that move them.

    **The roster is the store**: every branch under the hive's memento is a case, so a restart
    reattaches nothing — the cases are already there, and ``autopilot`` reattaches a carrier to
    the session the trace names. ``bind`` defines a *new* carrier and is the special act. No
    in-memory ledger is kept, because a ledger and a trajectory are two homes for one fact.
    """

    @property
    @abstractmethod
    def memento(self) -> Memento:
        """The store the cases' branches live in — the hive's own."""

    @abstractmethod
    def list_cases(self, state: CaseState | None = None) -> list[Case]:
        """Every case in the store, optionally filtered by its declared state."""

    @abstractmethod
    def get(self, case_name: str) -> Case:
        """One case by name.

        :raise NameError: no branch carries that name.
        """

    @abstractmethod
    async def protocols(self) -> list[ProtocolTemplate]:
        """The protocols this hive can offer, for choosing one when drafting a case."""

    @abstractmethod
    async def save_protocol(self, template: ProtocolTemplate) -> None:
        """Store a protocol template under its name, replacing any previous one with that name."""

    @abstractmethod
    async def draft(self, definition: CaseDefinition) -> Case:
        """Create a case: the branch, ``CASE.md``, and the control document; return the case.

        The returned case points at its document (``Case.document``), and that is the point of the
        verb: the case is authored by editing ``CASE.md`` with the shared file editor, so writing
        a case costs one command and a document, not a large structured payload. ``bind`` reads
        the document back.

        :raise NameError: a case of that name already exists — a name is never rebound.
        """

    @abstractmethod
    async def bind(self, request: BindRequest) -> CaseAgent[SESSION_REF]:
        """Define a case's carrier: read the documents, prime it, record an init commit.

        This is the **redefinition** path. The priming chain is: the preset's own prompt, the
        protocol introduction, the case brief, the op table, and the file manifest of the scratch
        directory — everything a session that knows nothing needs to become the case's carrier.

        The init commit carries the carrier's identity and configuration. A later bind with no
        configuration reuses that form but still opens a new session; continuing after a restart
        is ``autopilot``'s reattach, not a bind.

        Binding a case that already has a carrier attaches a second one only if the implementation
        says so; the contract requires the previous carrier to be stopped first, never left to
        compete for the same branch.
        """

    @abstractmethod
    async def autopilot(self, request: AutopilotRequest) -> CaseAgent[SESSION_REF]:
        """Continue the case: reattach a carrier if needed, then dispatch the loop.

        With no live carrier, the implementation reattaches to the session the last step (or the
        init) names — a reboot, not a bind: the session is resumed with its real history and the
        carrier marks the restart boundary before the next turn. It fails only when that session
        is archived or gone; then a new carrier is the caller's explicit act (``bind``).

        A run grant is written as a ``run.start`` commit before the first turn, and closed with a
        ``run.end`` commit when the loop stops. The loop stops when it is paused, when the budget
        is exhausted, when a step violates the protocol, or when a step's ``next`` says break.

        :raise RuntimeError: the case was never bound, or its session can no longer be resumed.
        """

    @abstractmethod
    async def pause(self, case_name: str, reason: str = "") -> None:
        """Stop the loop from the outside — the external half of the two-stop rule.

        ``reason`` reaches the carrier, not the control document: the in-flight turn is cancelled
        and the reason is injected as a system reminder, so the agent knows why it was stopped and
        can account for it on its next step. The run closes with a ``run.end`` record whose
        ``stop`` is ``paused`` and ``is_running`` becomes false. The case stays open and the
        carrier stays attached.
        """

    @abstractmethod
    async def resolve(self, case_name: str, reason: str = "") -> None:
        """Declare the case finished. The trajectory is kept; the name keeps resolving."""

    @abstractmethod
    async def close(self, case_name: str, reason: str = "") -> None:
        """Abandon the case: stop its carrier, archive its branch, drop the name.

        Nothing the carrier emits after this call reaches a caller. The trace stays on disk for
        whoever still cites a coordinate.
        """

    @abstractmethod
    def read(
        self, case_name: str, commit_coord: str = "", limit: int = 10
    ) -> list[CaseCommit]:
        """Read steps as steps: the latest ``limit``, or the one at ``commit_coord``.

        Reading is not driving: it never touches a carrier and works on idle or cancelled cases
        just as well.
        """

    @abstractmethod
    async def chat(self, case_name: str, commit_coord: str, question: str) -> str:
        """Ask a question of one step, with the conversation that had happened by then.

        The other half of "the trace is the interface": to find out why a step was taken, ask
        that step instead of reconstructing it from prose. The exchange is not written back.

        It is async because the answer comes from a model call, not from the store.
        """

    def as_channel(self) -> Channel | None:
        """Render this collection as its own channel — the control surface for cases.

        ``None`` is legal: a collection without a channel is one the issuer cannot drive.
        """
        return None


# --------------------------------------------------------------------------- #
# The place
# --------------------------------------------------------------------------- #


class CaseAgentHive(Generic[SESSION_REF], Hive):
    """A hive that also keeps a trajectory — and the cases written into it."""

    @property
    @abstractmethod
    def memento(self) -> Memento:
        """The store this hive's cases write their trajectory into.

        The hive's own store, one per hive: a case belongs to the place that made it, not to
        whoever is driving the place.
        """

    @property
    @abstractmethod
    def cases(self) -> CaseAgentManager[SESSION_REF]:
        """The cases this place holds."""
