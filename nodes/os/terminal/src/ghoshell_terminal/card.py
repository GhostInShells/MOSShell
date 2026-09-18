"""Card — the unit the terminal's human surface is made of.

A card is opened the moment the model starts emitting a command, filled while
its text streams, and then sits on the human's screen waiting for a verdict.
The card, not the process, is the first-class object: a card exists before any
process does (a pending command has no process yet), and a rule proposal never
gets one at all.

``CardState`` is the whole state machine. ``streaming`` is the transient phase
while the model is still writing the command; the tail packet moves it to
``awaiting`` (needs a verdict) or straight to ``running`` (auto-approved).
``rejected`` and ``cancelled`` are deliberately distinct: the former means the
human disagreed and the command should not be re-issued, the latter means the
proposal was withdrawn and may be sent again.
"""

from __future__ import annotations

import time
from enum import Enum

from pydantic import BaseModel, Field

__all__ = [
    "Card",
    "CardState",
    "CardType",
    "Dialogue",
    "Interaction",
    "Thread",
    "TERMINAL_STATES",
]


class CardType(str, Enum):
    """What a card means — which reading its accept/deny carries."""

    COMMAND = "command"
    """A shell command proposal. Accept = run it."""

    RULE = "rule"
    """An auto-approval regex proposal. Accept = register it (auto mode only)."""


class CardState(str, Enum):
    STREAMING = "streaming"
    """The model is still writing this card's content."""

    AWAITING = "awaiting"
    """Written, waiting for the human's verdict."""

    RUNNING = "running"
    """Spawned, output flowing."""

    DONE = "done"
    """Exited with code 0."""

    ERROR = "error"
    """Exited with a non-zero code."""

    REJECTED = "rejected"
    """The human said no. Do not re-issue."""

    CANCELLED = "cancelled"
    """Withdrawn — the model's streaming was interrupted, or it cancelled the
    proposal itself. May be re-issued."""


TERMINAL_STATES: frozenset[CardState] = frozenset(
    {
        CardState.DONE,
        CardState.ERROR,
        CardState.REJECTED,
        CardState.CANCELLED,
    }
)


class Interaction(str, Enum):
    """The three things a human can do to a card that awaits a verdict."""

    ACCEPT = "accept"
    DENY = "deny"
    ASK = "ask"
    """Talk about it. Leaves the card pending — asking decides nothing."""


AWAITING_INTERACTIONS: list[str] = [
    Interaction.ACCEPT.value,
    Interaction.DENY.value,
    Interaction.ASK.value,
]


class Dialogue(BaseModel):
    """One line of the card's conversation, appended by either side."""

    author: str
    """``human`` or ``model``."""

    text: str
    at: float = Field(default_factory=time.time)


class Thread(BaseModel):
    """A named working context: where commands run and what it is for.

    Threads are the model's handles. Every command carries an explicit thread
    name — there is no hidden "current thread" state.
    """

    name: str
    cwd: str
    description: str = ""


class Card(BaseModel):
    id: int
    """Node-level monotonic id. Not the subprocess index — a card exists before
    any process is spawned, and a rule card never gets one."""

    type: CardType
    title: str = ""
    """The card's intent label — the human's ``desc`` for a command card, the
    rule's name for a rule card. This is what the card IS; ``thread`` is only
    the context it runs in."""

    description: str = ""
    content: str = ""
    """Command text (streamed in) or, for a rule card, the regex."""

    state: CardState = CardState.STREAMING
    thread: str = ""
    cwd: str = ""
    level: str = "info"
    """Signal level the model asked for on completion."""

    process_index: int | None = None
    exit_code: int | None = None

    created: float = Field(default_factory=time.time)
    updated: float = Field(default_factory=time.time)
    ended: float | None = None

    dialogue: list[Dialogue] = Field(default_factory=list)
    output_tail: list[str] = Field(default_factory=list)
    """Bounded in-memory tail, for ``read()`` and for resyncing a late client.
    The complete record lives in ``output_file``."""

    output_file: str | None = None
    output_chars: int = 0

    def interactions(self) -> list[str]:
        """What the human may do to this card right now."""
        return list(AWAITING_INTERACTIONS) if self.state is CardState.AWAITING else []

    @property
    def settled(self) -> bool:
        return self.state in TERMINAL_STATES

    def elapsed(self) -> float:
        end = self.ended if self.ended is not None else time.time()
        return round(end - self.created, 3)

    def view(self) -> dict:
        """The JSON form shipped to the browser."""
        return {**self.model_dump(mode="json"), "interactions": self.interactions()}
