"""One session's state, read at one moment.

Session state used to be a pile of getters: `cwd()`, `agent_preset()`, `readonly()`,
`token_usage()`, `workspace()`, `routable()`, `model_selection()`. Each was individually
reasonable; together they were a trap. A driver that needs to decide what to do next has to
ask six questions in a row, and the session can change between two of them — so it can end up
acting on a picture where the model is from before a switch and the read-only flag from after.

One snapshot answers the question that was actually being asked: *what is this session now?*
It also compresses the contract, which matters more than it sounds: every getter is a promise
some implementation has to keep, and six promises to report roughly the same bytes is six
chances to disagree.

`instruction()` stays outside it: it is a large blob read rarely, and pulling it into every
snapshot would make the common read expensive.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from .nouns import WorkspaceView
from .session_events import TokenUsage
from .sessions import ModelSelection

__all__ = ["SessionState"]


class SessionState(BaseModel):
    """What one session is, at one moment."""

    model_config = ConfigDict(extra="allow")

    session_id: str = Field(default="")

    # ---- liveness: two different questions, both worth asking ---- #
    running: bool = Field(default=False, description="the conversation still exists.")
    busy: bool = Field(default=False, description="a turn is in flight right now.")

    # ---- permissions: two knobs, because one bit cannot say the useful thing ---- #
    sandbox: str | None = Field(
        default=None,
        description="where this session's tools may act: `read-only` or `workspace-write`. "
                    "`danger-full-access` is deliberately not reachable from us — a ghost may not "
                    "grant itself or its children what only the human can grant.",
    )
    sandbox_override: str | None = Field(
        default=None, description="the mode this session itself switched to; None = inherited default."
    )
    approval: str | None = Field(
        default=None, description="`ask` or `never`: whether a tool call needs a human nod."
    )

    # ---- where ---- #
    workspace: WorkspaceView | None = Field(
        default=None,
        description="the dsh workspace this session belongs to — immutable for the session's life.",
    )
    cwd: str | None = Field(default=None, description="the directory the agent runs in.")

    # ---- who is answering ---- #
    model: ModelSelection | None = Field(default=None, description="the model selected for this session.")
    agent_preset: str | None = Field(default=None, description="the preset that shaped the agent.")

    # ---- what it has cost ---- #
    token_usage: TokenUsage = Field(default_factory=TokenUsage, description="spend so far, as far as known.")

    # ---- can it be reached ---- #
    routable: bool = Field(default=False, description="whether the session can currently be routed to.")
