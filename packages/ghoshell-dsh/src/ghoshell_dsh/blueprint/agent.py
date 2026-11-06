"""The agent layer — what a *driver* does with a session, above what dsh exposes.

`DshAgentSession` is dsh's vocabulary, faithfully. This layer is ours: the verbs we can already
deliver with a dsh session today, plus the ones the plugin will deliver, bundled onto the same
object so a caller does not have to know which verb is native and which is a patch.

Inheriting means the target is written down: anything that satisfies `AgentSession` also speaks
every dsh verb, so nothing here has to re-declare them or hedge about whether the low layer is
present.

Three things this layer adds:

- `run()` — send and wait. dsh's `prompt()` only submits; waiting is a loop we own, and every
  caller would otherwise write it differently (and get the timeout semantics wrong).
- `anchor()` — write down where the session is, in a form that can leave the process.
- `chat_off_record()` — the bypass call: one model turn against the session's live context, with
  nothing written back to it.

### Why the chat verb is named that way

The call is deliberately outside the session's record: it reads the context, produces an answer,
and leaves no turn behind. Three names were reachable — `chat` (short, but says nothing about the
side effect that matters most), `ask` (already means a specific query in this project), and this
one. "Off record" is the property a caller must not be able to miss: if you want the answer to
*join* the conversation, you call `run()`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .anchor import DshSessionAnchor
from .models.session_events import ContentBlock, TokenUsage
from .models.sessions import SessionSummary
from .session import DshAgentSession

__all__ = ["AgentSession", "ChatReply", "RunResult", "SessionSource"]


class RunResult(BaseModel):
    """One `run()` — a prompt in, the turn that answered it out.

    `events` is the raw material of the turn; `final_response` is the same thing folded for the
    common case. When a caller needs more than the last message it reads `events` and projects —
    it does not ask the session for a second opinion.
    """

    model_config = ConfigDict(extra="allow")

    session_id: str = Field(default="")
    final_response: str = Field(default="", description="the last assistant message's text of this turn.")
    finish_reason: str = Field(default="", description="how the turn ended; empty when it never did.")
    events: list[Any] = Field(default_factory=list, description="the turn's events, in order.")


class ChatReply(BaseModel):
    """One answer from `chat_off_record()`.

    Nothing about a turn: no events, no session position. The point of the verb is that the
    session did not move, so there is nothing to report about where it went. `usage` is here
    because an off-record call still costs money, and the caller is the one who has to know.
    """

    model_config = ConfigDict(extra="allow")

    text: str = Field(default="", description="what the model answered.")
    usage: TokenUsage = Field(default_factory=TokenUsage, description="what the call cost.")
    finish_reason: str = Field(default="", description="why the answer stopped.")


class AgentSession(DshAgentSession, ABC):
    """A dsh session with the driver layer on top of it."""

    # ---- the high-level verb -------------------------------------------- #

    @abstractmethod
    async def run(
        self,
        content: str | list[ContentBlock],
        *,
        timeout: float | None = None,
    ) -> RunResult:
        """Send, then wait for the turn that answers it to end.

        A bare string is accepted here — and only here. At this layer "a string means a text
        part" is our decision, stated once; the protocol layer below never guesses.

        Raises `TimeoutError` when `timeout` elapses: the turn keeps running then, it is only
        this call that stopped waiting.
        """

    # ---- anchors -------------------------------------------------------- #

    @abstractmethod
    async def anchor(
        self,
        *,
        seq: int | None = None,
        note: str | None = None,
    ) -> DshSessionAnchor:
        """Write down where the session is — the point the next turn would start from.

        `seq=None` means "now"; naming a seq anchors an earlier point instead, which is what a
        caller does when it only later realises the moment was worth keeping. `note` is the
        anchor's own memory: what was decided, why this point — without it an anchor locates a
        place but says nothing about it.
        """

    # ---- the bypass call ------------------------------------------------ #

    @abstractmethod
    async def chat_off_record(
        self,
        message: str | list[ContentBlock],
        *,
        anchor: DshSessionAnchor | None = None,
        readonly: bool | None = None,
        timeout: float | None = None,
    ) -> ChatReply:
        """One model turn against the session's context, with nothing written back.

        `anchor` says *where* in the session to think from; `None` means the current end. The
        answer never becomes a turn: the session's log, its queue and its `seq` are untouched,
        so this can be called while another turn is running, and calling it twice costs two
        calls but leaves the same session behind.

        This is the verb for "what would the model say here" — a preview, an evaluation, a
        judgement — as opposed to `run()`, which is "make it part of the conversation".

        `readonly` restates the permission for this one call: `None` inherits the session's
        state, `True` requires the request be assembled with nothing that could write. It can
        only tighten, never relax — a session that is read-only cannot be talked into writing by
        asking nicely on a single call.
        """


class SessionSource(ABC):
    """Where sessions come from — the minimum a launcher has to hand back.

    A launcher opens something; what a caller needs from that something is "give me a
    conversation". Everything richer — raw calls, event distribution, workspaces — belongs to
    `Connection`, which is a `SessionSource` with more on it.

    Keeping this small is the point: a transport that can open sessions but has no notion of a
    connection-wide event stream still satisfies a launcher's promise.
    """

    @abstractmethod
    async def list_sessions(self) -> list[SessionSummary]:
        """The sessions this dsh is currently holding."""

    @abstractmethod
    async def create_session(
        self,
        *,
        cwd: str | None = None,
        agent_preset: str | None = None,
        readonly: bool | None = None,
    ) -> AgentSession:
        """Open a new conversation."""

    @abstractmethod
    async def attach_session(self, session_id: str) -> AgentSession:
        """Take up an existing conversation by its id."""
