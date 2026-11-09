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
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .anchor import DshSessionAnchor
from .models.session_events import ContentBlock, TokenUsage
from .models.sessions import SessionSummary
from .session import DshAgentSession

__all__ = [
    "AdvancedDshAgentSession",
    "ChatReply",
    "ChatToolCall",
    "ForkResult",
    "InjectResult",
    "RunResult",
    "SessionSource",
]


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


class ChatToolCall(BaseModel):
    """One tool call the model asked for, as it asked for it.

    A reply that carries these has not answered in prose — it has asked to run something. The
    caller decides what that means: an off-record caller usually just wants the arguments (that
    is how a schema-shaped answer arrives), a driver would go on to execute the call.
    """

    model_config = ConfigDict(extra="allow")

    id: str = Field(default="", description="the provider's call id, for pairing with a result.")
    name: str = Field(default="", description="which tool.")
    arguments: str = Field(default="", description="the raw arguments string, as sent — not parsed.")


class ChatReply(BaseModel):
    """One answer from `chat_off_record()`.

    Nothing about a turn: no events, no session position. The point of the verb is that the
    session did not move, so there is nothing to report about where it went. `usage` is here
    because an off-record call still costs money, and the caller is the one who has to know.
    """

    model_config = ConfigDict(extra="allow")

    text: str = Field(default="", description="what the model answered in prose; empty when it answered with a tool call.")
    tool_calls: list[ChatToolCall] = Field(
        default_factory=list, description="what it asked to run instead of answering."
    )
    usage: TokenUsage = Field(default_factory=TokenUsage, description="what the call cost.")
    finish_reason: str = Field(default="", description="why the answer stopped.")


class InjectResult(BaseModel):
    """One `inject()` — what went in, and whether a step actually took it.

    `woke` is the fact a caller cannot infer from the outcome: was the agent already running when
    the message was handed over? `inject` never wakes the driver, so on an idle session the
    message is *held* — `claimed=True` may be a long time coming, and asking for it is a decision
    about how long to wait, not about whether it worked.
    """

    model_config = ConfigDict(extra="allow")

    messageId: str = Field(default="", description="the message's id, as dsh minted it.")
    claimed: bool = Field(default=False, description="a step took it (only possible with wait='claimed').")
    turn: int | None = Field(default=None, description="which turn claimed it, when known.")
    woke: bool = Field(default=False, description="the agent was running when it was injected.")


class ForkResult(BaseModel):
    """One `fork_into()` — a new session that continues this one somewhere else."""

    model_config = ConfigDict(extra="allow")

    sessionId: str = Field(default="", description="the child session's id.")
    boundary: int = Field(default=0, description="the source seq the child inherits up to.")
    seededEvents: int = Field(default=0, description="how many events the seed carried.")


class AdvancedDshAgentSession(DshAgentSession, ABC):
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
    async def fork_into(
        self,
        cwd: str,
        *,
        at_seq: int | None = None,
        agent_preset: str | None = None,
        inherit_policy: bool = True,
    ) -> ForkResult:
        """Fork this session into a **different** working directory.

        dsh's own `session/fork` builds the seed correctly and then hard-codes the child's cwd to
        the source's; since the sandbox's writable root *is* the session cwd, that one field is
        what decides where a clone may act. This verb composes the same public pieces with that
        field chosen instead of copied.

        - `at_seq` cuts at a specific event; omitted, it cuts at the last completed turn (extended
          over trailing standalone events) — the same boundary dsh uses.
        - `inherit_policy` carries the parent's permission over via a `sandbox/mode` /
          `approval/policy` event marked `source: 'delegation'`. Only overrides are copied; a
          parent that never switched has nothing to say.
        - The child is **not** the same session: same history prefix, new environment. Measured on
          a live dsh, that prefix was 61 messages on both sides and the provider's cache read was
          identical — forking into a sibling directory and into `/tmp` alike.
        """

    @abstractmethod
    async def attach_workspace(self, path: str | None = None) -> str:
        """Register this session under the workspace that owns its directory (or `path`).

        A workspace groups sessions by directory; it is not a permission boundary — the writable
        root is the session's cwd. Returns the workspace id, which the caller may want to show.
        """

    @abstractmethod
    async def inject(
        self,
        message: str,
        *,
        wait: Literal["accepted", "claimed"] = "claimed",
        timeout: float = 30.0,
    ) -> InjectResult:
        """Put one message into the conversation **without waking it** — dsh's own `inject`.

        Three facts a caller must hold on to:

        - it lands in the inbox's next-step slot and is claimed at the next **step boundary**,
          which is the next model call — it never takes effect immediately;
        - it does not wake the driver, so on an idle session the message waits until something
          else starts a turn. `steer` lands in the same slot and does wake — that is the whole
          difference between the two verbs;
        - when a step claims it, it enters the log as a `user/message` with the same id. That is
          what `wait="claimed"` waits for; leaving the inbox without appearing there means it was
          cancelled, and raises.
        """

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
    ) -> DshAgentSession:
        """Open a new conversation, in dsh's own terms.

        Returns the low-level session on purpose: this is the source of *conversations*, not of
        our reinforcement. A caller that wants the governed one asks `DshController.create()`,
        which takes a spec and hands back an `AdvancedDshAgentSession`.
        """

    @abstractmethod
    async def attach_session(self, session_id: str) -> DshAgentSession:
        """Take up an existing conversation by its id — the operation dsh calls "attach"."""
