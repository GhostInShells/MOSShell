"""The dsh session protocol — one conversation, spoken in dsh's own verbs.

This is the **low layer**. Everything here is dsh's semantics: the verbs it exposes, the values
it returns, the content it accepts. Nothing is smoothed over, and nothing is invented. A
generation implements this shape and the caller can rely on dsh's own model of a session being
faithfully reachable.

High-level semantics — `run()` ("send and wait for the turn"), taking a bare `str`, folding the
event stream into a message view — deliberately do **not** live here. Those are our words, not
dsh's, and mixing the two layers in one contract is how a protocol contract turns into a
framework. When that layer is written, the blueprint splits: `DshAgentSession` here, an
`AgentSession` above it built on this one.

### The multimodal protocol

`prompt()` takes `list[PromptContentPart]`, which is the input modality of dsh 0.2.0 stated
precisely: a part is text or an image (`mediaType` + base64 `data`), optionally named. That is
the protocol — not a `str` with an image smuggled alongside it, and not the `ContentBlock`
union used on the way *out*, which carries reasoning and tool traffic a caller never sends.

Consequences worth knowing:

- adding a modality means adding a part type here, in one place, and every implementation
  either supports it or refuses it — no caller has to guess;
- a bare string is *not* accepted at this layer. `prompt("hello")` is a convenience that
  belongs to the layer above; accepting it here would mean the protocol quietly decides what
  string means, which is exactly the kind of hidden semantics this layer exists to avoid.

### Errors, in one sentence

Nothing in this file wraps failures: a verb that cannot complete raises. Implementations raise
their own error type carrying the protocol's error code — see `blueprint.models.rpc.RpcError`.
A named exception type for the whole blueprint has not been decided yet; until it is, callers
should catch broadly and read the code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Literal

from .models import sessions
from .models.attachments import Attachment
from .models.session_events import Message, SessionEvent, SessionEventModel

__all__ = [
    "WILDCARD_EVENT",
    "Disposer",
    "DshAgentSession",
    "SessionCallback",
]

#: Observing every event, regardless of type.
WILDCARD_EVENT = "*"

#: How a prompt joins the session: after the current turn (`queue`), or into it (`steer`).
PromptMode = Literal["queue", "steer"]

#: What registering a handler returns: call it and the handler is gone.
Disposer = Callable[[], None]

#: A consumer's raw event handler.
SessionCallback = Callable[[SessionEvent], Awaitable[None]]


class DshAgentSession(ABC):
    """One dsh session, driven with dsh's own verbs.

    Two states are distinct and both matter: is the session *alive* (`is_running`), and is a
    turn *in flight* (`is_busy`). Confusing them is how a scheduler sends into a dead session,
    or sends twice into a live one.

    Both are properties, and so is `session_id`: they read facts the object already holds. The
    verbs in this class — `prompt`, `cancel`, `history`, `close`, `state` — are the ones that
    ask dsh to do something, and they stay callable.
    """

    # ---- identity ------------------------------------------------------- #

    @property
    @abstractmethod
    def session_id(self) -> str:
        """The address of this conversation, stable for its whole life.

        A property, not a method: it reads a fact the object already holds. A caller writing
        `session.session_id()` would only be spelling out a call that does nothing.
        """

    # ---- driving -------------------------------------------------------- #

    @abstractmethod
    async def prompt(
        self,
        *,
        content: list[sessions.PromptContentPart],
        mode: PromptMode = "queue",
        client_timezone: str | None = None,
        request_id: str | None = None,
    ) -> sessions.SessionPromptValue:
        """Submit a prompt. Returns when dsh accepts it — *not* when the turn is over.

        `mode="queue"` puts it behind the current turn; `mode="steer"` feeds it into the turn
        that is running. Watching the answer is the event stream's job, not this call's.

        `request_id` is the idempotency key: mint one before sending, reuse it when retrying,
        and a prompt that was accepted twice is only delivered once. `None` means the
        implementation mints its own — fine for a single attempt, not for a retry.
        """

    @abstractmethod
    async def cancel(self) -> sessions.SessionCancelValue:
        """Ask dsh to stop the turn in flight. Does not close the session."""

    # ---- reading back --------------------------------------------------- #

    @abstractmethod
    async def history(
        self,
        *,
        before_seq: int | None = None,
        max_messages: int | None = None,
    ) -> list[SessionEvent]:
        """Load past events, newest page first going backwards.

        Envelopes, not projections: the payload stays in `SessionEvent.data` until a consumer
        decides what to make of it.
        """

    # ---- observing ------------------------------------------------------ #

    @abstractmethod
    def on_session_event(self, event_type: str, callback: SessionCallback) -> Disposer:
        """Call `callback` for every event of `event_type` — or all of them, for `WILDCARD_EVENT`."""

    @abstractmethod
    def on_session_event_model(
        self,
        model_cls: type[SessionEventModel],
        callback: Callable[[SessionEventModel], Awaitable[None]],
    ) -> Disposer:
        """The typed path: `callback` receives the event already projected onto `model_cls`."""

    # ---- state ---------------------------------------------------------- #

    @property
    @abstractmethod
    def is_running(self) -> bool:
        """Whether the conversation is still there — not whether it is busy."""

    @property
    @abstractmethod
    def is_busy(self) -> bool:
        """Whether a turn is in flight right now."""

    @abstractmethod
    async def state(self, *, refresh: bool = False) -> SessionState:
        """Everything cheap about this session, in one read.

        Coherence, not convenience: a driver deciding what to do next must not assemble its
        picture from six calls that a running turn can slip between. `refresh` re-reads from the
        source; without it an implementation may answer from cache.

        Mandatory, not optional: the whole point is that a driver can look before it leaps, and
        an optional state read would put back the "does this implementation support it" guess.
        `is_running` / `is_busy` stay as cheap synchronous predicates — the rule between them is
        that predicates are local facts, the snapshot is a coherent picture.
        """
        raise NotImplementedError

    # ---- lifecycle ------------------------------------------------------ #

    @abstractmethod
    async def close(self) -> None:
        """End the conversation. Idempotent."""

    async def __aenter__(self) -> DshAgentSession:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    # ------------------------------------------------------------------ #
    # Optional verbs. dsh exposes them; a given implementation may not be
    # able to. Raising beats silently ignoring: a caller that asked for a
    # capability deserves to know it is absent.
    # ------------------------------------------------------------------ #

    async def update_queue(
        self,
        *,
        item_id: str,
        action: sessions.QueueAction,
    ) -> sessions.SessionUpdateQueueValue:
        """Edit, remove, or steer an already-queued prompt."""
        raise NotImplementedError(f"{type(self).__name__} cannot edit its queue")

    async def select_model(
        self,
        *,
        provider: str,
        model: str,
        reasoning_effort: str | None = None,
    ) -> sessions.SessionSelectModelValue:
        """Change the model that answers from the next turn on."""
        raise NotImplementedError(f"{type(self).__name__} does not allow model selection")

    async def models(self) -> sessions.ModelCatalog:
        """The model catalog this session can be pointed at."""
        raise NotImplementedError(f"{type(self).__name__} does not expose a model catalog")

    async def fork(self, *, at_seq: int | None = None) -> sessions.SessionForkValue:
        """Branch this conversation at a point."""
        raise NotImplementedError(f"{type(self).__name__} cannot be forked")

    async def rename(self, *, title: str) -> sessions.SessionRenameValue:
        """Give the conversation a new title."""
        raise NotImplementedError(f"{type(self).__name__} cannot be renamed")

    async def attach(
        self,
        source: str,
        *,
        media_type: str = "",
        name: str = "",
    ) -> Attachment:
        """Put a file into the session's attachment store and return its reference.

        `source` is a host-side path; a caller that already holds bytes should write them
        somewhere first, because an attachment's identity is the bytes themselves and a store
        that cannot address them is not a store.

        Attachments are how anything that is not text reaches the model. Getting this wrong is
        not a cosmetic bug: it is the difference between "the agent saw the screenshot" and
        "the agent saw a filename".
        """
        raise NotImplementedError(f"{type(self).__name__} cannot store attachments")

    async def attachment(self, attachment_id: str) -> Attachment:
        """Look up an attachment the session has referenced, by its content address."""
        raise NotImplementedError(f"{type(self).__name__} has no attachments")

    async def attachment_bytes(self, attachment_id: str) -> bytes:
        """Read an attachment's bytes back — what a driver needs to actually look at it."""
        raise NotImplementedError(f"{type(self).__name__} cannot read attachment content")

    async def instruction(self) -> str | None:
        """The system instruction in force."""
        raise NotImplementedError(f"{type(self).__name__} does not expose its instruction")

    async def surface_messages(self) -> list[Message]:
        """The conversation folded into messages — dsh's own surface view, not a projection of ours."""
        raise NotImplementedError(f"{type(self).__name__} does not expose a message view")

    async def when_running(self) -> None:
        """Wait until a turn starts."""
        raise NotImplementedError(f"{type(self).__name__} cannot be waited on")

    async def when_idle(self) -> None:
        """Wait until no turn is in flight."""
        raise NotImplementedError(f"{type(self).__name__} cannot be waited on")
