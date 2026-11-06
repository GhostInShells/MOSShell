"""The controller — the same dsh, with our reinforcement on top of its native verbs.

`Connection` is dsh's API as dsh states it: native verbs, native shapes, native defaults. Nothing
in it is ours. `DshController` is the *same surface upgraded*: what a long-lived consumer needs,
stated in our words.

Two independent abstractions, not two layers of one:

- neither inherits the other — a controller is not a kind of connection, and it does not promise
  to hold one. What a controller needs underneath is its own business;
- a concrete implementation may satisfy **both**, and the v0.2.0 one does. Callers that want
  dsh's own verbs ask for `Connection`; callers that want the governed ones ask for
  `DshController`. The same object answers both, which is exactly the point: the reinforced face
  never costs you the native one.

Why the reinforcement exists at all — dsh's raw verbs are thin in the places that matter to a
long-lived agent:

- `session/create` takes four fields — workspace, cwd, id, preset. No title, no read-only, no
  model. Creating a usable conversation is therefore a *composition* (create, then rename, then
  select a model), and every consumer would otherwise re-derive it, differently.
- `session/fork` takes a session id and a seq. Nothing else. "Branch from this point, but hand the
  branch to a different model in a read-only workspace" is a sentence dsh cannot hear.
- `session/list` gives ids and timestamps. Titles, and the notion of a session that is *done*
  with, are ours.

So this layer states the operations we need, in our words. Where dsh can do it, an
implementation composes dsh's verbs. Where dsh cannot — moving a session between workspaces,
archiving one, honouring `readonly` at creation — **our own plugin supplies the primitive**.
That is the arrangement: the contract says what a consumer needs, and the gaps become plugin
work rather than consumer work.

### Naming

`DshController`, not `SessionController`: what it controls is the dsh it drives — conversations
*and* the workspaces they live in — and the `Dsh` prefix keeps it honest about the fact that
these are dsh's resources, not a generic session abstraction of ours.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict, Field

from .agent import AgentSession
from .models.attachments import Attachment  # noqa: F401  (re-exported for convenience)
from .models.nouns import WorkspaceView
from .models.sessions import SessionSummary

__all__ = ["DshController", "ForkSpec", "SessionSpec"]


class SessionSpec(BaseModel):
    """What a caller wants a new conversation to be.

    Every field is optional because a conversation is usable without any of them — but each one
    that dsh cannot take directly is a promise this layer keeps for the caller: create it, then
    make it so.
    """

    model_config = ConfigDict(extra="allow")

    title: str | None = Field(default=None, description="a human name; dsh has no create-time field for it.")
    workspace_id: str | None = Field(default=None, description="which workspace it belongs to.")
    cwd: str | None = Field(default=None, description="the directory the agent works in.")
    agent_preset: str | None = Field(default=None, description="which preset shapes the agent.")
    readonly: bool | None = Field(default=None, description="whether the session may change anything.")
    provider: str | None = Field(default=None, description="which provider answers, e.g. `deepseek-official`.")
    model: str | None = Field(default=None, description="which model answers, from the first turn on.")
    reasoning_effort: str | None = Field(default=None, description="thinking depth for that model.")
    session_id: str | None = Field(default=None, description="caller-chosen id, for a predictable address.")


class ForkSpec(BaseModel):
    """Branch an existing conversation, and say what the branch should be instead.

    Forking is where the difference between "dsh's verbs" and "what we need" is sharpest: dsh
    forks at a seq and nothing more, while a branch usually exists *because* something should
    differ — another model, read-only, a different workspace.
    """

    model_config = ConfigDict(extra="allow")

    source: str = Field(description="the session id to branch from.")
    at_seq: int | None = Field(
        default=None,
        description="branch point; None means the source's current end.",
    )
    title: str | None = Field(default=None, description="the branch's own name.")
    readonly: bool | None = Field(default=None, description="permission for the branch.")
    provider: str | None = Field(default=None, description="provider for the branch.")
    model: str | None = Field(default=None, description="model for the branch.")
    reasoning_effort: str | None = Field(default=None, description="thinking depth for the branch.")
    workspace_id: str | None = Field(default=None, description="workspace for the branch.")


class DshController(ABC):
    """dsh's resources, governed — the reinforced face of a connection.

    Deliberately not a subclass of `Connection`: a controller may be implemented over a
    connection, over a plugin bridge, or over a future transport that has no connection at all.
    What it promises is the governed verbs, not the plumbing underneath them.
    """

    # ---- discovery ------------------------------------------------------ #

    @abstractmethod
    async def workspaces(self) -> list[WorkspaceView]:
        """Every workspace, with what it holds."""

    @abstractmethod
    async def sessions(
        self,
        *,
        workspace_id: str | None = None,
        include_archived: bool = False,
    ) -> list[SessionSummary]:
        """The sessions a consumer should know about — the governed listing.

        Named differently from `Connection.list_sessions()` on purpose: this one filters, hides
        archived conversations by default and knows about workspaces, so it is a different
        question with a different answer. Two faces on one object, and the names say which.

        Archived ones are hidden by default: a list that grows forever stops being a list —
        this is the whole reason `archive()` exists.
        """

    # ---- creating ------------------------------------------------------- #

    @abstractmethod
    async def create(self, spec: SessionSpec) -> AgentSession:
        """Open a new conversation, as fully specified as the caller asked."""

    @abstractmethod
    async def fork(self, spec: ForkSpec) -> AgentSession:
        """Branch a conversation at a point, with the branch's own settings."""

    @abstractmethod
    async def connect(self, session_id: str) -> AgentSession:
        """Take up an existing conversation."""

    # ---- lifecycle of a conversation ------------------------------------ #

    @abstractmethod
    async def rename(self, session_id: str, *, title: str) -> None:
        """Give a conversation a name a human can find it by."""

    @abstractmethod
    async def archive(self, session_id: str) -> None:
        """Retire a conversation without deleting it.

        Deliberately not `delete`: a conversation is evidence, and the useful operation is
        "stop showing me this", not "forget it happened". Destructive removal, if it is ever
        offered, is a separate decision.
        """

    @abstractmethod
    async def restore(self, session_id: str) -> None:
        """Bring an archived conversation back into the list."""

    # ---- governance of a workspace -------------------------------------- #

    @abstractmethod
    async def create_workspace(self, *, path: str, title: str | None = None) -> WorkspaceView:
        """Register a directory as a workspace."""

    @abstractmethod
    async def move_session(self, session_id: str, *, workspace_id: str) -> None:
        """Move a conversation into another workspace.

        dsh has no verb for this. It is the clearest case of a primitive our plugin has to
        provide — the contract states the need, the plugin fills the hole.
        """
