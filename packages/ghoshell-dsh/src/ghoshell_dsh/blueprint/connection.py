"""Connection — an open dsh, as a driver holds it.

Three things make something a connection rather than a client:

1. **It holds the session factory.** `create_session()` / `attach_session()` / `list_sessions()`
   are how a driver gets conversations; this is the only place they come from.
2. **It distributes events.** A dsh surfaces one stream for everything it is doing; a connection
   is the thing that fans it out. Without this, every session would have to poll.
3. **It can be reached past the contract.** `client()` hands back the protocol layer, and its two
   doors — `call()` for routes this contract does not model, `plugin_call()` for a plugin's own —
   are lifted here so the common case stays short. The plugin route is load-bearing for us: the
   anchor call and the session read will arrive through it.

What a connection is *not*: the HTTP client underneath it. That is scaffolding — an
implementation may hold one, and a future transport may hold none. Nothing in this contract
mentions URLs, cookies, or unary verbs.

`Connection` is a `SessionSource` with more on it, so a caller that only wanted conversations
never sees the rest.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from .agent import SessionSource
from .client import Client, DshRpcError
from .models.llm import LlmProvider
from .models.nouns import WorkspaceView
from .session import Disposer

__all__ = ["Connection", "DshRpcError", "RemoteEmitHandler", "RemoteWaterfallHandler"]

#: A one-way notification dsh emits: an event name and its positional arguments.
RemoteEmitHandler = Callable[[str, list[Any]], Awaitable[None] | None]

#: A request dsh asks the client to answer; returning None means "nothing to say, carry on".
RemoteWaterfallHandler = Callable[
    [str, dict[str, Any]],
    Awaitable[dict[str, Any] | None] | dict[str, Any] | None,
]


class Connection(SessionSource, ABC):
    """An open dsh, driven."""

    # ---- workspaces: the other half of "where" ---- #

    @abstractmethod
    async def workspaces(self) -> list[WorkspaceView]:
        """Every workspace this dsh knows about."""

    @abstractmethod
    async def workspace_for_path(self, path: str) -> WorkspaceView | None:
        """The workspace a host path belongs to — `None` when it belongs to none."""

    # ---- models: what this dsh can be asked as ---- #

    @abstractmethod
    async def llm_providers(self) -> list[LlmProvider]:
        """Every model route this dsh can reach — live ones and merely-declared ones.

        The resource half of the LLM protocol: a caller that is about to name a provider and a
        model asks here first. `live` is what a call will actually succeed on today.
        """

    # ---- events: one stream, fanned out ---- #

    @abstractmethod
    def on_remote_emit(self, handler: RemoteEmitHandler) -> Disposer:
        """Handle one-way notifications dsh emits (an event names itself, with arguments)."""

    @abstractmethod
    def on_remote_waterfall(self, handler: RemoteWaterfallHandler) -> Disposer:
        """Handle requests dsh asks the client to answer, returning the outcome."""

    # ---- the protocol layer underneath ---- #

    @abstractmethod
    def client(self) -> Client:
        """The client this connection is built on.

        Exposed because not every address a consumer needs is a domain verb: our plugin routes
        (the anchor call, the session read) live one level down, and pretending otherwise would
        mean inventing a domain word for each of them.
        """

    # ---- the two doors, lifted for convenience ---- #

    @abstractmethod
    async def call(
        self,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Reach a route the contract does not model. Raw on purpose — it is a door, not a verb."""

    @abstractmethod
    async def plugin_call(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reach a **plugin** route — where our own patches live.

        Kept separate from `call()` because the two travel differently (a plugin registers plain
        HTTP routes; the protocol wraps its own), and because this one is load-bearing for us:
        the anchor call and the session read will arrive through here.
        """

    # ---- lifecycle ---- #

    @abstractmethod
    async def close(self) -> None:
        """Close the connection. Idempotent."""

    async def __aenter__(self) -> Connection:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
