"""Serve side of the web view kind — announce, own the condition, publish it.

The wire surface is three business keys over two primitives::

    query  state      snapshot, for whoever arrives after a change
    pub    state      change stream
    query  activate   a consumer asks this view to take focus

The serve side's own transitions are method calls (``touch`` / ``notify``), not
keys: a node has something to say about *itself*, and only the resulting
condition travels.  No consumer is addressed, so no consumer needs addressing.

What is deliberately absent: any notion of who is looking.  A cell cannot see the
mesh, so judging "is someone watching me" is not possible locally and is left to
the consumer — an unread count keeps counting while a screen holds the view open,
and the consumer decides what to render.
"""

import time
from typing import Awaitable, Callable

from typing_extensions import Self

from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.service import (
    Query,
    ServiceDeclaration,
    ServiceOperator,
    ServiceProvider,
    ServiceServer,
)

from .declaration import ACTIVATE_KEY, STATE_KEY, WebViewDeclaration, WebViewState

__all__ = ['WebViewServer']


class WebViewServer(ServiceServer):
    """A cell's web face, announced on the mesh.

    The class is a base for node servers: a web view's identity is runtime data
    (the url is wherever the node ended up listening), so the node either passes
    a declaration explicitly via ``serve`` or subclasses and overrides ``new``
    with its own startup logic.
    """

    def __init__(
            self,
            declaration: WebViewDeclaration,
            operator_factory: Callable[[], Awaitable[ServiceOperator]],
    ):
        self._declaration = declaration
        self._operator_factory = operator_factory
        self._provider: ServiceProvider | None = None
        # A view is born active: its activity stamp starts at its birth, so it
        # has a defined place in a recency-ordered region before anything happens.
        self._state = WebViewState(last_activity_at=declaration.created)

    # -- ServiceServer interface -----------------------------------------

    @property
    def declaration(self) -> ServiceDeclaration:
        return self._declaration

    @property
    def provider(self) -> ServiceProvider:
        if self._provider is None:
            raise RuntimeError("WebViewServer not entered")
        return self._provider

    @property
    def state(self) -> WebViewState:
        """Copy of the current condition — the node's own view of itself."""
        return self._state.model_copy()

    @classmethod
    async def serve(cls, matrix: Matrix, declaration: WebViewDeclaration) -> Self:
        """Construct with an explicit declaration and register into the matrix lifecycle.

        The counterpart of ``Matrix.serve_service`` for kinds whose identity is
        runtime data rather than declared configuration.
        """
        server = cls(declaration, lambda: matrix.service_operator())
        await matrix.add_lifecycle_object(server)
        return server

    @classmethod
    def new(cls, matrix: Matrix) -> Self:
        """Not derivable — see the class docstring.

        A web view's identity is where its page is served, which the matrix does
        not know.  Prefer ``serve(matrix, declaration)``.
        """
        raise RuntimeError(
            "webview identity is runtime data (url/port) known only to the node: "
            "use WebViewServer.serve(matrix, declaration), or subclass and "
            "override new() with the node's own startup logic"
        )

    @classmethod
    def from_operator(
            cls,
            operator: ServiceOperator,
            declaration: WebViewDeclaration,
    ) -> Self:
        """Construct against an already-built operator (test seam)."""

        async def _operator() -> ServiceOperator:
            return operator

        return cls(declaration, _operator)

    async def __aenter__(self) -> Self:
        operator = await self._operator_factory()
        provider = await operator.provide(self._declaration)
        self._provider = provider
        provider.queryable(STATE_KEY, self._on_state_query)
        provider.queryable(ACTIVATE_KEY, self._on_activate)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        provider = self._provider
        self._provider = None
        if provider is not None:
            await provider.__aexit__(exc_type, exc_val, exc_tb)

    # -- the node's own transitions --------------------------------------

    def touch(self) -> None:
        """Claim attention without claiming there is anything to read.

        A key interaction that is worth surfacing but produced nothing for a
        human: the ghost started speaking, a body moved.
        """
        self._require_running()
        self._state.last_activity_at = time.time()
        self._publish()

    def notify(self, message: str | None = None) -> None:
        """Claim attention and record something for a human.

        Note that this raises the view's ordering without taking focus: an IM
        conversation rises in the list without opening itself.
        """
        self._require_running()
        self._state.last_activity_at = time.time()
        self._state.unread_count += 1
        self._state.last_message = message
        self._publish()

    # -- operator handlers -----------------------------------------------

    # Both handlers are ``async def`` although neither awaits, and that is
    # deliberate: the operator offloads *sync* handlers to a worker thread, and
    # every mutation above runs on the loop.  A sync handler here would read or
    # write the condition from another thread and race the node's own calls.

    async def _on_state_query(self, _query: Query) -> bytes:
        return self._state.encode()

    async def _on_activate(self, _query: Query) -> bytes:
        now = time.time()
        self._state.focused_at = now
        self._state.last_activity_at = now
        # Looking at it is reading it.  ``last_message`` stays: having seen it
        # does not erase what it was.
        self._state.unread_count = 0
        self._publish()
        return self._state.encode()

    def _require_running(self) -> None:
        if self._provider is None:
            raise RuntimeError(
                "webview server is not running: touch/notify are only meaningful "
                "between __aenter__ and __aexit__"
            )

    def _publish(self) -> None:
        self.provider.pub(STATE_KEY, self._state.encode())
