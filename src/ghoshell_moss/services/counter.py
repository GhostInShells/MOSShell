"""Counter/echo service — V1 disposable validation case.

The simplest possible service: an incrementing counter + echo queryable.
Used by the V1 validation plan in FEATURE.md to prove the operator surface
is learnable from a single howto + this file alone.
"""

from typing import Awaitable, Callable
from typing_extensions import Self

from ghoshell_moss.core.blueprint.service import (
    ServiceDeclaration,
    ServiceServer,
    ServiceProvider,
    ServiceOperator,
    Query,
)
from ghoshell_moss.core.blueprint.matrix import Matrix


class CounterDeclaration(ServiceDeclaration):
    """Identity for the counter/echo test service."""

    description: str = "counter/echo disposable test service"

    @classmethod
    def kind(cls) -> str:
        return "counter"


class CounterServer(ServiceServer):
    """Lifecycle wrapper: provide + register queryable handlers.

    Constructed either from a matrix (``new``) or directly from an operator
    (``from_operator``) — the operator seam lets unit tests drive the service
    without a full Matrix/node harness.
    """

    def __init__(
            self,
            operator_factory: Callable[[], Awaitable[ServiceOperator]],
            declaration: CounterDeclaration,
    ):
        self._operator_factory = operator_factory
        self._declaration = declaration
        self._provider: ServiceProvider | None = None
        self._counter = 0

    @property
    def declaration(self) -> ServiceDeclaration:
        return self._declaration

    @property
    def provider(self) -> ServiceProvider:
        if self._provider is None:
            raise RuntimeError("CounterServer not entered")
        return self._provider

    @classmethod
    def new(cls, matrix: Matrix) -> Self:
        return cls(lambda: matrix.service_operator(), CounterDeclaration())

    @classmethod
    def from_operator(cls, operator: ServiceOperator) -> Self:
        """Construct against an already-built operator (test seam)."""

        async def _get() -> ServiceOperator:
            return operator

        return cls(_get, CounterDeclaration())

    async def __aenter__(self) -> Self:
        op = await self._operator_factory()
        self._provider = await op.provide(self._declaration)

        self._provider.queryable('inc', self._on_inc)
        self._provider.queryable('echo', self._on_echo)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._provider = None

    async def _on_inc(self, _query: Query) -> bytes:
        self._counter += 1
        return str(self._counter).encode()

    async def _on_echo(self, query: Query) -> bytes:
        return query['payload'] or b''
