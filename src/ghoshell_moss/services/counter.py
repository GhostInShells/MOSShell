"""Counter/echo service — V1 disposable validation case.

The simplest possible service: an incrementing counter + echo queryable.
Used by the V1 validation plan in FEATURE.md to prove the operator surface
is learnable from a single howto + this file alone.
"""

from typing_extensions import Self

from ghoshell_moss.core.blueprint.service import (
    ServiceDeclaration,
    ServiceServer,
    ServiceProvider,
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
    """Lifecycle wrapper: provide + register queryable handlers."""

    def __init__(self, matrix: Matrix, declaration: CounterDeclaration):
        self._matrix = matrix
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
        return cls(matrix, CounterDeclaration())

    async def __aenter__(self) -> Self:
        op = await self._matrix.service_operator()
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
        return query.payload or b''
