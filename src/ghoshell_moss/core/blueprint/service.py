"""
Addressed service communication switchboard.

A cell declares its capabilities through the Service.  Each declaration
is a ``ServiceDeclaration`` — a singleton per service kind per cell.
It owns lifecycle (enter announces, exit revokes) and exposes the
service's business keys.

Consumers discover declared services and communicate via methods keyed by address and business key.
Wildcard / aggregation are the caller's responsibility .

The operator itself does NOT encapsulate the business server or client
— it provides the wire layer.
"""

from typing import Callable, Awaitable, Any, TypedDict, Optional, TYPE_CHECKING
from typing_extensions import Self
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from .cell import CellAddress

if TYPE_CHECKING:
    from .matrix import Matrix

__all__ = [
    'ServiceServer', 'ServiceClient',
    'ServiceMeta', 'ServiceDeclaration',
    'ServiceProvider', 'ServiceOperator',
    'Query', 'Sample', 'Reply', 'Handle',
]


class ServiceMeta(TypedDict):
    """Identity of a declared service."""
    address: str  # service cell address on the matrix mesh
    kind: str  # service kind: webview / resource / ...
    data: dict[str, Any]  # "kind-specific metadata (schema owned by each kind)"


class ServiceDeclaration(BaseModel, ABC):
    """Identity of a declared service."""

    @classmethod
    @abstractmethod
    def kind(cls) -> str:
        """Service kind."""
        ...

    def to_meta(self, address: CellAddress) -> ServiceMeta:
        return ServiceMeta(
            address=address,
            kind=self.kind(),
            data=self.model_dump(
                mode='json',
                exclude_none=True,
            )
        )

    @classmethod
    def from_meta(cls, meta: ServiceMeta) -> Self | None:
        if meta['kind'] != cls.kind():
            return None
        # handle the exception if the metadata is not match the schema
        return cls.model_validate(meta['data'])


class Query(TypedDict):
    """Request received by a service's ``queryable`` handler."""
    address: str  # caller address — who sent the request
    key: str  # business key being queried (one of the kind's n semantics)
    payload: Optional[bytes]  # params payload
    timestamp: float


class Sample(TypedDict):
    """the data published or listened by a service"""
    address: str  # caller address — who sent the request
    key: str  # business key being queried (one of the kind's n semantics)
    payload: bytes  # params payload
    timestamp: float


class Reply(TypedDict):
    """Response returned by ``get``, keyed for aggregation across services."""
    address: str  # replier address
    key: str  # business key replied to
    payload: bytes
    timestamp: float


class Handle(ABC):
    """Reusable close handle — subscription, queryable, token.  ``close()`` is idempotent."""

    @property
    @abstractmethod
    def key(self) -> str:
        """busyness key."""
        ...

    @abstractmethod
    def close(self) -> None:
        """close the handler"""
        ...


class ServiceProvider(ABC):
    """A capability declared by this cell — singleton per kind per cell.

    Enter manages the lifecycle (announce — token + queryable; the operator
    auto-registers the meta queryable so ``services()`` can discover it).
    Exit revokes (undeclares) cleanly.
    """

    @property
    @abstractmethod
    def meta(self) -> ServiceMeta:
        ...

    @abstractmethod
    def queryable(
            self,
            key: str,
            handler: Callable[[Query], Awaitable[bytes]],
    ) -> Handle:
        """Register a request-reply handler for a business ``key``."""

    @abstractmethod
    def pub(self, key: str, payload: bytes) -> None:
        """publish an event on a business ``key`` (fan-out to subscribers)."""
        ...

    @abstractmethod
    def listen(self, key: str, handler: Callable[[Sample], Awaitable[None]]) -> Handle:
        """listen to any client emitted events"""
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        ...


class ServiceOperator(ABC):
    """Matrix Cell-Addressed service switchboard.

    Bound to ``matrix``, obtained via ``await matrix.operator()``.

    Lifecycle is composed into ``matrix`` — callers do not enter/exit
    the operator directly.
    """

    # -- declaration -------------------------------------------------

    @abstractmethod
    async def provide(self, declaration: ServiceDeclaration) -> ServiceProvider:
        """Declare a service.  Singleton per kind — second call raises."""

    # -- discovery ---------------------------------------------------

    @abstractmethod
    async def get_services_by_kind(self, kind: str) -> list[ServiceMeta]:
        """All currently declared (and live) services of a given ``kind``."""

    @abstractmethod
    async def get_services_by_address(self, address: str) -> list[ServiceMeta]:
        """All service kinds declared at a given ``address``."""

    @abstractmethod
    def on_service_start(
            self,
            kind: str,
            callback: Callable[[ServiceMeta], None],
    ) -> Handle:
        """``callback`` is fired when a service of ``kind`` comes online."""

    @abstractmethod
    def on_service_stop(
            self,
            kind: str,
            callback: Callable[[ServiceMeta], None],
    ) -> Handle:
        """``callback`` is fired when a service of ``kind`` goes offline."""

    # -- connection (typed by ServiceMeta + business key) -----------

    @abstractmethod
    async def get(
            self,
            kind: str,
            key: str,
            params: bytes | None,
            *services: ServiceMeta,
    ) -> list[Reply]:
        """send query to the specific services, or all services."""
        ...

    @abstractmethod
    def sub(
            self,
            kind: str,
            key: str,
            handler: Callable[[Sample], Awaitable[None]],
            *services: ServiceMeta,
    ) -> Handle:
        """ subscribe the samples published from a service or all services."""
        ...

    @abstractmethod
    async def emit(
            self,
            kind: str,
            key: str,
            payload: bytes,
            *services: ServiceMeta,
    ) -> None:
        """Publish an event on a business ``key`` listening by the services or all"""
        ...

    # -- lifecycle ---------------------------------------------------

    @abstractmethod
    async def __aenter__(self) -> Self: ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...


class ServiceServer(ABC):
    """lifecycle interface of a service with service provider"""

    @property
    @abstractmethod
    def declaration(self) -> ServiceDeclaration:
        """the declaration of the service"""
        ...

    @property
    @abstractmethod
    def provider(self) -> ServiceProvider:
        """the service provider"""
        ...

    @classmethod
    @abstractmethod
    def new(cls, matrix: 'Matrix') -> Self:
        """create from the matrix"""
        ...

    @abstractmethod
    async def __aenter__(self) -> Self: ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...


class ServiceClient(ABC):
    """lifecycle interface of a service client with operator"""

    @abstractmethod
    async def get_connected(self) -> list[ServiceDeclaration]:
        """ query the alive services"""
        ...

    @classmethod
    @abstractmethod
    def new(cls, matrix: 'Matrix') -> Self:
        ...

    @abstractmethod
    async def __aenter__(self) -> Self: ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None: ...
