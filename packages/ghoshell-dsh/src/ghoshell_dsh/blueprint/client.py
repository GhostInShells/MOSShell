"""Client — the protocol-level caller underneath a connection.

A connection speaks in conversations; a client speaks in **methods**. That distinction is the
whole reason this layer exists: `session/create`, `session/prompt`, a plugin's HTTP route and a
cordis service call are all the same shape at this depth — a name, a typed payload, a typed
result — and that shape does not change when a transport does.

Why it is a contract and not an implementation detail:

- **Our plugin patches call it.** The anchor call and the session read are plugin routes; they
  are not domain verbs of a session, and they should not be smuggled into one. A client is where
  "reach this route with this payload" belongs.
- **dsh has more than one address family.** The protocol wraps its own requests, a plugin
  registers plain routes, and cordis services are called yet another way. They differ in
  transport, not in intent, so the intent gets one contract and each family gets a method.
- **Every generation would otherwise re-invent it.** Typed params in, validated value out, an
  error that carries the protocol's code — that is a stable idea across dsh versions, and the
  versions differ in what the methods are called, not in that they exist.

What is **not** here: cookies, base URLs, rpc id counters. Those are HTTP's business. A future
transport authenticates its own way and still satisfies this contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, TypeVar

from pydantic import BaseModel

from .models.rpc import RpcError

__all__ = ["Client", "DshRpcError"]

_ValueT = TypeVar("_ValueT", bound=BaseModel)


class DshRpcError(Exception):
    """A call was answered with a protocol error: `code` / `message` / `details`.

    Part of the contract, not of an implementation: a caller has to be able to tell "dsh
    refused this" from "the transport broke", and both arrive as exceptions.
    """

    def __init__(self, method: str, error: RpcError) -> None:
        self.method = method
        self.error = error
        super().__init__(f"dsh rpc {method} failed [{error['code']}]: {error['message']}")


class Client(ABC):
    """The protocol's verbs, with nothing of dsh's domain on top."""

    @abstractmethod
    async def call(
        self,
        method: str,
        params: Optional[BaseModel],
        value_cls: type[_ValueT],
        *,
        args_key: Optional[str] = "request",
    ) -> _ValueT:
        """Call one protocol method: params in, a validated value out.

        `value_cls` is how the result becomes typed rather than a dict — the caller states what
        it expects, and a mismatch surfaces here instead of three frames later.
        """

    @abstractmethod
    async def rpc(self, method: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Call a method whose result has no declared shape — raw in, raw out."""

    @abstractmethod
    async def typert(
        self,
        method: str,
        request: dict[str, Any] | None = None,
        *,
        args_key: str = "_request",
        timeout: float | None = None,
    ) -> dict[str, Any]:
        """Call a typert-wrapped unary route: the request rides under its descriptor's key.

        Newer dsh generations describe their routes with typert, and the gateway validates the
        request against that descriptor — the key is part of the route's shape, not a convention
        (``session/list`` takes ``_request``, ``session/rename`` takes ``request``). ``args_key``
        states it at the call site so one door serves them all.
        """

    @abstractmethod
    async def plugin_call(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        """Reach a route a **plugin** registered — where our own patches live.

        Deliberately a separate method from `call()`: plugins register plain routes and speak
        their own JSON, while the protocol wraps its requests. Two address families, one
        contract.
        """

    @abstractmethod
    async def close(self) -> None:
        """Release whatever the client holds. Idempotent."""
