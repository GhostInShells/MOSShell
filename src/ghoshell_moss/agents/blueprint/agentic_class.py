"""Agentic Class: a class whose ``@agent_method`` functions are blocking ops
implemented by a per-instance agent.

Declare::

    @agent_class(
        AgentConfig(effort="low"),
        driver=DoloresAgentClassDriver,
        reflect_source=[SomeModel],       # imported, small source
        reflect_interface=[SomeAPI],      # imported, compact interface
    )
    class Example:
        @agent_method
        async def op(self, ...) -> ...: ...

Contract:
- One instance owns one session; one method call is one blocking op.
- The agent answers with one module-level ``async def __run__(self, ...)`` or
  ``__debug__``; the turn must contain exactly one of them, otherwise the call
  fails with ``AgentMethodProtocolError``.
- The driver compiles the generated function into a facsimile of the defining
  module and calls it with the live instance; module imports and ``self`` fields
  are available. The return value is passed through unchanged; wrap an op in a
  plain method if you need extra validation.
- Optional ``__context__`` returns a string appended to the first instruction in
  ``<instance-context>``; it is called once per session.
- The class source is the op table; the current op travels in the call envelope.
- Sessions are per instance and process-scoped. An explicit ``AgentConfig.session_id``
  is not shared and must be set before the first call; the driver archives by
  session id on ``close()`` when ``archive_after_close`` is true, and an archived
  session is not resumed.
- Same-instance reentry is rejected; inline the logic in the same ``__run__``.
  Different-instance composition is allowed.
- ``workspace`` / ``writable`` / ``readonly`` are class-level defaults; override
  them per instance with ``AgentCtx.configure(...)`` before the first call.
- Protocol/execution errors retry up to ``max_attempts``. If a failure is a known
  irreversible side effect, raise ``AgentMethodAbortError`` to stop retries.

Use it from a bash sandbox (``python -c``), not a CTML channel.

Debug without a session::

    ctx = AgentCtx.get(instance)
    await ctx.debug_interface()
    await ctx.debug_prompt("op", ...)

``reflect_source`` / ``reflect_interface`` are explicit hints for imported
dependencies only. Use ``reflect_source`` for pydantic/dataclass contracts and
small sources; ``reflect_interface`` for plain classes, modules, or functions
where signatures and docstrings suffice. The class's own module source is always
printed.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import uuid
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Generic, Literal, TypeVar
from weakref import WeakKeyDictionary

from pydantic import BaseModel, Field
from typing_extensions import Self

from ghoshell_moss.agents.blueprint.agent import AgentSession

__all__ = [
    "AgentClassDriver",
    "AgentConfig",
    "AgentCtx",
    "AgentMethod",
    "AgentMethodAbortError",
    "AgentMethodError",
    "AgentMethodProtocolError",
    "AgentMethodReentryError",
    "AgentMethodTimeoutError",
    "AgentMethodUnavailableError",
    "AgentSessionConflictError",
    "CallRecord",
    "agent_class",
    "agent_method",
]


class AgentConfig(BaseModel):
    """Class-level declaration. workspace/writable/readonly are overridable per instance before the first call."""

    preset: str = Field(
        default="",
        description="Agent preset id; empty = platform default.",
    )
    model: str = Field(
        default="",
        description="Model id; empty = platform default.",
    )
    effort: Literal["off", "low", "high", "max"] = Field(
        default="low",
        description="Thinking depth.",
    )
    workspace: str | None = Field(
        default=None,
        description="Entry directory; None = Path.cwd(); must be inside the project home.",
    )
    writable: str | None = Field(
        default=None,
        description="Writable root; None = workspace; must be inside the workspace.",
    )
    readonly: bool = Field(
        default=False,
        description="Declare no writes; enforcement is driver-defined.",
    )
    session_id: str = Field(
        default="",
        description="Explicit session id; not shared; set before the first call. The driver "
        "archives by id on close() when archive_after_close is true.",
    )
    archive_after_close: bool = Field(
        default=True,
        description="Archive the session on close; an archived session is not resumed.",
    )
    max_attempts: int = Field(default=10, description="Maximum protocol retries.")
    max_debug_rounds: int = Field(default=10, description="Maximum debug rounds.")
    timeout: float = Field(default=600, description="Per-call timeout in seconds.")
    debug_result_cap: int = Field(
        default=2000,
        description="Debug result characters before spilling to a file.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Driver-specific escape hatch. Default empty; may rot, keep it out of the blueprint.",
    )


@dataclass
class CallRecord:
    """One op — the last-result handle. The session holds the rich trace."""

    method: str
    args: list[Any]
    kwargs: dict[str, Any]
    returned: Any | None = None
    session_id: str = ""
    error: str = ""


AgentMethod = Callable[..., Awaitable[Any]]
SessionT = TypeVar("SessionT", bound=AgentSession)


class AgentMethodError(RuntimeError):
    """Base class for agentic class failures."""


class AgentMethodUnavailableError(AgentMethodError):
    """The instance/session/driver is not available."""


class AgentMethodProtocolError(AgentMethodError):
    """The agent's answer does not satisfy the __run__/__debug__ protocol."""


class AgentMethodReentryError(AgentMethodError):
    """A method re-entered the same instance's agent call."""


class AgentMethodTimeoutError(AgentMethodError):
    """The dsh turn did not finish inside the call's timeout."""


class AgentMethodAbortError(AgentMethodError):
    """Raised from a generated body to stop retries when a side effect makes retrying unsafe."""


class AgentSessionConflictError(AgentMethodError):
    """The explicit session id is already bound to another instance."""


_INSTANCES: WeakKeyDictionary[object, AgentCtx[Any]] = WeakKeyDictionary()


class AgentCtx(Generic[SessionT]):
    """Per-instance runtime state.

    The driver owns the session; this object is the lookup/debug handle. The
    default lifetime is the process.
    """

    def __init__(
        self,
        config: AgentConfig,
        *,
        driver_type: type[AgentClassDriver[SessionT]],
    ) -> None:
        self.config = config
        self.modulename = ""
        self.class_name = ""
        self.class_doc = ""
        self.file: str | None = None
        self.sandbox: ModuleType | None = None
        self.calls: list[CallRecord] = []
        self.driver_type = driver_type
        self.session: SessionT | None = None
        self.active = False
        self.lock = asyncio.Lock()
        self.instance_id = uuid.uuid4().hex[:8]
        self.reflect_source: list[Any] = []
        self.reflect_interface: list[Any] = []
        self._instance: object | None = None
        self._session_id = config.session_id

    # -- binding -------------------------------------------------------- #

    def bind(self, instance: object) -> None:
        """Bind to an instance. The weak map is a lookup, not the session's owner."""
        self._instance = instance
        _INSTANCES[instance] = self

    @classmethod
    def get(cls, instance: object) -> Self | None:
        return _INSTANCES.get(instance)

    # -- runtime -------------------------------------------------------- #

    @property
    def driver(self) -> AgentClassDriver[SessionT]:
        return self.driver_type.for_loop()

    @property
    def title(self) -> str:
        first = (self.class_doc or "").strip().splitlines()
        name = f"{self.modulename}.{self.class_name}"
        parts = [name, first[0] if first else "", f"#{self.instance_id}"]
        return " · ".join(part for part in parts if part)[:120]

    # -- session -------------------------------------------------------- #

    @property
    def session_id(self) -> str:
        if self.session is not None:
            return self.session.id
        return self._session_id

    def configure(
        self,
        *,
        workspace: str | None = None,
        writable: str | None = None,
        readonly: bool | None = None,
    ) -> None:
        """Override workspace/writable/readonly before the first call."""
        if self.session is not None or self.calls:
            raise AgentMethodError("agentic class: configure() must run before the first call")
        update: dict[str, Any] = {}
        if workspace is not None:
            update["workspace"] = workspace
        if writable is not None:
            update["writable"] = writable
        if readonly is not None:
            update["readonly"] = readonly
        if update:
            self.config = self.config.model_copy(update=update)

    def set_session_id(self, session_id: str) -> None:
        """Set the explicit session before the first call; raise after it."""
        if self.session is not None or self.calls:
            raise AgentMethodError("agentic class: session_id must be set before the first call")
        self._session_id = session_id

    @property
    def last_call(self) -> CallRecord | None:
        return self.calls[-1] if self.calls else None

    # -- context -------------------------------------------------------- #

    async def context_call(self, instance: object) -> str:
        """Render the instance's ``__context__`` hook, if it has one."""
        method = getattr(instance, "__context__", None)
        if method is None:
            return ""
        value = method()
        if inspect.isawaitable(value):
            value = await value
        return value if isinstance(value, str) else str(value)

    # -- debug (RSI) ---------------------------------------------------- #

    async def debug_interface(self) -> str:
        """Render the reflected interface without opening a session."""
        if self._instance is None:
            raise AgentMethodUnavailableError("agentic class: debug needs a bound instance")
        from ._agentic_class_helpers import render_interface

        return render_interface(self, self._instance)

    async def debug_prompt(self, method_name: str, *args: Any, **kwargs: Any) -> str:
        """Render the exact first instruction + call envelope without opening a session."""
        if self._instance is None:
            raise AgentMethodUnavailableError("agentic class: debug needs a bound instance")
        method = getattr(self._instance, method_name, None)
        if method is None:
            raise AgentMethodUnavailableError(f"agentic class: no method {method_name!r}")
        return await self.driver.debug_prompt(self, self._instance, method, args, kwargs)


class AgentClassDriver(ABC, Generic[SessionT]):
    """The runtime behind an agentic class — used from a bash sandbox, not a CTML channel.

    Lifecycle contract:
    - one driver per event loop (`for_loop()`); the loop's owner calls `close()`.
    - `prepare` is locked and registers the process lifecycle on first use.
    - `call` is **blocking**: one method call is one op; the class source is the op table,
      the current op travels in the request envelope.
    - `close` archives every session it opened **by session id**, never by workspace;
      `AgentCtx` itself has no lifecycle and never closes anything.
    """

    @classmethod
    @abstractmethod
    def for_loop(cls) -> Self:
        """The driver instance bound to the running event loop."""

    @abstractmethod
    async def prepare(self, ctx: AgentCtx[SessionT], target: object, method: AgentMethod) -> None:
        """Ensure the instance's session and the module facsimile exist.

        Locked per loop: concurrent first calls must not open two sessions or two
        facsimiles for the same instance.
        """

    @abstractmethod
    async def call(
        self,
        ctx: AgentCtx[SessionT],
        target: object,
        method: AgentMethod,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run one blocking ``__debug__``/``__run__`` exchange and return the method's value.

        The current op is named in the request envelope; the method is not re-rendered.
        """

    async def debug_prompt(
        self,
        ctx: AgentCtx[SessionT],
        target: object,
        method: AgentMethod,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> str:
        """Render the exact first instruction + call envelope without opening a session."""
        raise NotImplementedError

    async def cancel(self, ctx: AgentCtx[SessionT]) -> None:
        """Cancel this instance's in-flight turn; called when the caller is cancelled."""
        return None

    async def close(self) -> None:
        """Archive sessions opened with archive_after_close=True, by id; called by the loop owner."""
        return None


T = TypeVar("T", bound=type)


def agent_class(
    config: AgentConfig | None = None,
    *,
    driver: type[AgentClassDriver] | None = None,
    reflect_source: list[Any] | None = None,
    reflect_interface: list[Any] | None = None,
) -> Callable[[T], T]:
    """Declare a class whose methods are blocking ops implemented by a per-instance agent.

    Returns a real subclass: `isinstance(instance, Original)` holds, class attributes and
    methods stay reachable. The class-level config is copied per instance; override
    workspace/writable/readonly with AgentCtx.configure before the first call.

    `reflect_source` / `reflect_interface` are explicit hints for **imported** dependencies
    only — the class's own module source is always printed. Use `reflect_source` for
    pydantic/dataclass contracts and other small imported sources (the field definitions are
    the API; the model knows the implementation); use `reflect_interface` for plain
    classes/modules/functions where compact signatures/docstrings suffice. This reduces
    interaction rounds when an op needs an imported API. Do not list the class's own module
    here.
    """

    def decorator(cls: T) -> T:
        class Agentic(cls):
            __slots__ = ()

            def __new__(subcls, *args: Any, **kwargs: Any) -> Any:
                if cls.__new__ is object.__new__:
                    instance = object.__new__(subcls)
                else:
                    instance = cls.__new__(subcls, *args, **kwargs)
                if driver is None:
                    raise AgentMethodUnavailableError(
                        "agentic class: driver is required; pass a concrete AgentClassDriver"
                    )
                ctx = AgentCtx((config or AgentConfig()).model_copy(deep=True), driver_type=driver)
                ctx.modulename = cls.__module__
                ctx.class_name = cls.__name__
                ctx.class_doc = inspect.getdoc(cls) or ""
                ctx.file = inspect.getsourcefile(cls)
                ctx.reflect_source = list(reflect_source or [])
                ctx.reflect_interface = list(reflect_interface or [])
                ctx.bind(instance)
                return instance

        for attr in ("__name__", "__qualname__", "__module__", "__doc__"):
            setattr(Agentic, attr, getattr(cls, attr))
        Agentic.__agent_class__ = cls  # type: ignore[attr-defined]
        return Agentic  # type: ignore[return-value]

    return decorator


def agent_method(method: AgentMethod) -> AgentMethod:
    """Declare a blocking op on an agentic class; the body is never executed.

    The agent answers with one ``<run>``/``__run__`` or ``__debug__`` block; a turn
    without one fails with ``AgentMethodProtocolError``. The return value is passed
    through unchanged; wrap the op in a plain method for extra validation.
    Same-instance reentry is rejected; inline the logic instead.
    """

    @functools.wraps(method)
    async def _wrapper(self: object, *args: Any, **kwargs: Any) -> Any:
        ctx = AgentCtx.get(self)
        if ctx is None:
            raise AgentMethodUnavailableError(
                f"agentic class: {type(self).__name__}.{method.__name__} is not bound"
            )
        if ctx.active:
            raise AgentMethodReentryError(
                f"agentic class: {type(self).__name__}.{method.__name__} "
                "called a method on the same instance from inside a generated body"
            )
        driver = ctx.driver
        entered = False
        try:
            async with ctx.lock:
                entered = True
                await driver.prepare(ctx, self, method)
                return await driver.call(ctx, self, method, *args, **kwargs)
        except asyncio.CancelledError:
            # Only cancel the session this call actually entered; a task cancelled while
            # waiting on the instance lock must not cancel another call's turn.
            if entered:
                with contextlib.suppress(Exception, asyncio.CancelledError):
                    await driver.cancel(ctx)
            raise

    _wrapper.__agent_method__ = True  # type: ignore[attr-defined]
    return _wrapper
