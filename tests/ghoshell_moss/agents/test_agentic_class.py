"""Contract tests for the agentic class surface — a fake driver, no dsh."""

from __future__ import annotations

import asyncio

import pytest

from ghoshell_moss.agents.blueprint.agentic_class import (
    AgentClassDriver,
    AgentConfig,
    AgentCtx,
    AgentMethodError,
    AgentMethodUnavailableError,
    CallRecord,
    agent_class,
    agent_method,
)


class FakeDriver(AgentClassDriver):
    _last: FakeDriver | None = None

    @classmethod
    def for_loop(cls) -> FakeDriver:
        if cls._last is None:
            cls._last = cls()
        return cls._last

    def __init__(self) -> None:
        self.prepared: list[tuple[AgentCtx, object, str]] = []
        self.called: list[tuple[object, str, list, dict]] = []

    async def prepare(self, ctx: AgentCtx, target: object, method) -> None:
        self.prepared.append((ctx, target, method.__name__))

    async def call(self, ctx: AgentCtx, target: object, method, *args, **kwargs):
        self.called.append((target, method.__name__, list(args), dict(kwargs)))
        result = f"secret={target.secret}"
        ctx.calls.append(
            CallRecord(
                method=method.__name__,
                args=list(args),
                kwargs=dict(kwargs),
                returned=result,
                session_id=ctx.session_id,
            )
        )
        return result


def _example():
    @agent_class(AgentConfig(effort="high"), driver=FakeDriver)
    class Example:
        def __init__(self, secret: str) -> None:
            self.secret = secret

        @agent_method
        async def tell_secret(self) -> str:
            ...

    return Example


def test_context_is_per_instance_and_session_is_bindable() -> None:
    FakeDriver._last = None
    Example = _example()

    async def _run() -> None:
        driver = FakeDriver.for_loop()
        first = Example("a")
        second = Example("b")
        first_ctx = AgentCtx.get(first)
        second_ctx = AgentCtx.get(second)
        assert first_ctx is not None
        assert second_ctx is not None
        assert first_ctx is not second_ctx

        first_ctx.set_session_id("session-a")
        assert await first.tell_secret() == "secret=a"
        assert await second.tell_secret() == "secret=b"

        assert driver.called[0][0] is first
        assert driver.called[1][0] is second
        assert first_ctx.session_id == "session-a"
        assert first_ctx.last_call is not None
        assert first_ctx.last_call.returned == "secret=a"
        assert first_ctx.last_call.session_id == "session-a"

    asyncio.run(_run())


def test_session_id_is_frozen_after_the_first_call() -> None:
    FakeDriver._last = None
    Example = _example()

    async def _run() -> None:
        instance = Example("a")
        await instance.tell_secret()
        ctx = AgentCtx.get(instance)
        assert ctx is not None
        with pytest.raises(AgentMethodError, match="before the first call"):
            ctx.set_session_id("late")

    asyncio.run(_run())


def test_unbound_method_is_refused() -> None:
    class Other:
        @agent_method
        async def tell_secret(self) -> str:
            ...

    with pytest.raises(AgentMethodUnavailableError, match="not bound"):
        asyncio.run(Other().tell_secret())
