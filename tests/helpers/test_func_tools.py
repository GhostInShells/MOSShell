from typing import Coroutine, Awaitable
from ghoshell_moss.helpers.func import awaitable_caller
import pytest


@pytest.mark.asyncio
async def test_awaitable_caller():
    def foo() -> int:
        return 123

    foo = awaitable_caller(foo)
    r = await foo()
    assert r == 123

    async def bar() -> int:
        return 456

    bar = awaitable_caller(bar)
    r = await bar()
    assert r == 456

    def baz() -> Awaitable[int]:
        b = bar()
        assert isinstance(b, Awaitable)
        return b

    baz = awaitable_caller(baz)
    r = await baz()
    assert r == 456
