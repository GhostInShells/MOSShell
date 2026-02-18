import asyncio
from collections.abc import AsyncIterable

import pytest

from ghoshell_moss.core.concepts.command import CommandType, PyCommand


async def foo(a: int, b: str = "hello") -> int:
    return a + len(b)


foo_itf_expect = """
async def foo(a: int, b: str = 'hello') -> int:
    pass
""".strip()


def test_pycommand_baseline():
    expect = foo_itf_expect

    command = PyCommand(foo)

    async def main():
        v = await command(1, b="world")
        assert v == 6

        meta = command.meta()
        assert meta.name == "foo"
        assert meta.chan == ""
        assert meta.type is CommandType.FUNCTION.value
        assert meta.delta_arg is None
        assert meta.available
        assert meta.interface == expect

    asyncio.run(main())


@pytest.mark.asyncio
async def test_pycommand_rename():
    command = PyCommand(foo, name="bar")
    meta = command.meta()
    assert meta.name == "bar"
    assert meta.interface == foo_itf_expect.replace("foo", "bar")


@pytest.mark.asyncio
async def test_func_with_args_and_kwargs():
    async def bar(a: int, *b: str, c: str, d: int = 1) -> int:
        """example with args and kwargs"""
        return a + len(list(b)) + len(c) + d

    bar_itf_expect = """
async def bar(a: int, *b: str, c: str, d: int = 1) -> int:
    '''
    example with args and kwargs
    '''
    pass
""".strip()

    command = PyCommand(bar)
    meta = command.meta()
    assert meta.interface == bar_itf_expect

    # assert the args and kwargs are parsed into kwargs
    kwargs = command.parse_kwargs(1, "foo", "bar", c="hello")
    assert kwargs == {"a": 1, "b": ("foo", "bar"), "c": "hello", "d": 1}


@pytest.mark.asyncio
async def test_method_command():
    class Foo:
        async def bar(self) -> int:
            return 1

        @classmethod
        async def baz(cls) -> int:
            return 1

    expect_bar = """
async def bar() -> int:
    pass
""".strip()
    expect_baz = """
async def baz() -> int:
    pass
""".strip()

    bar_cmd = PyCommand(Foo().bar)
    baz_cmd = PyCommand(Foo.baz)
    # the self has been ignored
    bar_meta = bar_cmd.meta()
    assert bar_meta.interface == expect_bar

    # the cls has been ignored
    baz_meta = baz_cmd.meta()
    assert baz_meta.interface == expect_baz

    # and the calling need not pass self or cls
    assert await baz_cmd() == 1
    assert await bar_cmd() == 1


@pytest.mark.asyncio
async def test_delta_args_command():
    async def bar(a: int, b: str = "hello", text__: str = "") -> int:
        return 123

    command = PyCommand(bar)
    meta = command.meta()
    assert meta.name == "bar"
    assert meta.delta_arg == "text__"

    async def baz(a: int, b: str, tokens__: AsyncIterable[str]) -> int:
        return 123

    command = PyCommand(baz)
    meta = command.meta()
    assert meta.name == "baz"
    assert meta.delta_arg == "tokens__"


@pytest.mark.asyncio
async def test_command_rename():
    async def _foo():
        return 123

    command = PyCommand(_foo, name="bar")
    assert command.name() == "bar"
    assert command.meta().name == "bar"

    command = PyCommand(_foo, name="bar", chan="test")
    assert command.name() == "bar"
    assert command.meta().chan == "test"


@pytest.mark.asyncio
async def test_command_with_sync_func():
    def bar():
        return 123

    command = PyCommand(bar)
    assert await command() == 123
