from ghoshell_moss.concepts.command import PyCommand, CommandType, CommandMeta
import asyncio
import pytest


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
        assert 6 == v

        meta = await command.meta()
        assert meta.name == 'foo'
        assert meta.chan == ""
        assert meta.description == ""
        assert meta.type is CommandType.FUNCTION.value
        assert meta.delta_arg is None
        assert meta.available
        assert meta.interface == expect

    asyncio.run(main())


@pytest.mark.asyncio
async def test_pycommand_rename():
    command = PyCommand(foo, meta=CommandMeta(name="bar"))
    meta = await command.meta()
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
    meta = await command.meta()
    assert meta.interface == bar_itf_expect

    # assert the args and kwargs are parsed into kwargs
    kwargs = command.parse_kwargs(1, "foo", "bar", c="hello")
    assert kwargs == {"a": 1, "b": ('foo', 'bar'), "c": "hello", "d": 1}


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
    bar_meta = await bar_cmd.meta()
    assert bar_meta.interface == expect_bar

    # the cls has been ignored
    baz_meta = await baz_cmd.meta()
    assert baz_meta.interface == expect_baz

    # and the calling need not pass self or cls
    assert await baz_cmd() == 1
    assert await bar_cmd() == 1
