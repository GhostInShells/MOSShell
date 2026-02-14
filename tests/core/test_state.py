from ghoshell_moss.core.concepts.states import BaseStateStore, StateBaseModel
from contextlib import AsyncExitStack
import pytest
import asyncio


class FooState(StateBaseModel):
    foo: int = 123

    @classmethod
    def get_state_name(cls) -> str:
        return "foo"


class BarState(StateBaseModel):
    bar: int = 123

    @classmethod
    def get_state_name(cls) -> str:
        return "bar"


class BazState(StateBaseModel):
    baz: int = 123

    @classmethod
    def get_state_name(cls) -> str:
        return "baz"


@pytest.mark.asyncio
async def test_state_baseline():
    parent = BaseStateStore("parent")
    child_1 = BaseStateStore("child_1", parent=parent)
    child_1.register(FooState(), BazState(baz=234))
    child_2 = BaseStateStore("child_2", parent=parent)
    child_2.register(BarState(), BazState(baz=345))

    stack = AsyncExitStack()
    await stack.enter_async_context(parent)
    await stack.enter_async_context(child_1)
    await stack.enter_async_context(child_2)
    async with stack:
        assert child_1.get_model(BarState) is None
        assert child_1.get_model(FooState) is not None

        assert child_2.get_model(FooState) is None
        assert child_2.get_model(BarState) is not None

        assert parent.get_model(BarState) is not None
        assert parent.get_model(FooState) is not None
        assert parent.get_model(BazState).baz == 234

        # 第一个注册的为准.
        assert child_1.get_model(BazState).baz == 234
        assert child_2.get_model(BazState).baz == 234

        await child_1.save(BazState(baz=567))
        assert child_2.get_model(BazState).baz == 567


@pytest.mark.asyncio
async def test_state_parallel():
    parent = BaseStateStore("parent")
    children = []
    for i in range(10):
        child = BaseStateStore("child_{}".format(i), parent=parent)
        child.register(FooState(foo=i), BarState(baz=234))
        children.append(child)
        for j in range(10):
            sub_child = BaseStateStore("child_{}_{}".format(i, j), parent=parent)
            sub_child.register(FooState(foo=i * 10 + i), BarState(baz=234))
            children.append(sub_child)

    async with parent:
        starting = []
        for c in children:
            starting.append(c.start())
        await asyncio.gather(*starting)

        bar = parent.get_model(BarState)
        foo = parent.get_model(FooState)
        assert bar is not None
        assert foo is not None

        for child in children:
            assert child.get_model(BarState).bar == bar.bar
            assert child.get_model(FooState).foo == foo.foo

        updating = []
        count = 100
        for c in reversed(children):
            count += 1
            updating.append(asyncio.create_task(c.save(FooState(foo=count))))
        # 乱续
        await asyncio.wait(updating, return_when=asyncio.ALL_COMPLETED)

        bar = parent.get_model(BarState)
        foo = parent.get_model(FooState)

        for child in children:
            assert child.get_model(BarState).bar == bar.bar
            assert child.get_model(FooState).foo == foo.foo
