import asyncio

import pytest

from ghoshell_moss.core.concepts.channel import ChannelCtx
from ghoshell_moss.core.concepts.command import CommandTask, PyCommand
from ghoshell_moss.core.concepts.errors import CommandError, CommandErrorCode
from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.message import Message, new_text_message

chan = PyChannel(name="test")


@chan.build.command()
def add(a: int, b: int) -> int:
    """测试一个同步函数是否能正确被调用."""
    return a + b


@chan.build.command()
async def foo() -> int:
    return 9527


@chan.build.command()
async def bar(text: str) -> str:
    return text


@chan.build.command(name="help")
async def some_command_name_will_be_changed_helplessly() -> str:
    return "help"


class Available:
    def __init__(self):
        self.available = True

    def get(self) -> bool:
        return self.available


available_mutator = Available()


@chan.build.command(available=available_mutator.get)
async def available_test_fn() -> int:
    return 123


@pytest.mark.asyncio
async def test_py_channel_baseline() -> None:
    async with chan.bootstrap() as broker:
        assert chan.name() == "test"
        assert broker.is_connected()
        assert broker.is_running()
        assert broker.is_connected()

        # commands 存在.
        commands = list(broker.commands().values())
        assert len(commands) > 0

        # 所有的命令应该都以 channel 开头.
        for command in commands:
            assert command.meta().chan == "test"

        # 不用全名来获取函数.
        foo_cmd = broker.get_command("foo")
        assert foo_cmd is not None
        assert await foo_cmd() == 9527

        # 测试名称有效.
        help_cmd = broker.get_command("help")
        assert help_cmd is not None
        assert await help_cmd() == "help"

        # 测试乱取拿不到东西
        none_cmd = broker.get_command("never_exists_command")
        assert none_cmd is None
        # full name 不正确也拿不到.
        help_cmd = broker.get_command("help")
        assert help_cmd is not None

        # available 测试.
        available_test_cmd = broker.get_command("available_test_fn")
        assert available_test_cmd is not None
        # 当为 True 的时候.
        assert available_mutator.available
        assert available_test_cmd.is_available() == available_mutator.available
        # 当为 False 的时候, 应该都不能用.
        available_mutator.available = False
        assert available_test_cmd.is_available() == available_mutator.available


@pytest.mark.asyncio
async def test_py_channel_children() -> None:
    assert len(chan.children()) == 0

    a_chan = chan.new_child("a")
    assert isinstance(a_chan, PyChannel)
    assert chan.children()["a"] is a_chan

    async def zoo():
        return 123

    zoo_cmd = a_chan.build.command(return_command=True)(zoo)
    assert isinstance(zoo_cmd, PyCommand)

    async with a_chan.bootstrap():
        meta = a_chan.broker.meta()
        assert meta.name == "a"
        assert len(meta.commands) == 1
        command = a_chan.broker.get_command("zoo")
        # 实际执行的是 zoo.
        assert await command() == 123

    async with chan.bootstrap():
        meta = chan.broker.meta()
        assert meta.children == ["a"]


@pytest.mark.asyncio
async def test_py_channel_with_children() -> None:
    main = PyChannel(name="main")
    a_chan = PyChannel(name="a")
    b_chan = PyChannel(name="b")
    main.import_channels(a_chan, b_chan)
    c = PyChannel(name="c")
    d = PyChannel(name="d")
    c.import_channels(d)
    main.import_channels(c)

    channels = main.all_channels()
    assert len(channels) == 5
    assert channels[""] is main
    assert channels["c"] is c
    assert channels["c.d"] is c.children()["d"]
    assert c.get_channel("") is c
    assert c.get_channel("d") is c.children()["d"]
    assert main.get_channel("c.d") is c.children()["d"]


@pytest.mark.asyncio
async def test_py_channel_execute_task() -> None:
    main = PyChannel(name="main")

    async def foo() -> int:
        _t = ChannelCtx.task()
        _chan = ChannelCtx.channel()
        assert _t is not None
        assert _chan is not None
        return 123

    main.build.command()(foo)
    async with main.bootstrap() as broker:
        task = broker.create_command_task("foo")
        await broker.execute_task_soon(task)
        result = await task
        assert result == 123


@pytest.mark.asyncio
async def test_py_channel_desc_and_doc_with_ctx() -> None:
    main = PyChannel(name="main")

    def foo_doc() -> str:
        _chan = ChannelCtx.channel()
        return _chan.name()

    async def foo() -> int:
        _t = ChannelCtx.task()
        _chan = ChannelCtx.channel()
        assert _t is None
        assert _chan is not None
        return 123

    main.build.command(doc=foo_doc)(foo)
    async with main.bootstrap() as broker:
        _foo = broker.get_command("foo")
        r = await _foo()
        assert r == 123
        assert await _foo() == 123
        assert await _foo() == 123
        assert await _foo() == 123
        assert "main" in _foo.meta().interface


@pytest.mark.asyncio
async def test_py_channel_bind():
    class Foo:
        def __init__(self, val: int):
            self.val = val

    main = PyChannel(name="main")
    main.build.with_binding(Foo, Foo(123))

    @main.build.command()
    async def foo() -> int:
        _foo = ChannelCtx.get_contract(Foo)
        return _foo.val

    async with main.bootstrap() as broker:
        _foo = broker.get_command("foo")
        assert await _foo() == 123


@pytest.mark.asyncio
async def test_py_channel_context() -> None:
    main = PyChannel(name="main")

    messages = [new_text_message("hello", role="system")]

    def foo() -> list[Message]:
        return messages

    # 添加 context message 函数.
    main.build.context_messages(foo)

    async with main.bootstrap() as broker:
        # 启动时 meta 中包含了生成的 messages.
        meta = broker.meta()
        assert len(meta.context) == 1
        messages.append(new_text_message("world", role="system"))

        # 更新后, messages 也变更了.
        await broker.refresh_meta()
        assert len(broker.meta().context) == 2


@pytest.mark.asyncio
async def test_py_channel_exec_tasks() -> None:
    import asyncio
    main = PyChannel(name="main")

    _sleep = 0.0

    @main.build.command()
    async def foo() -> bool:
        await asyncio.sleep(_sleep)
        t = ChannelCtx.task()
        return t is not None

    async with main.bootstrap() as broker:
        task = broker.create_command_task("foo")
        await broker.execute_task_soon(task)
        assert await task
        task = broker.create_command_task("foo")
        await broker.execute_task_soon(task)
        assert await task
        task = broker.create_command_task("foo")
        await broker.execute_task_soon(task)
        assert await task

    async with main.bootstrap() as broker:
        _sleep = 2.0
        task1 = broker.create_command_task("foo")
        await broker.execute_task_soon(task1)
        assert not task1.done()
        await broker.clear_all()
        assert task1.done()
        assert task1.exception() is not None
        with pytest.raises(CommandError):
            await task1


@pytest.mark.asyncio
async def test_py_channel_idle() -> None:
    import asyncio
    main = PyChannel(name="main")

    idled = []

    @main.build.command()
    async def foo() -> bool:
        await asyncio.sleep(0.1)
        return True

    @main.build.idle
    async def idle() -> None:
        br = ChannelCtx.broker()
        if br:
            idled.append(1)

    async with main.bootstrap() as broker:
        task = broker.execute_command("foo")
        await task
        await broker.idle()
        await asyncio.sleep(0.0)
        task = broker.execute_command("foo")
        await task
        await broker.idle()
        await asyncio.sleep(0.0)
    assert len(idled) == 2


@pytest.mark.asyncio
async def test_py_channel_startup_and_close() -> None:
    main = PyChannel(name="main")

    @main.build.command()
    async def foo() -> bool:
        return True

    done = []

    @main.build.start_up
    @main.build.close
    async def count_running() -> None:
        _broker = ChannelCtx.broker()
        if _broker:
            done.append(1)

    async with main.bootstrap() as broker:
        task = broker.execute_command("foo")
        await task

    assert len(done) == 2


@pytest.mark.asyncio
async def test_py_channel_on_running_and_task_callback() -> None:
    main = PyChannel(name="main")

    @main.build.command()
    async def foo() -> bool:
        return True

    done = []

    @main.build.running
    async def count_tasks() -> None:
        _broker = ChannelCtx.broker()

        def add_done_tasks(_task: CommandTask) -> None:
            done.append(_task)

        _broker.on_task_done(add_done_tasks)
        await _broker.wait_closing()

    async with main.bootstrap() as broker:
        task = broker.execute_command("foo")
        await task
        await asyncio.sleep(0.0)
        task = broker.execute_command("foo")
        await task
    await asyncio.sleep(0.2)
    assert len(done) == 2


@pytest.mark.asyncio
async def test_py_channel_child_orders() -> None:
    main = PyChannel(name="main")
    a_chan = PyChannel(name="a_chan")
    b_chan = PyChannel(name="b_chan")
    c_chan = PyChannel(name="c_chan")
    d_chan = PyChannel(name="d_chan")
    e_chan = PyChannel(name="e_chan")
    main.import_channels(a_chan, b_chan)
    a_chan.import_channels(c_chan, d_chan)
    b_chan.import_channels(e_chan)

    # 深度优先排序.
    order = list(main.all_channels().values())
    assert order == [main, a_chan, c_chan, d_chan, b_chan, e_chan]
    # 运行第二次.
    order = list(main.all_channels().values())
    assert order == [main, a_chan, c_chan, d_chan, b_chan, e_chan]
