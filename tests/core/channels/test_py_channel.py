import pytest

from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.concepts.command import CommandTask, PyCommand
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
        assert available_mutator.available
        assert available_test_cmd.is_available() == available_mutator.available
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
        _t = CommandTask.get_from_context()
        _chan = Channel.get_from_context()
        assert _t is not None
        assert _chan is not None
        return 123

    main.build.command()(foo)
    async with main.bootstrap() as client:
        task = main.create_command_task("foo")
        result = await main.execute_task(task)
        assert result == 123


@pytest.mark.asyncio
async def test_py_channel_desc_and_doc_with_ctx() -> None:
    main = PyChannel(name="main")

    def foo_doc() -> str:
        _chan = Channel.get_from_context()
        return _chan.name()

    async def foo() -> int:
        _t = CommandTask.get_from_context()
        _chan = Channel.get_from_context()
        assert _t is not None
        assert _chan is not None
        return 123

    main.build.command(doc=foo_doc)(foo)
    async with main.bootstrap() as client:
        foo = main.broker.get_command("foo")
        assert "main" in foo.meta().interface


@pytest.mark.asyncio
async def test_py_channel_bind():
    class Foo:
        def __init__(self, val: int):
            self.val = val

    main = PyChannel(name="main")
    main.build.with_binding(Foo, Foo(123))

    @main.build.command()
    async def foo() -> int:
        _chan = Channel.get_from_context()
        foo = _chan.get_contract(Foo)
        return foo.val

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
