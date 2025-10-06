from ghoshell_moss.channels.py_channel import PyChannel
from ghoshell_moss.concepts.command import PyCommand
import pytest

chan = PyChannel(name="test")


@chan.build.command()
def add(a: int, b: int) -> int:
    """测试一个同步函数是否能正确被调用. """
    return a + b


@chan.build.with_description()
def desc() -> str:
    return "hello world"


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
    async with chan.bootstrap() as client:
        assert chan.name() == "test"

        # commands 存在.
        commands = list(client.commands().values())
        assert len(commands) > 0

        # 所有的命令应该都以 channel 开头.
        for command in commands:
            assert command.meta().chan == "test"

        # 不用全名来获取函数.
        foo_cmd = client.get_command("foo")
        assert foo_cmd is not None
        assert await foo_cmd() == 9527

        # 测试名称有效.
        help_cmd = client.get_command("help")
        assert help_cmd is not None
        assert await help_cmd() == "help"

        # 测试乱取拿不到东西
        none_cmd = client.get_command("never_exists_command")
        assert none_cmd is None
        # full name 不正确也拿不到.
        help_cmd = client.get_command("help")
        assert help_cmd is not None

        # available 测试.
        available_test_cmd = client.get_command("available_test_fn")
        assert available_test_cmd is not None
        assert available_mutator.available
        assert available_test_cmd.is_available() == available_mutator.available
        available_mutator.available = False
        assert available_test_cmd.is_available() == available_mutator.available

        # description 测试.
        meta = client.meta()
        assert meta.description == desc()


@pytest.mark.asyncio
async def test_py_channel_children() -> None:
    assert len(chan.children()) == 0

    a_chan = chan.new_child("a")
    assert isinstance(a_chan, PyChannel)
    assert chan.children()['a'] is a_chan

    async def zoo():
        return 123

    zoo_cmd = a_chan.build.command(return_command=True)(zoo)
    assert isinstance(zoo_cmd, PyCommand)

    async with a_chan.bootstrap():
        meta = a_chan.client.meta()
        assert meta.name == "a"
        assert len(meta.commands) == 1
        command = a_chan.client.get_command('zoo')
        # 实际执行的是 zoo.
        assert await command() == 123

    async with chan.bootstrap():
        meta = chan.client.meta()
        assert meta.children == ['a']
