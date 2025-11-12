import time
from typing import List
import asyncio

import pytest

from ghoshell_moss import Interpreter, Channel, CommandTask, MOSSShell, CommandTaskStack


@pytest.mark.asyncio
async def test_shell_execution_baseline():
    from ghoshell_moss.shell import new_shell

    shell = new_shell()
    a_chan = shell.main_channel.new_child('a')
    b_chan = shell.main_channel.new_child('b')

    @a_chan.build.command()
    async def foo() -> int:
        return 123

    @b_chan.build.command()
    async def bar() -> int:
        # 晚执行 0.1 秒.
        await asyncio.sleep(0.1)
        return 456

    async with shell:
        interpreter = await shell.interpreter()
        assert isinstance(interpreter, Interpreter)
        assert shell.is_running()
        foo_cmd = await shell.get_command("a", "foo")
        assert foo_cmd is not None
        async with interpreter:
            interpreter.feed("<a:foo /><b:bar />")
            assert shell.is_running()
            tasks = await interpreter.wait_execution_done(1)

            assert len(tasks) == 2
            result = []
            for task in tasks.values():
                assert task.success()
                result.append(task.result())
            # 获取到结果.
            assert result == [123, 456]
            assert ['a', 'b'] == [t.exec_chan for t in tasks.values()]
            # 验证并发执行.
            task_list = list(tasks.values())
            # 两个任务几乎同时启动.
            running_gap = abs(task_list[0].trace.get('running') - task_list[1].trace.get('running'))
            assert running_gap < 0.01
            done_gap = abs(task_list[1].trace.get('done') - task_list[0].trace.get('done'))
            assert done_gap > 0.05


@pytest.mark.asyncio
async def test_shell_outputted():
    from ghoshell_moss.shell import new_shell

    shell = new_shell()

    @shell.main_channel.build.command()
    async def foo() -> int:
        return 123

    async with shell:
        foo_cmd = await shell.get_command("", "foo")
        assert foo_cmd is not None
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<foo />hello")
            tasks = await interpreter.wait_execution_done(10)
            task_list = list(tasks.values())
            assert len(tasks) == 2
            assert task_list[0].result() == 123
            assert interpreter.outputted() == ["hello"]


@pytest.mark.asyncio
async def test_shell_command_run_in_order():
    """测试 get command exec in chan 可以使命令进入 channel 队列有序执行. """
    from ghoshell_moss.shell import new_shell

    shell = new_shell()

    order = {}

    async def foo(i: float):
        await asyncio.sleep(i)
        order[i] = time.time()
        return i

    # register the foo command
    shell.main_channel.build.command(block=True)(foo)

    async with shell:
        # get the origin command
        foo_cmd: foo = await shell.get_command("", "foo")
        assert foo_cmd is not None

        values = await asyncio.gather(foo_cmd(0.2), foo_cmd(0.1))
        assert values == [0.2, 0.1]
        assert len(order) == 2
        # the command execute in concurrent
        assert order[0.1] > order[0.2]

        # 重新开始.
        order.clear()
        foo_cmd: foo = await shell.get_command("", "foo", exec_in_chan=True)
        values = await asyncio.gather(foo_cmd(0.2), foo_cmd(0.1))
        # the gather order is the same
        assert values == [0.2, 0.1]
        assert len(order) == 2
        # second command execute after first one
        assert order[0.1] > order[0.2]


@pytest.mark.asyncio
async def test_shell_task_can_get_channel():
    from ghoshell_moss.shell import new_shell

    shell = new_shell()
    a_chan = shell.main_channel.new_child('a')

    @a_chan.build.command()
    async def foo() -> bool:
        # 可以在运行时获取到 channel 本体.
        chan = Channel.get_from_context()
        return chan is a_chan

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<a:foo />")
            tasks = await interpreter.wait_execution_done(10)
            assert len(tasks) == 1
            assert list(tasks.values())[0].result() is True


@pytest.mark.asyncio
async def test_shell_task_can_get_task():
    from ghoshell_moss.shell import new_shell

    shell = new_shell()
    a_chan = shell.main_channel.new_child('a')

    @a_chan.build.command()
    async def foo() -> str:
        # 可以在运行时获取到 channel 本体.
        task = CommandTask.get_from_context()
        return task.cid

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<a:foo />")
            tasks = await interpreter.wait_execution_done(10)
            assert len(tasks) == 1
            first = list(tasks.values())[0]
            assert first.cid == first.result()


@pytest.mark.asyncio
async def test_shell_loop():
    from ghoshell_moss.shell import new_shell

    shell = new_shell()
    a_chan = shell.main_channel.new_child('a')

    @shell.main_channel.build.command()
    async def loop(times: int, tokens__):
        if times == 0:
            return None

        chan = Channel.get_from_context()
        # get shell from channel's container
        _shell = chan.broker.container.get(MOSSShell)
        _tasks = []
        async for t in _shell.parse_tokens_to_command_tasks(tokens__):
            _tasks.append(t)

        async def _iter():
            for i in range(times):
                for _task in _tasks:
                    yield _task.copy()

        async def on_success(generated: List[CommandTask]):
            await asyncio.gather(*[g.wait() for g in generated])

        return CommandTaskStack(_iter(), on_success)

    outputs = []

    @a_chan.build.command()
    async def foo() -> int:
        outputs.append(1)
        return 123

    content = '<loop times="2">hello<a:foo />world</loop>'
    async with shell:
        interpreter = await shell.interpreter()
        async with interpreter:
            for c in content:
                interpreter.feed(c)
            tasks = await interpreter.wait_execution_done()
            for task in tasks.values():
                assert task.done()
        assert interpreter.is_stopped()
    # 执行了两次.
    assert len(outputs) == 2


@pytest.mark.asyncio
async def test_shell_clear():
    from ghoshell_moss.shell import new_shell

    shell = new_shell()
    a_chan = shell.main_channel.new_child('a')
    b_chan = shell.main_channel.new_child('b')
    c_chan = a_chan.new_child('c')

    sleep = [0.1]

    @a_chan.build.command()
    async def foo() -> str:
        await asyncio.sleep(sleep[0])
        return "foo"

    @b_chan.build.command()
    async def bar() -> str:
        await asyncio.sleep(sleep[0])
        return "bar"

    @c_chan.build.command()
    async def baz() -> str:
        await asyncio.sleep(sleep[0])
        return "baz"

    content = '<a:foo /><b:bar /><a.c:baz />'
    async with shell:
        # baseline
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed(content)
            tasks = await interpreter.wait_execution_done()
            assert len(tasks) == 3
            assert [t.result() for t in tasks.values()] == ["foo", "bar", "baz"]

        # clear
        sleep[0] = 10
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed(content)
            await interpreter.wait_parse_done()
            parsed_tasks = interpreter.parsed_tasks()
            for t in parsed_tasks.values():
                assert not t.done()
            # clear all
            await shell.clear()
            parsed_tasks = interpreter.parsed_tasks()
            for t in parsed_tasks.values():
                assert t.cancelled()
