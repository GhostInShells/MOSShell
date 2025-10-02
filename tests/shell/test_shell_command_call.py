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
        interpreter = shell.interpret()
        assert isinstance(interpreter, Interpreter)
        assert shell.is_running()
        async with interpreter:
            interpreter.feed("<a:foo /><b:bar />")
            assert shell.is_running()
            tasks = await interpreter.wait_execution_done(10)

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
        async with shell.interpret() as interpreter:
            interpreter.feed("<foo />hello")
            await interpreter.wait_execution_done(10)
            assert interpreter.outputted() == ["hello"]


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
        async with shell.interpret() as interpreter:
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
        async with shell.interpret() as interpreter:
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
        _shell = chan.client.container.get(MOSSShell)
        tasks = []
        async for t in _shell.parse_tokens_to_tasks(tokens__):
            tasks.append(t)

        async def _iter():
            for i in range(times):
                for task in tasks:
                    yield task.copy()

        async def on_success(generated: List[CommandTask]):
            await asyncio.gather(*[g.wait() for g in generated])

        return CommandTaskStack(_iter(), on_success)

    outputs = []

    @a_chan.build.command()
    async def foo() -> int:
        outputs.append(1)
        return 123

    content = '<loop times="2"><a:foo /></loop>'
    async with shell:
        async with shell.interpret() as interpreter:
            for c in content:
                interpreter.feed(c)
            await interpreter.wait_execution_done()
    # 执行了两次.
    assert len(outputs) == 2
