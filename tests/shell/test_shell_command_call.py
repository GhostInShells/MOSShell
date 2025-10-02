import asyncio

import pytest

from ghoshell_moss.concepts.interpreter import Interpreter


@pytest.mark.asyncio
async def test_shell_execution_baseline():
    from ghoshell_moss.shell import new_shell

    shell = new_shell()
    a_chan = shell.main_channel.new_child('a')
    b_chan = shell.main_channel.new_child('b')
    return

    @a_chan.build.command()
    async def foo() -> int:
        return 123

    @b_chan.build.command()
    async def bar() -> int:
        await asyncio.sleep(0.01)
        return 456

    async with shell:
        interpreter = shell.interpret()
        assert isinstance(interpreter, Interpreter)
        async with interpreter:
            interpreter.feed("<a:foo /><b:bar />")
            tasks = await interpreter.wait_execution_done(10)

            assert len(tasks) == 2
            result = []
            for task in tasks.values():
                assert task.success()
                result.append(task.result)
            # 获取到结果.
            assert result == [123, 456]
            assert ['a', 'b'] == [t.exec_chan for t in tasks.values()]
