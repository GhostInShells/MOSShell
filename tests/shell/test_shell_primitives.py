from ghoshell_moss.core.shell.primitives import wait
from ghoshell_moss.core.shell import new_ctml_shell
from ghoshell_moss.core import PyChannel
import pytest
import asyncio


@pytest.mark.asyncio
async def test_wait_primitive():
    a_chan = PyChannel(name='a')
    b_chan = PyChannel(name='b')

    ordered = []

    @a_chan.build.command()
    @b_chan.build.command()
    async def foo():
        ordered.append('foo')
        return 123

    @b_chan.build.command()
    async def bar():
        await asyncio.sleep(0.2)
        ordered.append('bar')
        return 456

    shell = new_ctml_shell()
    shell.main_channel.import_channels(a_chan, b_chan)
    shell.main_channel.build.command()(wait)
    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<a:foo/><b:bar/><a:foo/>")
            interpreter.commit()
            await interpreter.wait_execution_done()
            # bar is later because sleep
            assert ordered == ['foo', 'foo', 'bar']

        # 验证添加了 wait 后改变了排序.
        ordered.clear()
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<wait><a:foo/><b:bar/></wait><a:foo/>")
            interpreter.commit()
            tasks = await interpreter.wait_execution_done()
            # bar is executed before second foo
            for t in tasks.values():
                assert t.success()
            assert ordered == ['foo', 'bar', 'foo']

        # 验证多组 wait
        ordered.clear()
        async with shell.interpreter_in_ctx() as interpreter:
            print(interpreter.moss_instruction())
            interpreter.feed("<wait><a:foo/><b:bar/></wait><wait><a:foo/><b:bar/></wait>")
            interpreter.commit()
            tasks = await interpreter.wait_execution_done()
            # bar is executed before second foo
            for t in tasks.values():
                assert t.success()
            assert ordered == ['foo', 'bar', 'foo', 'bar']

        # 验证 timeout
        ordered.clear()
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<wait timeout:float='0.1'><a:foo/><b:bar/><a:foo/><b:bar/></wait>")
            interpreter.commit()
            tasks = await interpreter.wait_execution_done()
            # 只有 foo 成功了. 其它的都被 timeout 了.
            assert ordered == ['foo', 'foo']
