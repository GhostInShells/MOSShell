from ghoshell_moss.core.shell.primitives import wait
from ghoshell_moss.core.shell import new_shell
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

    shell = new_shell()
    shell.main_channel.import_channels(a_chan, b_chan)
    shell.main_channel.build.command()(wait)
    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<a:foo/><b:bar/><a:foo/>")
            interpreter.commit()
            await interpreter.wait_execution_done()
            # bar is later because sleep
            assert ordered == ['foo', 'foo', 'bar']

        ordered.clear()
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<wait><a:foo/><b:bar/></wait><a:foo/>")
            interpreter.commit()
            tasks = await interpreter.wait_execution_done()
            # bar is executed before second foo
            for t in tasks.values():
                assert t.success()
            assert ordered == ['foo', 'bar', 'foo']
