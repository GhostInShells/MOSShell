import pytest

from ghoshell_moss.core.ctml.shell.primitives.condition import branch
from ghoshell_moss.core import PyChannel, new_ctml_shell


@pytest.mark.asyncio
async def test_condition_basic_functionality():
    """
    测试 clear 基本功能：清空子轨道的运行状态
    """
    # 创建父 Channel 和子 Channel
    chan = PyChannel(name="chan")

    done = []

    @chan.build.command()
    async def check() -> bool:
        return True

    @chan.build.command()
    async def foo():
        done.append('foo')

    @chan.build.command()
    async def bar():
        done.append('bar')

    shell = new_ctml_shell()
    shell.main_channel.import_channels(chan)
    shell.main_channel.build.command()(branch)

    async with shell:
        # 启动子 Channel 上的长时间任务
        async with shell.interpreter_in_ctx() as interpreter:
            for msg in interpreter.instruction_messages():
                print(msg)
            interpreter.feed("<branch><chan:check/><chan:foo/><chan:bar/></branch>")
            interpreter.commit()
            # 验证任务被取消
            await interpreter.wait_stopped()
            assert done == ['foo']
