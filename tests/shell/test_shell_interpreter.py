import pytest
from ghoshell_moss.core import PyChannel, new_ctml_shell, InterpretError
import time


@pytest.mark.asyncio
async def test_run_not_exists_command():
    """
    测试 wait_idle 与其他原语的配合
    """
    shell = new_ctml_shell()
    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            # 复杂场景：启动后台任务，sleep，然后 wait_idle
            interpreter.feed("""
                <bg:background_work/>
                <sleep duration:float="0.1"/>
            """)
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            with pytest.raises(Exception):
                interpreter.raise_exception()

            interpretation = interpreter.interpretation()
        assert len(interpretation.exception) > 0


@pytest.mark.asyncio
async def test_interpreter_parse_error():
    """
    测试 wait_idle 与其他原语的配合
    """
    shell = new_ctml_shell()
    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpretation = interpreter.interpretation()
            # 复杂场景：启动后台任务，sleep，然后 wait_idle
            interpreter.feed("""
                <bg:background_work/>
                <sleep duration:floa
            """)
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            with pytest.raises(Exception):
                interpreter.raise_exception()

        assert len(interpretation.exception) > 0


@pytest.mark.asyncio
async def test_interpreter_feed_stop_by_error():
    """
    测试 wait_idle 与其他原语的配合
    """
    shell = new_ctml_shell()
    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            # 复杂场景：启动后台任务，sleep，然后 wait_idle
            interpreter.feed("""
                <bg:background_work/>
                <sleep duration:floa
            """)
            interpreter.feed("<<<<skskdkjfskd")
            with pytest.raises(InterpretError):
                await interpreter.wait_compiled()

            assert interpreter.exception() is not None
            assert interpreter.is_stopped()
            assert not interpreter.is_closed()
            with pytest.raises(InterpretError):
                interpreter.feed("<<<<skskdkjfskd", throw=True)

            interpretation = await interpreter.close()
            assert len(interpretation.exception) > 0


@pytest.mark.asyncio
async def test_run_shell_concurrent():
    shell = new_ctml_shell()

    started_at = []

    async def foo():
        started_at.append(time.time())
        return

    # 20 个解析并发, 期待能达到 20hz 精度.
    # 达不到这个精度的是计算性能不太行.
    # 实际链路中, 链路延时可能有 10~1000ms. 所以 python asyncio task 的延时是可以忽略.
    n = 20

    for i in range(n):
        chan = PyChannel(name=f"chan{i}")
        chan.build.command()(foo)
        shell.main_channel.import_channels(chan)

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            content = ""
            for i in range(n):
                content += f"<chan{i}:foo/>"
            # 虽然是一次提交, 但是 xml parser 也有延时.
            interpreter.feed(content)
            interpreter.commit()
            await interpreter.wait_stopped()
    assert len(started_at) == n
    first = started_at[0]
    total_gap = 0.0
    for t in started_at:
        total_gap += abs(t - first)
    even_gap = total_gap / n
    # 期待能达到 20hz 的同步精度.
    assert even_gap < 0.05
