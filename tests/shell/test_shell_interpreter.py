import pytest
import asyncio
import contextlib

from ghoshell_moss.core.ctml.shell.primitives.wait_idle import wait_idle
from ghoshell_moss.core import PyChannel, new_ctml_shell, InterpretError


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
            tasks = await interpreter.wait()
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
            # 复杂场景：启动后台任务，sleep，然后 wait_idle
            interpreter.feed("""
                <bg:background_work/>
                <sleep duration:floa
            """)
            interpreter.commit()
            tasks = await interpreter.wait()
            with pytest.raises(Exception):
                interpreter.raise_exception()

            interpretation = interpreter.interpretation()
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
