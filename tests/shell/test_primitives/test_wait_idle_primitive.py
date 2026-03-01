import pytest
import asyncio
import contextlib

from ghoshell_moss.core.ctml.shell.primitives.wait_idle import wait_idle
from ghoshell_moss.core import PyChannel, new_ctml_shell


@pytest.mark.asyncio
async def test_wait_idle_basic():
    """
    测试 wait_idle 基本功能：等待子轨道任务完成
    """
    # 创建子 Channel
    child_chan = PyChannel(name="child", dynamic=True)

    # 记录执行状态
    execution_log = []

    @child_chan.build.command()
    async def long_task():
        nonlocal execution_log
        execution_log.append("task_started")
        await asyncio.sleep(0.1)  # 模拟长时间任务
        execution_log.append("task_completed")
        return "done"

    shell = new_ctml_shell()
    shell.main_channel.import_channels(child_chan)
    # 动态 Channel 会自动注册到主 Channel
    # 不需要手动注册 wait_idle，因为它已经是原语

    async with shell:
        # 创建解释器
        async with shell.interpreter_in_ctx() as interpreter:
            # 启动子轨道任务
            interpreter.feed("<child:long_task/><wait_idle/>")
            interpreter.commit()

            # 等待执行完成
            tasks = await interpreter.wait()

            # 验证任务已完成
            assert "task_started" in execution_log
            assert "task_completed" in execution_log


@pytest.mark.asyncio
async def test_wait_idle_with_timeout():
    """
    测试 wait_idle 超时功能
    """
    child_chan = PyChannel(name="child", dynamic=True)

    execution_log = []
    task_cancelled = False

    @child_chan.build.command()
    async def very_long_task():
        nonlocal execution_log, task_cancelled
        execution_log.append("task_started")
        try:
            await asyncio.sleep(10.0)  # 非常长的任务
            execution_log.append("task_completed")
        except asyncio.CancelledError:
            task_cancelled = True
            execution_log.append("task_cancelled")
            raise

    shell = new_ctml_shell()
    shell.main_channel.import_channels(child_chan)

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            # 启动非常长的任务
            interpreter.feed("<child:very_long_task/>")

            # 调用 wait_idle 并设置短超时
            interpreter.feed('<wait_idle timeout:float="0.1"/>')  # 100ms 超时
            interpreter.commit()

            tasks = await interpreter.wait()

            # 任务应该被取消
            assert task_cancelled
            assert "task_started" in execution_log
            assert "task_cancelled" in execution_log
            assert "task_completed" not in execution_log


@pytest.mark.asyncio
async def test_wait_idle_specific_channel():
    """
    测试等待特定轨道
    """
    # 创建多个 Channel
    audio_chan = PyChannel(name="audio", dynamic=True)
    video_chan = PyChannel(name="video", dynamic=True)

    # 记录各 Channel 任务状态
    audio_done = False
    video_done = False

    @audio_chan.build.command()
    async def audio_task():
        nonlocal audio_done
        await asyncio.sleep(0.1)
        audio_done = True
        return "audio"

    @video_chan.build.command()
    async def video_task():
        nonlocal video_done
        await asyncio.sleep(0.3)
        video_done = True
        return "video"

    shell = new_ctml_shell()
    shell.main_channel.import_channels(audio_chan, video_chan)
    expect = False

    @shell.main_channel.build.command()
    async def audio_done_but_video_not():
        nonlocal audio_done, video_done, expect
        expect = audio_done and not video_done

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            # 在两个子轨道上启动任务
            interpreter.feed("<audio:audio_task/><video:video_task/>")
            # 只等待 audio 轨道
            interpreter.feed('<wait_idle chan="audio"/><audio_done_but_video_not/>')
            interpreter.commit()

            tasks = await interpreter.wait()
            assert expect


@pytest.mark.asyncio
async def test_wait_idle_recursive():
    """
    测试 wait_idle 的递归等待：等待子轨道及其子轨道
    """
    # 创建多层 Channel 结构
    level1_chan = PyChannel(name="level1", dynamic=True)
    level2_chan = PyChannel(name="level2", dynamic=True)

    execution_order = []

    @level1_chan.build.command()
    async def level1_task():
        nonlocal execution_order
        execution_order.append("level1_start")
        # 启动 level2 任务
        await level2_task()
        await asyncio.sleep(0.1)
        execution_order.append("level1_end")

    @level2_chan.build.command()
    async def level2_task():
        nonlocal execution_order
        execution_order.append("level2_start")
        await asyncio.sleep(0.2)
        execution_order.append("level2_end")

    shell = new_ctml_shell()
    shell.main_channel.import_channels(level1_chan, level2_chan)

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            # 启动多层任务
            interpreter.feed("<wait_idle/><level1:level1_task/><wait_idle/>")
            interpreter.commit()

            tasks = await interpreter.wait()

            # 验证执行顺序
            assert execution_order == [
                "level1_start",
                "level2_start",
                "level2_end",
                "level1_end"
            ]


@pytest.mark.asyncio
async def test_wait_idle_with_empty_channels():
    """
    测试空轨道的 wait_idle
    """
    shell = new_ctml_shell()

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            # 在没有子任务的情况下调用 wait_idle
            interpreter.feed("<wait_idle/>")
            interpreter.commit()

            tasks = await interpreter.wait()

            # 应该正常完成，不抛出异常
            assert len(tasks) == 1
            wait_idle_task = list(tasks.values())[0]
            assert wait_idle_task.success()


@pytest.mark.asyncio
async def test_wait_idle_negative_timeout():
    """
    测试负超时值的错误处理
    """
    shell = new_ctml_shell()

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            # 负超时应该抛出错误
            interpreter.feed('<wait_idle timeout:float="-1.0"/>')
            interpreter.commit()

            tasks = await interpreter.wait()

            # 任务应该失败
            assert len(tasks) == 1
            wait_idle_task = list(tasks.values())[0]
            assert not wait_idle_task.success()
            # 应该包含错误信息


@pytest.mark.asyncio
async def test_wait_idle_with_other_primitives():
    """
    测试 wait_idle 与其他原语的配合
    """
    shell = new_ctml_shell()

    # 创建动态 Channel
    bg_chan = PyChannel(name="bg", dynamic=True)

    execution_log = []

    @bg_chan.build.command(blocking=False)
    async def background_work():
        nonlocal execution_log
        execution_log.append("bg_start")
        await asyncio.sleep(0.3)
        execution_log.append("bg_end")

    @bg_chan.build.command(blocking=True)
    async def run_after_idle():
        execution_log.append("run_after_idle")

    shell.main_channel.import_channels(bg_chan)

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            # 复杂场景：启动后台任务，sleep，然后 wait_idle
            interpreter.feed("""
                <bg:background_work/>
                <sleep duration:float="0.05"/>
            """)
            interpreter.feed("<wait_idle/><bg:run_after_idle/>")
            interpreter.commit()
            await asyncio.sleep(0.1)
            # sleep 应该结束了.
            assert "bg_start" in execution_log
            assert "bg_end" not in execution_log
            assert "run_after_idle" in execution_log
            assert interpreter.is_running()
            tasks = await interpreter.wait()
            assert "bg_end" in execution_log


@pytest.mark.asyncio
async def test_wait_idle_zero_timeout():
    """
    测试零超时：应该立即清空
    """
    child_chan = PyChannel(name="child", dynamic=True)

    task_cancelled = False

    @child_chan.build.command()
    async def task():
        nonlocal task_cancelled
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            task_cancelled = True
            raise

    shell = new_ctml_shell()
    shell.main_channel.import_channels(child_chan)

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            # 启动任务
            interpreter.feed("<child:task/>")
            interpreter.feed('<wait_idle timeout:float="0.0"/>')
            interpreter.commit()

            tasks = await interpreter.wait()

            # 任务应该被取消
            assert task_cancelled
