import pytest
import asyncio

from ghoshell_moss.core.ctml.shell.primitives.clear import clear
from ghoshell_moss.core import PyChannel, new_ctml_shell


@pytest.mark.asyncio
async def test_clear_basic_functionality():
    """
    测试 clear 基本功能：清空子轨道的运行状态
    """
    # 创建父 Channel 和子 Channel
    parent_chan = PyChannel(name="parent")
    child_chan = PyChannel(name="child", dynamic=True)

    # 记录执行状态
    execution_log = []
    task_cancelled = False
    cmd_done = asyncio.Event()

    @child_chan.build.command()
    async def long_running_task():
        nonlocal task_cancelled
        execution_log.append("task_started")
        try:
            await asyncio.sleep(1.0)  # 长时间运行的任务
            execution_log.append("task_completed")
        except asyncio.CancelledError:
            task_cancelled = True
            execution_log.append("task_cancelled")
            raise
        except Exception as e:
            raise
        finally:
            cmd_done.set()

    shell = new_ctml_shell()
    shell.main_channel.import_channels(parent_chan, child_chan)
    shell.main_channel.build.command()(clear)

    async with shell:
        # 启动子 Channel 上的长时间任务
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<child:long_running_task/><clear/>")
            interpreter.commit()
            # 验证任务被取消
            await cmd_done.wait()
            assert task_cancelled
            assert "task_cancelled" in execution_log
            assert "task_completed" not in execution_log


@pytest.mark.asyncio
async def test_clear_specific_channel():
    """
    测试在指定 Channel 上清空
    """
    # 创建多个 Channel
    main_chan = PyChannel(name="main")
    audio_chan = PyChannel(name="audio", dynamic=True)
    video_chan = PyChannel(name="video", dynamic=True)

    # 记录各 Channel 任务状态
    audio_cancelled = False
    video_cancelled = False

    @audio_chan.build.command()
    async def audio_task():
        nonlocal audio_cancelled
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            audio_cancelled = True
        except Exception as e:
            raise

    @video_chan.build.command()
    async def video_task():
        nonlocal video_cancelled
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            video_cancelled = True
        except Exception as e:
            raise

    shell = new_ctml_shell()
    shell.main_channel.import_channels(main_chan, audio_chan, video_chan)
    shell.main_channel.build.command()(clear)

    async with shell:
        # 在 audio 和 video Channel 上启动任务
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<audio:audio_task/><video:video_task/>")
            interpreter.feed("<clear chan='audio'/>")
            interpreter.commit()
            # 验证只有 audio 任务被取消
            await interpreter.wait()
            assert not video_cancelled  # video 任务应该还在运行
            assert audio_cancelled


@pytest.mark.asyncio
async def test_clear_recursive():
    """
    测试 clear 的递归清空功能
    """
    # 创建多层 Channel 结构
    root_chan = PyChannel(name="root")
    level1_chan = PyChannel(name="level1", dynamic=True)
    level2_chan = PyChannel(name="level2", dynamic=True)

    # 记录各层任务状态
    level1_cancelled = False
    level2_cancelled = False

    @level1_chan.build.command()
    async def level1_task():
        nonlocal level1_cancelled
        try:
            # 在 level1 任务中启动 level2 任务
            await level2_task()
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            level1_cancelled = True
            raise

    @level2_chan.build.command()
    async def level2_task():
        nonlocal level2_cancelled
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            level2_cancelled = True
            raise

    shell = new_ctml_shell()
    shell.main_channel.import_channels(root_chan, level1_chan, level2_chan)
    shell.main_channel.build.command()(clear)

    async with shell:
        # 启动多层任务
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<level1:level1_task/>")
            # 在根 Channel 调用 clear，应该递归清空所有子 Channel
            interpreter.feed("<clear/>")
            interpreter.commit()
            await interpreter.wait()
            # 验证所有层级的任务都被取消
            assert level1_cancelled
            assert level2_cancelled


@pytest.mark.asyncio
async def test_clear_with_wait_and_sleep():
    """
    测试 clear 与 wait、sleep 原语的配合
    """
    shell = new_ctml_shell()

    # 注册所有原语
    from ghoshell_moss.core.ctml.shell.primitives.wait import wait
    from ghoshell_moss.core.ctml.shell.primitives.sleep import sleep

    shell.main_channel.build.command()(clear)
    shell.main_channel.build.command()(wait)
    shell.main_channel.build.command()(sleep)

    # 创建一个动态 Channel 用于测试
    bg_chan = PyChannel(name="bg", dynamic=True)

    execution_log = []

    @bg_chan.build.command()
    async def background_task():
        execution_log.append("bg_start")
        try:
            await asyncio.sleep(0.5)
            execution_log.append("bg_end")
        except asyncio.CancelledError:
            execution_log.append("bg_cancelled")
            raise

    shell.main_channel.import_channels(bg_chan)

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            # 启动后台任务，然后 sleep，再 clear
            interpreter.feed("""
                <bg:background_task/>
                <sleep duration="0.1"/>
                <clear/>
            """)
            interpreter.commit()

            tasks = await interpreter.wait()

            # 验证执行顺序
            assert execution_log == ["bg_start", "bg_cancelled"]
            # bg_end 不应该出现，因为被 clear 了


@pytest.mark.asyncio
async def test_clear_empty_channels():
    """
    测试清空空 Channel（无子轨道）
    """
    shell = new_ctml_shell()
    shell.main_channel.build.command()(clear)

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            # 在没有任何子任务的情况下调用 clear
            interpreter.feed("<clear/>")
            interpreter.commit()

            tasks = await interpreter.wait()

            # 应该正常完成，不抛出异常
            assert len(tasks) == 1
            clear_task = list(tasks.values())[0]
            assert clear_task.success()


@pytest.mark.asyncio
async def test_clear_in_ctml_complex_scenario():
    """
    测试复杂场景：在 CTML 流中适时使用 clear
    """
    shell = new_ctml_shell()

    # 注册原语
    from ghoshell_moss.core.ctml.shell.primitives.wait import wait
    from ghoshell_moss.core.ctml.shell.primitives.sleep import sleep

    shell.main_channel.build.command()(clear)
    shell.main_channel.build.command()(wait)
    shell.main_channel.build.command()(sleep)

    # 创建多个动态 Channel
    music_chan = PyChannel(name="music", dynamic=True)
    effects_chan = PyChannel(name="effects", dynamic=True)

    execution_log = []

    @music_chan.build.command()
    async def play_music():
        execution_log.append("music_start")
        try:
            await asyncio.sleep(10.0)  # 长时间播放
            execution_log.append("music_end")
        except asyncio.CancelledError:
            execution_log.append("music_cancelled")
            raise

    @effects_chan.build.command()
    async def play_effect():
        execution_log.append("effect_start")
        try:
            await asyncio.sleep(0.3)  # 短时间音效
            execution_log.append("effect_end")
        except asyncio.CancelledError:
            execution_log.append("effect_cancelled")
            raise

    shell.main_channel.import_channels(music_chan, effects_chan)

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            # 模拟一个交互场景：播放背景音乐，播放音效，等待音效完成，然后清除所有
            interpreter.feed("""
                <music:play_music/>
                <effects:play_effect/>
                <wait timeout="0.4">
                    <sleep duration="0.35"/>
                </wait>
                <clear/>
            """)
            interpreter.commit()

            tasks = await interpreter.wait()

            # 验证执行顺序
            # 音乐和音效应该都启动了
            assert "music_start" in execution_log
            assert "effect_start" in execution_log

            # 音效可能完成也可能被取消（取决于时间）
            # 但音乐应该被取消了
            assert "music_cancelled" in execution_log
            assert "music_end" not in execution_log
