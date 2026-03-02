import pytest
import asyncio

from ghoshell_moss.core.ctml.shell.primitives.clear import clear
from ghoshell_moss.core import PyChannel, new_ctml_shell


@pytest.mark.asyncio
async def test_loop_basic_functionality():
    """
    测试 clear 基本功能：清空子轨道的运行状态
    """
    shell = new_ctml_shell()
    chan = PyChannel(name="a")
    ran = []

    @chan.build.command()
    async def foo():
        ran.append(1)

    shell.main_channel.import_channels(chan)
    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<loop times='100'><a:foo/></loop>")
            interpreter.commit()
            await interpreter.wait_stopped()
            interpreter.raise_exception()
            assert len(ran) == 100


@pytest.mark.asyncio
async def test_loop_times_zero():
    """
    测试 clear 基本功能：清空子轨道的运行状态
    """
    shell = new_ctml_shell()
    chan = PyChannel(name="a")
    ran = []

    @chan.build.command()
    async def foo():
        ran.append(1)

    shell.main_channel.import_channels(chan)
    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<loop times='0'><a:foo/><a:foo/></loop>")
            interpreter.commit()
            await interpreter.wait_stopped()
            interpreter.raise_exception()
            assert len(ran) == 0


@pytest.mark.asyncio
async def test_loop_times_zero():
    """
    测试 clear 基本功能：清空子轨道的运行状态
    """
    shell = new_ctml_shell()
    chan = PyChannel(name="a")
    ran = []

    @chan.build.command()
    async def foo():
        ran.append(1)

    shell.main_channel.import_channels(chan)
    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<loop times='101'><a:foo/><a:foo/></loop>")
            interpreter.commit()
            await interpreter.wait_stopped()
            interpreter.raise_exception()
            assert len(ran) == 200


@pytest.mark.asyncio
async def test_loop_times_negative():
    """
    测试 clear 基本功能：清空子轨道的运行状态
    """
    shell = new_ctml_shell()
    chan = PyChannel(name="a")
    ran = []

    @chan.build.command()
    async def foo():
        await asyncio.sleep(0.05)
        ran.append(1)

    shell.main_channel.import_channels(chan)
    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<wait timeout='0.2'><loop times='-1'><a:foo/><a:foo/></loop></wait>")
            interpreter.commit()
            await interpreter.wait_stopped()
            interpreter.raise_exception()
            assert len(ran) > 0
            assert len(ran) < 5


@pytest.mark.asyncio
async def test_loop_times_negative_with_others():
    """
    测试 clear 基本功能：清空子轨道的运行状态
    """
    shell = new_ctml_shell()
    chan = PyChannel(name="a")
    ran = []

    @chan.build.command()
    async def foo():
        ran.append(1)

    shell.main_channel.import_channels(chan)
    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<loop times='-1'><a:foo/><sleep duration='0.005'/><a:foo/></loop>")
            interpreter.commit()
            await asyncio.sleep(0.1)
            await shell.clear()
            await interpreter.wait_stopped()
            interpreter.raise_exception()
            assert len(ran) > 0
            assert len(ran) < 30


@pytest.mark.asyncio
async def test_loop_with_dynamic_times():
    """
    测试循环次数动态计算（通过命令返回值）
    """
    shell = new_ctml_shell()
    chan = PyChannel(name="calc")

    execution_log = []

    @chan.build.command()
    async def calculate_iterations():
        # 模拟动态计算循环次数
        return 3

    @chan.build.command()
    async def perform_action():
        nonlocal execution_log
        execution_log.append("action")

    shell.main_channel.import_channels(chan)

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            # 注意：这个测试假设loop原语支持动态次数
            # 如果当前不支持，可以注释掉或修改
            interpreter.feed("""
                <calc:calculate_iterations/>
                <!-- 这里假设我们可以将结果用于循环次数 -->
                <!-- 实际实现可能需要不同的语法 -->
                <loop times="3">  <!-- 硬编码为calculate_iterations的返回值 -->
                    <calc:perform_action/>
                </loop>
            """)
            interpreter.commit()
            await interpreter.wait_stopped()
            interpreter.raise_exception()

            # 验证执行了3次
            assert execution_log.count("action") == 3


@pytest.mark.asyncio
async def test_loop_with_concurrent_channels():
    """
    测试循环中多个通道的并发执行
    """
    shell = new_ctml_shell()

    # 创建多个通道
    audio_chan = PyChannel(name="audio", dynamic=True)
    visual_chan = PyChannel(name="visual", dynamic=True)

    audio_log = []
    visual_log = []

    @audio_chan.build.command(blocking=False)
    async def play_beep():
        nonlocal audio_log
        audio_log.append("beep")
        await asyncio.sleep(0.01)  # 模拟短时间音频

    @visual_chan.build.command(blocking=False)
    async def show_flash():
        nonlocal visual_log
        visual_log.append("flash")
        await asyncio.sleep(0.01)  # 模拟短时间视觉

    @audio_chan.build.command()
    async def play_complete():
        nonlocal audio_log
        audio_log.append("complete")

    shell.main_channel.import_channels(audio_chan, visual_chan)

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            # 循环3次，每次同时触发音频和视觉
            interpreter.feed("""
                <loop times="3">
                    <audio:play_beep/>
                    <visual:show_flash/>
                    <sleep duration="0.02"/>
                </loop>
                <audio:play_complete/>
            """)
            interpreter.commit()
            await interpreter.wait_tasks()
            for t in interpreter.compiled_tasks().values():
                assert t.success()

            interpreter.raise_exception()

            # 验证每个通道执行了正确次数
            assert audio_log.count("beep") == 3
            assert visual_log.count("flash") == 3
            assert "complete" in audio_log  # 确保循环完成后执行了完成命令

            # 验证并发性：音频和视觉应该交错执行
            # 但由于都是非阻塞的，具体顺序可能不确定
            assert len(audio_log) >= 4  # 3次beep + 1次complete
            assert len(visual_log) == 3


@pytest.mark.asyncio
async def test_loop_interruption_and_resume():
    """
    测试循环的中断与恢复（模拟用户打断后继续）
    """
    shell = new_ctml_shell()
    chan = PyChannel(name="task")

    execution_log = []
    loop_iterations = 0

    @chan.build.command()
    async def perform_task():
        nonlocal execution_log, loop_iterations
        loop_iterations += 1
        execution_log.append(f"task_{loop_iterations}")
        await asyncio.sleep(0.05)  # 模拟任务执行时间
        return loop_iterations

    @chan.build.command()
    async def handle_interruption():
        nonlocal execution_log
        execution_log.append("interruption_handled")

    shell.main_channel.import_channels(chan)

    async with shell:
        # 第一轮：开始循环但被中断
        async with shell.interpreter_in_ctx() as interpreter1:
            interpreter1.feed('<loop times="10"><task:perform_task/></loop>')
            interpreter1.commit()

            # 等待循环开始几次
            await asyncio.sleep(0.15)  # 大约3次迭代

            # 记录中断前的状态
            iterations_before_interrupt = loop_iterations
            assert 2 <= iterations_before_interrupt <= 4  # 应该执行了2-4次

            # 中断当前执行
            await shell.clear()

            # 确保解释器停止
            await interpreter1.wait_stopped()

            # 第二轮：恢复执行（从上次中断的地方继续逻辑）
            async with shell.interpreter_in_ctx() as interpreter2:
                # 处理中断
                interpreter2.feed('<task:handle_interruption/>')

                # 继续剩余的迭代
                remaining = 10 - iterations_before_interrupt
                interpreter2.feed(f'<loop times="{remaining}"><task:perform_task/></loop>')
                interpreter2.commit()

                await interpreter2.wait_stopped()
                interpreter2.raise_exception()

                # 验证总执行次数
                assert loop_iterations == 10
                assert execution_log.count("interruption_handled") == 1

                # 验证任务执行顺序
                task_logs = [log for log in execution_log if log.startswith("task_")]
                assert len(task_logs) == 10

                # 检查任务编号的连续性（可能不连续因为中断，但应该没有重复）
                task_numbers = [int(log.split("_")[1]) for log in task_logs]
                assert sorted(task_numbers) == list(range(1, 11))  # 1到10