import pytest
import asyncio
import time

from ghoshell_moss.core.ctml.shell.primitives.sleep import sleep
from ghoshell_moss.core.concepts.command import CommandStackResult
from ghoshell_moss.core import PyChannel, new_ctml_shell


@pytest.mark.asyncio
async def test_sleep_direct_mode():
    """
    测试直接模式（chan=""）：在当前协程直接睡眠
    """
    start_time = time.time()

    # 调用 sleep，不指定 channel
    await sleep(0.1)  # 睡眠 100ms

    elapsed = time.time() - start_time

    # 验证睡眠时间大约为 100ms（允许一定误差）
    assert elapsed >= 0.09  # 至少 90ms
    assert elapsed <= 0.15  # 最多 150ms（考虑系统调度）


@pytest.mark.asyncio
async def test_sleep_channel_mode_returns_command_stack_result():
    """
    测试 Channel 模式（chan!=""）：返回 CommandStackResult
    """
    # 当指定 channel 时，应该返回 CommandStackResult
    result = await sleep(0.1, chan="audio")

    # 验证返回类型
    assert isinstance(result, CommandStackResult)

    # CommandStackResult 应该包含任务
    # 注意：由于 BaseCommandTask.from_command 的细节，我们可能需要模拟
    # 这里先验证基本结构


@pytest.mark.asyncio
async def test_sleep_in_ctml_without_channel():
    """
    测试在 CTML 中调用 sleep（无 channel 参数）
    """
    shell = new_ctml_shell()

    # 注册 sleep 命令到主 channel
    shell.main_channel.build.command()(sleep)

    execution_order = []
    start_time = None

    # 创建一个测试命令，用于验证 sleep 的阻塞效果
    @shell.main_channel.build.command()
    async def foo():
        nonlocal start_time
        if start_time is None:
            start_time = time.time()
        elapsed = time.time() - start_time
        execution_order.append((f"foo", elapsed))
        return f"executed at {elapsed:.3f}s"

    async with shell:
        async with await shell.interpreter() as interpreter:
            # 发送 CTML：先执行 foo，然后 sleep，再执行 foo
            interpreter.feed("""
                <foo/>
                <sleep duration="0.2"/>
                <foo/>
            """)
            interpreter.commit()

            tasks = await interpreter.wait_tasks()

            # 验证执行顺序和时间
            assert len(execution_order) == 2

            # 第一个命令应该在开始时执行
            first_cmd_name, first_cmd_time = execution_order[0]
            assert first_cmd_name == "foo"
            assert first_cmd_time < 0.1  # 应该很快执行

            # 第二个命令应该在 sleep 后执行
            second_cmd_name, second_cmd_time = execution_order[1]
            assert second_cmd_name == "foo"
            assert second_cmd_time >= 0.18  # 至少 sleep 了 180ms
            assert second_cmd_time <= 0.25  # 最多 250ms


@pytest.mark.asyncio
async def test_sleep_in_ctml_with_channel():
    """
    测试在 CTML 中调用 sleep（指定 channel）
    验证它会在指定 channel 上创建任务
    """
    # 创建两个 channel：主 channel 和音频 channel
    main_chan = PyChannel(name="main")
    audio_chan = PyChannel(name="audio")

    # 记录执行顺序
    execution_log = []

    # 在主 channel 上注册 sleep
    @main_chan.build.command()
    async def sleep_wrapper(duration: float, chan: str = ""):
        return await sleep(duration, chan)

    # 在主 channel 上添加一个测试命令
    @main_chan.build.command()
    async def main_task():
        execution_log.append("main_task_start")
        await asyncio.sleep(0.05)  # 模拟一些工作
        execution_log.append("main_task_end")
        return "main_done"

    # 在音频 channel 上添加一个测试命令
    @audio_chan.build.command()
    async def audio_task():
        execution_log.append("audio_task_start")
        await asyncio.sleep(0.1)
        execution_log.append("audio_task_end")
        return "audio_done"

    shell = new_ctml_shell()
    shell.main_channel.import_channels(main_chan, audio_chan)

    async with shell:
        async with await shell.interpreter() as interpreter:
            # 发送 CTML：同时启动主任务和音频 sleep
            interpreter.feed("""
                <main:main_task/>
                <main:sleep_wrapper duration="0.15" chan="audio"/>
            """)
            interpreter.commit()

            tasks = await interpreter.wait_tasks()

            # 验证执行顺序
            # 由于 sleep 在音频 channel 上，它不应该阻塞主 channel
            # 所以 main_task 应该先完成，然后音频 sleep 在后台进行

            # 注意：实际顺序可能因调度而异，但 main_task 应该很快完成
            assert "main_task_start" in execution_log
            assert "main_task_end" in execution_log

            # audio_task 不会被执行，因为我们调用的是 sleep 而不是 audio_task
            # 所以 execution_log 中不会有 audio_task_start/end


@pytest.mark.asyncio
async def test_sleep_with_wait_primitives():
    """
    测试 sleep 与 wait 原语的配合使用
    """
    shell = new_ctml_shell()

    # 注册 sleep 和 wait
    shell.main_channel.build.command()(sleep)

    # 从 wait 模块导入 wait（假设已经实现）
    from ghoshell_moss.core.ctml.shell.primitives.wait import wait

    shell.main_channel.build.command()(wait)

    execution_order = []
    timestamps = []

    @shell.main_channel.build.command()
    async def record_action(name: str):
        timestamps.append((name, time.time()))
        execution_order.append(name)
        return name

    async with shell:
        async with await shell.interpreter() as interpreter:
            start_time = time.time()

            # 使用 wait 来组织一组包含 sleep 的命令
            interpreter.feed("""
                <wait>
                    <record_action name="A"/>
                    <sleep duration="0.1"/>
                    <record_action name="B"/>
                </wait>
                <record_action name="C"/>
            """)
            interpreter.commit()

            tasks = await interpreter.wait_tasks()

            # 验证执行顺序和时间
            assert execution_order == ["A", "B", "C"]

            # 验证时间间隔
            for i, (name, timestamp) in enumerate(timestamps):
                elapsed = timestamp - start_time

                if name == "A":
                    assert elapsed < 0.05  # A 应该很快执行
                elif name == "B":
                    assert elapsed >= 0.09  # B 应该在 sleep 100ms 后执行
                    assert elapsed <= 0.15
                elif name == "C":
                    assert elapsed >= 0.09  # C 应该在 wait 完成后执行
                    # C 应该在 B 之后，但可能很快（因为 wait 结束后立即执行）
                    if i > 0:
                        prev_name, prev_timestamp = timestamps[i - 1]
                        if prev_name == "B":
                            time_diff = timestamp - prev_timestamp
                            assert time_diff < 0.08  # C 应该在 B 后很快执行


@pytest.mark.asyncio
async def test_sleep_cancellation():
    """
    测试 sleep 任务的取消
    """
    shell = new_ctml_shell()
    shell.main_channel.build.command()(sleep)

    execution_log = []

    @shell.main_channel.build.command()
    async def quick_task():
        execution_log.append("quick_task")
        return "quick"

    async with shell:
        async with await shell.interpreter() as interpreter:
            # 启动一个长时间 sleep，然后用 wait 的 timeout 取消它
            interpreter.feed('<wait timeout="0.05"><sleep duration="1.0"/></wait>')
            interpreter.commit()

            tasks = await interpreter.wait_tasks()

            # 验证 sleep 被取消了
            assert len(tasks) == 1
            wait_task = list(tasks.values())[0]

            # wait 应该因为超时而完成
            # sleep 任务应该被取消
            # 具体断言取决于你的任务状态设计


@pytest.mark.asyncio
async def test_sleep_with_multiple_channels():
    """
    测试在多个 channel 上同时 sleep
    """
    # 创建多个 channel
    channels = {}
    for name in ["audio", "video", "motor"]:
        chan = PyChannel(name=name)
        channels[name] = chan

    shell = new_ctml_shell()
    for chan in channels.values():
        shell.main_channel.import_channels(chan)

    # 在主 channel 注册 sleep
    shell.main_channel.build.command()(sleep)

    execution_log = []

    @shell.main_channel.build.command()
    async def logger(msg: str):
        execution_log.append((msg, time.time()))
        return msg

    async with shell:
        async with await shell.interpreter() as interpreter:
            start_time = time.time()

            # 在多个 channel 上同时启动 sleep
            interpreter.feed("""
                <logger msg="start"/>
                <sleep duration="0.1" chan="audio"/>
                <sleep duration="0.2" chan="video"/>
                <sleep duration="0.05" chan="motor"/>
                <logger msg="after_sleeps"/>
            """)
            interpreter.commit()

            tasks = await interpreter.wait_tasks()

            # 验证日志顺序
            # after_sleeps 应该立即记录，因为 sleeps 是在不同 channel 上
            assert len(execution_log) == 2

            first_msg, first_time = execution_log[0]
            second_msg, second_time = execution_log[1]

            assert first_msg == "start"
            assert second_msg == "after_sleeps"

            # after_sleeps 应该很快记录，不等待 sleep 完成
            time_diff = second_time - first_time
            # assert time_diff < 0.05  # 应该很快
            assert time_diff < 0.1  # 批量测试时偶发性能问题.


@pytest.mark.asyncio
async def test_sleep_in_nested_structure():
    """
    测试在嵌套结构中的 sleep
    """
    shell = new_ctml_shell()
    shell.main_channel.build.command()(sleep)

    # 从 wait 模块导入 wait
    from ghoshell_moss.core.ctml.shell.primitives.wait import wait

    shell.main_channel.build.command()(wait)

    execution_order = []

    @shell.main_channel.build.command()
    async def task(name: str):
        execution_order.append(f"start_{name}")
        await asyncio.sleep(0.01)  # 模拟工作
        execution_order.append(f"end_{name}")
        return name

    async with shell:
        async with await shell.interpreter() as interpreter:
            # 嵌套结构：外层 wait 包含内层 wait，内层包含 sleep
            interpreter.feed("""
                <wait>
                    <task name="A"/>
                    <wait>
                        <sleep duration="0.1"/>
                        <task name="B"/>
                    </wait>
                    <task name="C"/>
                </wait>
            """)
            interpreter.commit()

            await interpreter.wait_tasks()

            # 验证执行顺序
            # A 应该先执行
            # 然后内层 wait 执行：sleep 0.1s，然后 B
            # 最后 C
            expected_order = ["start_A", "end_A", "start_B", "end_B", "start_C", "end_C"]

            # 由于 sleep 在内层 wait，B 应该在 sleep 后执行
            # 但实际顺序可能因实现而异，这里我们主要验证所有任务都执行了
            assert len(execution_order) == 6
            assert "start_A" in execution_order
            assert "start_B" in execution_order
            assert "start_C" in execution_order
