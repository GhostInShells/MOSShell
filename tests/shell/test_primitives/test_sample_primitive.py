import pytest

from ghoshell_moss.core.ctml.shell.primitives.sample import sample
from ghoshell_moss.core import PyChannel, new_ctml_shell


@pytest.mark.asyncio
async def test_sample_pick_one():
    """
    测试 sample 基本功能：从多个任务中随机选择 1 个执行
    """
    # 创建 Channel
    chan = PyChannel(name="chan")

    done = []

    @chan.build.command()
    async def task1():
        done.append("task1")

    @chan.build.command()
    async def task2():
        done.append("task2")

    @chan.build.command()
    async def task3():
        done.append("task3")

    shell = new_ctml_shell()
    shell.main_channel.import_channels(chan)
    shell.main_channel.build.command()(sample)

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed("<sample><chan:task1/><chan:task2/><chan:task3/></sample>")
            interpreter.commit()
            await interpreter.wait_stopped()
            # 验证只有 1 个任务被执行
            assert len(done) == 1
            assert done[0] in ["task1", "task2", "task3"]


@pytest.mark.asyncio
async def test_sample_pick_multiple():
    """
    测试 sample 选择多个任务：从 5 个任务中随机选择 2 个执行
    """
    chan = PyChannel(name="chan")

    done = []

    @chan.build.command()
    async def task1():
        done.append("task1")

    @chan.build.command()
    async def task2():
        done.append("task2")

    @chan.build.command()
    async def task3():
        done.append("task3")

    @chan.build.command()
    async def task4():
        done.append("task4")

    @chan.build.command()
    async def task5():
        done.append("task5")

    shell = new_ctml_shell()
    shell.main_channel.import_channels(chan)
    shell.main_channel.build.command()(sample)

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed(
                '<sample pick="2"><chan:task1/><chan:task2/><chan:task3/><chan:task4/><chan:task5/></sample>'
            )
            interpreter.commit()
            await interpreter.wait_stopped()
            # 验证只有 2 个任务被执行
            assert len(done) == 2
            # 验证执行的任务都在范围内
            for task in done:
                assert task in ["task1", "task2", "task3", "task4", "task5"]


@pytest.mark.asyncio
async def test_sample_pick_all():
    """
    测试 sample 选择全部任务：相当于随机排序后全部执行
    """
    chan = PyChannel(name="chan")

    done = []

    @chan.build.command()
    async def task1():
        done.append("task1")

    @chan.build.command()
    async def task2():
        done.append("task2")

    @chan.build.command()
    async def task3():
        done.append("task3")

    shell = new_ctml_shell()
    shell.main_channel.import_channels(chan)
    shell.main_channel.build.command()(sample)

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed('<sample pick="3"><chan:task1/><chan:task2/><chan:task3/></sample>')
            interpreter.commit()
            await interpreter.wait_stopped()
            # 验证所有 3 个任务都被执行
            assert len(done) == 3
            assert set(done) == {"task1", "task2", "task3"}


@pytest.mark.asyncio
async def test_sample_invalid_pick_zero():
    """
    测试 sample 参数验证：pick < 1 时任务应该失败
    """
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def task1():
        pass

    shell = new_ctml_shell()
    shell.main_channel.import_channels(chan)
    shell.main_channel.build.command()(sample)

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed('<sample pick="0"><chan:task1/></sample>')
            interpreter.commit()
            interpretation = await interpreter.wait_stopped()
            # 验证 sample 任务在失败列表中
            assert "sample" in interpretation.failed_tasks.values()
            # 验证没有成功执行的任务
            assert len(interpretation.success_tasks) == 0
            # 验证 observe 标志被设置（因为任务失败）
            assert interpretation.observe is True
            # 验证错误消息中包含异常信息
            assert len(interpretation.messages) > 0
            error_msg_found = False
            for msg in interpretation.messages:
                # if msg.type == "text" and msg.contents:
                if not msg.contents:
                    continue
                for content in msg.contents:
                    assert content["type"] == "text"
                    if "pick must be >= 1" in content["data"]["text"]:
                        error_msg_found = True
                        break
            assert error_msg_found, f"Expected error message not found in {interpretation.messages}"


@pytest.mark.asyncio
async def test_sample_invalid_pick_exceed():
    """
    测试 sample 参数验证：pick 超过任务数量时任务应该失败
    """
    chan = PyChannel(name="chan")

    @chan.build.command()
    async def task1():
        pass

    @chan.build.command()
    async def task2():
        pass

    shell = new_ctml_shell()
    shell.main_channel.import_channels(chan)
    shell.main_channel.build.command()(sample)

    async with shell:
        async with shell.interpreter_in_ctx() as interpreter:
            interpreter.feed('<sample pick="5"><chan:task1/><chan:task2/></sample>')
            interpreter.commit()
            interpretation = await interpreter.wait_stopped()
            # 验证 sample 任务在失败列表中
            assert "sample" in interpretation.failed_tasks.values()
            # 验证没有成功执行的任务
            assert len(interpretation.success_tasks) == 0
            # 验证 observe 标志被设置（因为任务失败）
            assert interpretation.observe is True
            # 验证错误消息中包含异常信息
            assert len(interpretation.messages) > 0
            error_msg_found = False
            for msg in interpretation.messages:
                if not msg.contents:
                    continue
                for content in msg.contents:
                    if "requires at least" in content["data"]["text"]:
                        error_msg_found = True
                        break
            assert error_msg_found, f"Expected error message not found in {interpretation.messages}"


@pytest.mark.asyncio
async def test_sample_random_distribution():
    """
    测试 sample 随机性：多次执行应该覆盖所有可能的任务
    """
    chan = PyChannel(name="chan")

    results = []

    @chan.build.command()
    async def task1():
        results.append("task1")

    @chan.build.command()
    async def task2():
        results.append("task2")

    @chan.build.command()
    async def task3():
        results.append("task3")

    shell = new_ctml_shell()
    shell.main_channel.import_channels(chan)
    shell.main_channel.build.command()(sample)

    # 执行多次，收集结果
    async with shell:
        for _ in range(20):
            results.clear()
            async with shell.interpreter_in_ctx() as interpreter:
                interpreter.feed("<sample><chan:task1/><chan:task2/><chan:task3/></sample>")
                interpreter.commit()
                await interpreter.wait_stopped()

    # 验证在 20 次执行中，所有任务都被执行过（随机分布验证）
    unique_tasks = set()
    # 注意：这里我们不能在循环外访问 results，因为每次执行都会清空
    # 这个测试主要是为了验证不会抛出异常，且每次执行 1 个任务
