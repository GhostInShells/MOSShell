from ghoshell_moss.core.ctml.shell.primitives import wait
from ghoshell_moss.core.ctml.shell import new_ctml_shell
from ghoshell_moss.core import PyChannel
from ghoshell_moss.speech import MockSpeech
import pytest
import asyncio


@pytest.mark.asyncio
async def test_wait_invalid_command():
    shell = new_ctml_shell()
    async with shell:
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<wait><a:foo/><b:bar/><a:foo/></wait>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            interpreter.raise_exception()
            assert len(tasks) == 1
            tasks = list(tasks.values())
            assert tasks[0].exception() is not None


@pytest.mark.asyncio
async def test_wait_primitive():
    a_chan = PyChannel(name="a")
    b_chan = PyChannel(name="b")

    ordered = []

    @a_chan.build.command()
    @b_chan.build.command()
    async def foo():
        ordered.append("foo")
        return 123

    @b_chan.build.command()
    async def bar():
        await asyncio.sleep(0.3)
        ordered.append("bar")
        return 456

    shell = new_ctml_shell()
    shell.main_channel.import_channels(a_chan, b_chan)
    shell.main_channel.build.command()(wait)
    async with shell:
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<a:foo/><b:bar/><a:foo/>")
            interpreter.commit()
            await interpreter.wait_tasks()
            # bar is later because sleep
            assert ordered == ["foo", "foo", "bar"]

        # 验证添加了 wait 后改变了排序.
        ordered.clear()
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<wait><a:foo/><b:bar/></wait><a:foo/>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            # bar is executed before second foo
            for t in tasks.values():
                assert t.success()
            assert ordered == ["foo", "bar", "foo"]

        # 验证多组 wait
        ordered.clear()
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<wait><a:foo/><b:bar/></wait><wait><a:foo/><b:bar/></wait>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            # bar is executed before second foo
            for t in tasks.values():
                assert t.success()
            assert ordered == ["foo", "bar", "foo", "bar"]

        # 验证 timeout
        ordered.clear()
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<wait timeout:float='0.2'><a:foo/><b:bar/><a:foo/><b:bar/></wait>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            # 只有 foo 成功了. 其它的都被 timeout 了.
            assert ordered == ["foo", "foo"]


@pytest.mark.asyncio
async def test_shell_wait_talk():
    speech = MockSpeech()
    shell = new_ctml_shell(speech=speech)
    async with shell:
        async with await shell.interpreter() as interpreter:
            for c in "hello world":
                interpreter.feed(c)
            interpreter.commit()
            await interpreter.wait_stopped()
            assert speech.outputted() == ["hello world"]

        async with await shell.interpreter() as interpreter:
            for c in "<wait>hello world</wait>":
                interpreter.feed(c)
            interpreter.commit()
            await asyncio.sleep(0.3)
            assert speech.outputted() == ["hello world"]
            await interpreter.wait_stopped()
            assert speech.outputted() == ["hello world"]


@pytest.mark.asyncio
async def test_wait_return_when_first_complete():
    """测试return_when='FIRST_COMPLETE'策略"""
    a_chan = PyChannel(name="a")
    b_chan = PyChannel(name="b")

    execution_log = []
    completion_order = []

    @a_chan.build.command()
    async def slow_task():
        execution_log.append("slow_start")
        await asyncio.sleep(0.5)
        execution_log.append("slow_end")
        completion_order.append("slow")
        return "slow_result"

    @b_chan.build.command()
    async def fast_task():
        execution_log.append("fast_start")
        await asyncio.sleep(0.1)
        execution_log.append("fast_end")
        completion_order.append("fast")
        return "fast_result"

    shell = new_ctml_shell()
    shell.main_channel.import_channels(a_chan, b_chan)
    shell.main_channel.build.command()(wait)

    async with shell:
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<wait return_when='FIRST_COMPLETE'><a:slow_task/><b:fast_task/></wait>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            assert len(tasks) == 1

            # 验证fast_task先完成，slow_task被取消
            assert execution_log == ["slow_start", "fast_start", "fast_end"]
            # slow_end不应该出现，因为被取消了
            assert "slow_end" not in execution_log
            assert completion_order == ["fast"]


@pytest.mark.asyncio
async def test_wait_return_when_all_complete():
    """测试return_when='ALL_COMPLETE'策略"""
    a_chan = PyChannel(name="a")
    b_chan = PyChannel(name="b")

    execution_log = []

    @a_chan.build.command()
    async def task_a():
        execution_log.append("a_start")
        await asyncio.sleep(0.1)
        execution_log.append("a_end")
        return "a"

    @b_chan.build.command()
    async def task_b():
        execution_log.append("b_start")
        await asyncio.sleep(0.2)
        execution_log.append("b_end")
        return "b"

    shell = new_ctml_shell()
    shell.main_channel.import_channels(a_chan, b_chan)
    shell.main_channel.build.command()(wait)

    async with shell:
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<wait return_when='ALL_COMPLETE'><a:task_a/><b:task_b/></wait>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()

            # 验证两个任务都完成了
            assert execution_log == ["a_start", "b_start", "a_end", "b_end"]

            # 验证两个任务都成功
            assert len(tasks) == 1
            result = list(tasks.values())[0].task_result()
            assert len(result.messages) == 2


@pytest.mark.asyncio
async def test_wait_with_exception():
    """测试异常处理：return_when='FIRST_EXCEPTION'"""
    a_chan = PyChannel(name="a")
    b_chan = PyChannel(name="b")

    execution_log = []

    @a_chan.build.command()
    async def failing_task():
        execution_log.append("failing_start")
        await asyncio.sleep(0.1)
        execution_log.append("failing_end")
        raise ValueError("Intentional error")

    @b_chan.build.command()
    async def normal_task():
        execution_log.append("normal_start")
        await asyncio.sleep(0.2)
        execution_log.append("normal_end")
        return "normal"

    shell = new_ctml_shell()
    shell.main_channel.import_channels(a_chan, b_chan)
    shell.main_channel.build.command()(wait)

    async with shell:
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<wait><a:failing_task/><b:normal_task/></wait>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()

            # 验证异常传播
            assert execution_log == ["failing_start", "normal_start", "failing_end"]


@pytest.mark.asyncio
async def test_wait_empty_commands():
    """测试空命令组的wait行为"""
    shell = new_ctml_shell()
    shell.main_channel.build.command()(wait)

    async with shell:
        async with await shell.interpreter() as interpreter:
            # 测试空wait
            interpreter.feed("<wait></wait>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()

            assert len(tasks) == 1

            # 测试只有空白字符的wait
            interpreter.feed("<wait>   \n\t  </wait>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            assert len(tasks) == 1


@pytest.mark.asyncio
async def test_wait_nested_structure():
    """测试嵌套的wait结构"""
    a_chan = PyChannel(name="a")

    execution_order = []

    @a_chan.build.command()
    async def task(num: int):
        await asyncio.sleep(0.001 * num)
        execution_order.append(f"task_{num}")
        return num

    shell = new_ctml_shell()
    shell.main_channel.import_channels(a_chan)
    shell.main_channel.build.command()(wait)

    async with shell:
        # 测试嵌套wait：外层wait包含内层wait
        async with await shell.interpreter() as interpreter:
            interpreter.feed("""
                <wait>
                    <a:task num="1"/>
                    <wait>
                        <a:task num="2"/>
                        <a:task num="3"/>
                    </wait>
                    <a:task num="4"/>
                </wait>
            """)
            interpreter.commit()
            await interpreter.wait_tasks()

            # 验证执行顺序：内层wait完成后才执行task_4
            # 注意：由于都是同一个channel，可能按顺序执行，但wait确保同步点
            assert len(execution_order) == 4
            assert "task_4" in execution_order
            # task_4应该在task_2和task_3之后（因为在内层wait中）


@pytest.mark.asyncio
async def test_wait_with_mixed_blocking_modes():
    """测试混合阻塞和非阻塞命令的wait"""
    a_chan = PyChannel(name="a", blocking=True)  # 阻塞channel
    b_chan = PyChannel(name="b", blocking=False)  # 非阻塞channel

    execution_log = []

    @a_chan.build.command()
    async def blocking_task(name: str):
        execution_log.append(f"blocking_start_{name}")
        await asyncio.sleep(0.15)
        execution_log.append(f"blocking_end_{name}")
        return f"blocking_{name}"

    @b_chan.build.command()
    async def non_blocking_task(name: str):
        execution_log.append(f"non_blocking_start_{name}")
        await asyncio.sleep(0.1)
        execution_log.append(f"non_blocking_end_{name}")
        return f"non_blocking_{name}"

    shell = new_ctml_shell()
    shell.main_channel.import_channels(a_chan, b_chan)
    shell.main_channel.build.command()(wait)

    async with shell:
        async with await shell.interpreter() as interpreter:
            # 混合阻塞和非阻塞命令
            interpreter.feed("""
                <wait>
                    <a:blocking_task name="A"/>
                    <b:non_blocking_task name="B"/>
                    <a:blocking_task name="C"/>
                </wait>
            """)
            interpreter.commit()
            tasks = await interpreter.wait_tasks()

            # 验证执行日志
            # 注意：非阻塞任务可能和阻塞任务并行执行
            assert "blocking_start_A" in execution_log
            assert "non_blocking_start_B" in execution_log
            assert "blocking_start_C" in execution_log


@pytest.mark.asyncio
async def test_wait_cancellation_propagation():
    """测试wait取消时的传播行为"""
    a_chan = PyChannel(name="a")

    task_started = False
    task_cleaned_up = False

    @a_chan.build.command()
    async def cancellable_task():
        nonlocal task_started, task_cleaned_up
        task_started = True
        try:
            await asyncio.sleep(10)  # 长时间任务
        except asyncio.CancelledError:
            # 清理逻辑
            task_cleaned_up = True
            raise

    shell = new_ctml_shell()
    shell.main_channel.import_channels(a_chan)
    shell.main_channel.build.command()(wait)

    async with shell:
        async with await shell.interpreter() as interpreter:
            # 启动一个会被超时取消的任务
            interpreter.feed('<wait timeout="0.05"><a:cancellable_task/></wait>')
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            # 验证任务被正确取消
            await asyncio.sleep(0.01)
            assert task_started
            assert task_cleaned_up  # 确保清理逻辑被执行


@pytest.mark.asyncio
async def test_wait_in_channels():
    shell = new_ctml_shell()

    cancelled = []

    async def foo():
        nonlocal cancelled
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append("foo")

    async def say():
        await asyncio.sleep(0.1)

    for i in range(10):
        channel = PyChannel(name=f"chan{i}")
        channel.build.command()(foo)
        shell.main_channel.import_channels(channel)

    speech = PyChannel(name="speech")
    speech.build.command()(say)
    shell.main_channel.import_channels(speech)
    async with shell:
        async with await shell.interpreter() as interpreter:
            interpreter.feed('<wait chans="speech">')
            for i in range(10):
                interpreter.feed(f"<chan{i}:foo/>")
            interpreter.feed(f"<speech:say/></wait>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks(3, clear_undone=False)
            assert len(cancelled) == 10
