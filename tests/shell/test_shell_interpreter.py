import pytest
from ghoshell_moss.core import PyChannel, new_ctml_shell, InterpretError
import time
import queue
import asyncio


@pytest.mark.asyncio
async def test_text_token_parser_with_invalid_input():
    """
    测试 wait_idle 与其他原语的配合
    """
    shell = new_ctml_shell()
    receiver = []
    async with shell:
        interpreter = await shell.interpreter()

        # test 1: invalid format
        input_queue = queue.Queue()
        t = asyncio.create_task(
            asyncio.to_thread(interpreter.parse_text_to_command_tokens, input_queue, receiver.append),
        )
        for c in "invalid <a>ctml</> text":
            input_queue.put(c)
        input_queue.put(None)
        with pytest.raises(InterpretError):
            await t
        receiver.clear()

        # test 2: invalid format
        input_queue = queue.Queue()
        t = asyncio.create_task(
            asyncio.to_thread(interpreter.parse_text_to_command_tokens, input_queue, receiver.append),
        )
        for c in "<foo> not done text":
            input_queue.put(c)
        input_queue.put(None)
        with pytest.raises(InterpretError):
            await t
        receiver.clear()

        # test 3: empty input
        input_queue = queue.Queue()
        t = asyncio.create_task(
            asyncio.to_thread(interpreter.parse_text_to_command_tokens, input_queue, receiver.append),
        )
        input_queue.put(None)
        await t
        assert len(receiver) == 3
        assert receiver[0].seq == "start"
        assert receiver[1].seq == "end"
        assert receiver[2] is None
        receiver.clear()

        # test 4: stopped while sending. no exception raised
        stopped = asyncio.Event()
        input_queue = queue.Queue()
        t = asyncio.create_task(
            asyncio.to_thread(
                interpreter.parse_text_to_command_tokens,
                input_queue,
                receiver.append,
                stopped=stopped.is_set,
            ),
        )
        for c in "<foo> not done text":
            input_queue.put(c)
        stopped.set()
        await t
        receiver.clear()


@pytest.mark.asyncio
async def test_shell_interpreter_async_parse_text():
    """
    测试 wait_idle 与其他原语的配合
    """
    shell = new_ctml_shell()
    async with shell:
        interpreter = await shell.interpreter()

        content = "invalid <a>ctml</> text"

        async def gen():
            nonlocal content
            for c in content:
                yield c

        with pytest.raises(InterpretError):
            tokens = []
            async for token in interpreter.aparse_text_to_command_tokens(gen()):
                tokens.append(token)

        content = "<foo/><bar/>"
        tokens = []
        async for token in interpreter.aparse_text_to_command_tokens(gen()):
            tokens.append(token)
        assert len(tokens) == 2 + 2 * 2


@pytest.mark.asyncio
async def test_run_not_exists_command():
    """
    测试 wait_idle 与其他原语的配合
    """
    shell = new_ctml_shell()
    async with shell:
        async with await shell.interpreter() as interpreter:
            # 复杂场景：启动后台任务，sleep，然后 wait_idle
            interpreter.feed("""
                <bg:background_work/>
                <sleep duration:float="0.1"/>
            """)
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            for task in tasks:
                print(task)
            with pytest.raises(InterpretError):
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
        async with await shell.interpreter() as interpreter:
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

    bg = PyChannel(name="bg")

    @bg.build.command()
    async def foo():
        return

    shell.main_channel.import_channels(bg)

    async with shell:
        async with shell.interpreter_in_ctx(clear_after_exit=True) as interpreter:
            interpretation = interpreter.interpretation()
            # 复杂场景：启动后台任务，sleep，然后 wait_idle
            interpreter.feed("""
                <bg:foo/>
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
        async with await shell.interpreter() as interpreter:
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
    assert even_gap < 0.07


@pytest.mark.asyncio
async def test_run_shell_raise_exception():
    shell = new_ctml_shell()
    with pytest.raises(RuntimeError):
        async with shell:
            async with await shell.interpreter() as interpreter:
                raise RuntimeError("failed")
