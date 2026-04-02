from ghoshell_moss.speech.mock import MockSpeech
from ghoshell_moss.core import new_ctml_shell, new_channel, CommandErrorCode
import pytest
import asyncio


@pytest.mark.asyncio
async def test_shell_with_output_channel_in_wait():
    speech = MockSpeech()
    shell = new_ctml_shell(speech=speech)

    async with shell:
        async with await shell.interpreter() as interpreter:
            # use wait to call imagining commands.
            interpreter.feed("<wait><a:foo/>hello</wait><wait><a:foo/>world</wait>")
            interpreter.commit()
            await interpreter.wait_stopped()
            interpreter.raise_exception()
            interpretation = interpreter.interpretation()
            assert interpretation.interrupted is False

            assert len(interpretation.execution_messages()) == 1
            for msg in interpretation.execution_messages():
                # 暴露了异常. 深层异常是 a:foo 不存在.
                assert CommandErrorCode.INTERPRET_ERROR.name in str(msg)


@pytest.mark.asyncio
async def test_shell_speech_baseline_prepare():
    speech = MockSpeech(typing_sleep=0.0)
    shell = new_ctml_shell(speech=speech)
    a_chan = new_channel(name="a")

    @a_chan.build.command()
    async def foo():
        return 123

    shell.main_channel.import_channels(a_chan)

    async def say(chunks__):
        stream = speech.new_stream()
        await stream.speak(chunks__)

    shell.main_channel.build.command()(say)

    async with shell:
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<a:foo/>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            assert len(tasks) == 1
            task = list(tasks.values())[0]
            assert task.success()
            task_result = task.task_result()
            assert task_result.result is 123
            assert len(task_result.as_messages()) == 1

        async with await shell.interpreter() as interpreter:
            interpreter.feed("<wait><a:foo/>hello</wait><wait><a:foo/>world</wait>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            assert len(tasks) == 2

            interpreter.raise_exception()
            assert speech.outputted() == ["hello", "world"]
            interpretation = interpreter.interpretation()
            assert interpretation.interrupted is False
            assert len(interpretation.exception) == 0
            assert len(interpretation.execution_messages()) == 2

        async with await shell.interpreter() as interpreter:
            content = "你好，我是MOSS。"
            for c in content:
                await asyncio.sleep(0.01)
                interpreter.feed(c)
            interpreter.commit()
            await interpreter.wait_stopped()
            assert speech.outputted() == ["你好，我是MOSS。"]

        content = "<wait><say>你好，我是MOSS。</say></wait>"
        tokens = []
        async for token in shell.parse_text_to_command_tokens(content):
            tokens.append(token)
        assert len(tokens) == 7
        tasks = []
        async for task in shell.parse_text_to_tasks(content):
            tasks.append(task)
        assert len(tasks) == 1

        async with await shell.interpreter() as interpreter:
            for c in content:
                await asyncio.sleep(0.01)
                interpreter.feed(c)
            interpreter.commit()
            await asyncio.sleep(0.05)
            interpreter.raise_exception()
            await interpreter.wait_tasks()
            interpreter.raise_exception()
            outputted = speech.outputted()
            assert speech.outputted()[0] == "你好，我是MOSS。"


@pytest.mark.asyncio
async def test_shell_speech_baseline():
    speech = MockSpeech(typing_sleep=0.0)
    shell = new_ctml_shell(speech=speech)
    a_chan = new_channel(name="a")

    @a_chan.build.command()
    async def foo():
        return 123

    shell.main_channel.import_channels(a_chan)

    async def say(chunks__):
        stream = speech.new_stream()
        await stream.speak(chunks__)

    shell.main_channel.build.command()(say)
    content = "<wait><say>你好，我是MOSS。</say></wait>"

    async with shell:
        async with await shell.interpreter() as interpreter:
            for c in content:
                await asyncio.sleep(0.01)
                interpreter.feed(c)
            interpreter.commit()
            await asyncio.sleep(0.05)
            interpreter.raise_exception()
            await interpreter.wait_tasks()
            interpreter.raise_exception()
            outputted = speech.outputted()
            assert speech.outputted()[0] == "你好，我是MOSS。"


@pytest.mark.asyncio
async def test_shell_speech_10_times():
    speech = MockSpeech(typing_sleep=0.0)
    shell = new_ctml_shell(speech=speech)
    a_chan = new_channel(name="a")

    @a_chan.build.command()
    async def foo():
        return 123

    shell.main_channel.import_channels(a_chan)

    async def say(chunks__):
        stream = speech.new_stream()
        await stream.speak(chunks__)

    shell.main_channel.build.command()(say)
    content = "hello<wait><say>你好，我是MOSS。</say></wait> world"

    async with shell:
        for i in range(10):
            async with await shell.interpreter() as interpreter:
                for c in content:
                    await asyncio.sleep(0.001)
                    interpreter.feed(c)
                interpreter.commit()
                interpreter.raise_exception()
                await interpreter.wait_tasks()
                interpreter.raise_exception()
                outputted = speech.outputted()
                print(outputted)
                assert outputted[1] == "你好，我是MOSS。"
