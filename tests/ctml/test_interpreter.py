from ghoshell_moss.speech.mock import MockSpeech
from ghoshell_moss.ctml.interpreter import CTMLInterpreter
from ghoshell_moss.concepts.command import PyCommand, make_command_group
from collections import deque
import asyncio
import pytest


@pytest.mark.asyncio
async def test_interpreter_baseline():
    async def foo() -> int:
        return 123

    queue = deque()
    interpreter = CTMLInterpreter(
        commands=make_command_group(PyCommand(foo)),
        stream_id="test",
        speech=MockSpeech(),
        callback=queue.append,
    )

    content = "<foo>h</foo>"

    async with interpreter:
        # system prompt is not none
        assert len(interpreter.meta_system_prompt()) > 0
        for c in content:
            interpreter.feed(c)
        await interpreter.wait_parse_done()

    # 所有的 input 被 buffer 了.
    assert content == interpreter.inputted()
    assert len(list(interpreter.parsed_tokens())) == 5
    for token in interpreter.parsed_tokens():
        if token.name == "foo":
            assert token.chan == ""

    assert len(queue) == 4
    assert len(interpreter.parsed_tasks()) == 3


@pytest.mark.asyncio
async def test_interpreter_cancel():
    async def foo() -> int:
        return 123

    queue = deque()
    interpreter = CTMLInterpreter(
        commands=make_command_group(PyCommand(foo)),
        stream_id="test",
        speech=MockSpeech(),
        callback=queue.append,
    )

    content = ["<foo>", "hello", "</foo>"]

    async def consumer():
        async with interpreter:
            for c in content:
                interpreter.feed(c)
                await asyncio.sleep(0.1)

            await interpreter.wait_execution_done()

    async def cancel():
        await asyncio.sleep(0.2)
        await interpreter.stop()

    await asyncio.gather(cancel(), consumer())
    inputted = interpreter.inputted()
    # 有一部分输入, 但是输入不完整.
    assert len(inputted) > 0 and content != inputted
