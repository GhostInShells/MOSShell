import asyncio

import pytest

from ghoshell_moss.core.contracts.speech import SpeechStream
from ghoshell_moss.speech.mock import MockSpeech


@pytest.mark.asyncio
async def test_output_in_asyncio():
    content = "hello world"

    async def buffer_stream(_stream: SpeechStream, idx_: int):
        for c in content:
            _stream.feed(c)
            await asyncio.sleep(0)
        # add a tail at the mock_speech end
        _stream.feed(str(idx_))
        _stream.commit()

    mock_speech = MockSpeech(typing_sleep=0.0)
    for i in range(5):
        idx = i
        stream = mock_speech.new_stream(batch_id=str(idx))
        stream = stream
        sending_task = asyncio.create_task(buffer_stream(stream, idx))

        # assert the tasks run in order
        cmd_task = stream.as_command_task()
        await asyncio.gather(sending_task, asyncio.create_task(cmd_task.run()))

    outputted = await mock_speech.clear()
    assert len(outputted) == 5
    idx = 0
    for item in outputted:
        assert item == f"{content}{idx}"
        idx += 1

    # test clear success
    outputted2 = await mock_speech.clear()
    assert len(outputted2) == 0


@pytest.mark.asyncio
async def test_output_in_concurrent():
    content = "hello world"

    async def buffer_stream(_stream: SpeechStream, idx_: int):
        for c in content:
            _stream.feed(c)
            await asyncio.sleep(0)
        # add a tail at the mock_speech end
        _stream.feed(str(idx_))
        _stream.commit()

    mock_speech = MockSpeech(typing_sleep=0.0)
    gathering = []
    for i in range(2):
        idx = i
        stream = mock_speech.new_stream(batch_id=str(idx))
        stream = stream
        cmd_task = stream.as_command_task()
        gathering.append(buffer_stream(stream, idx))
        gathering.append(cmd_task.run())

        # assert the tasks run in order
    await asyncio.gather(*gathering)
    outputted = await mock_speech.clear()
    assert len(outputted) == 2
