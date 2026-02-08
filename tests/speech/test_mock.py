import asyncio

import pytest

from ghoshell_moss.core.concepts.speech import SpeechStream
from ghoshell_moss.speech.mock import MockSpeech


@pytest.mark.asyncio
async def test_output_in_asyncio():
    content = "hello world"

    async def buffer_stream(_stream: SpeechStream, idx_: int):
        for c in content:
            _stream.buffer(c)
            await asyncio.sleep(0)
        # add a tail at the mock_speech end
        _stream.buffer(str(idx_))
        _stream.commit()

    mock_speech = MockSpeech()
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
