import asyncio

from ghoshell_moss.mocks.outputs import ArrOutput, ArrOutputStream
from ghoshell_moss.concepts.shell import OutputStream
import pytest


@pytest.mark.asyncio
async def test_output_in_asyncio():
    content = "hello world"

    async def buffer_stream(_stream: OutputStream, idx_: int):
        for c in content:
            _stream.buffer(c)
        # add a tail at the output end
        _stream.buffer(str(idx_))
        _stream.commit()

    output = ArrOutput()
    for i in range(5):
        idx = i
        stream = output.new_stream(batch_id=str(idx))
        stream = stream
        sending_task = asyncio.create_task(buffer_stream(stream, idx))

        # assert the tasks run in order
        cmd_task = stream.as_command_task()
        await asyncio.gather(sending_task, asyncio.create_task(cmd_task.run()))

    outputted = output.clear()
    assert len(outputted) == 5
    idx = 0
    for item in outputted:
        assert item == f"{content}{idx}"
        idx += 1

    # test clear success
    outputted2 = output.clear()
    assert len(outputted2) == 0
