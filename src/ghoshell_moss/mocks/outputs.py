from typing import Optional, List, Dict

from ghoshell_moss.concepts.command import CommandTask, BaseCommandTask, CommandMeta, PyCommand
from ghoshell_moss.concepts.shell import Output, OutputStream
from ghoshell_common.helpers import uuid
import asyncio


class ArrOutputStream(OutputStream):

    def __init__(self, outputs: List[str], id: str = "", ):
        self.outputs = outputs
        self.id = id or uuid()
        self.output_queue = asyncio.Queue()
        self.play_task: Optional[asyncio.Task] = None
        self.output_is_done = asyncio.Event()
        self.output_buffer = ""

    async def stop(self):
        if self.output_is_done.is_set():
            return
        self.output_is_done.set()
        if self.play_task is not None:
            self.play_task.cancel()

    def buffer(self, text: str, *, complete: bool = False) -> None:
        self.output_queue.put_nowait(text)
        if complete:
            self.output_queue.put_nowait(None)

    async def output_start(self) -> None:
        if self.play_task is None and not self.output_is_done.is_set():
            self.play_task = asyncio.create_task(self.output_start())

    def is_done(self) -> bool:
        return self.output_is_done.is_set()

    async def _play(self) -> None:
        try:
            has_content = False
            while not self.output_is_done.is_set():
                item = await self.output_queue.get()
                if item is None:
                    break
                self.output_buffer += item
                if has_content:
                    self.outputs.append(item)
                elif self.output_buffer.strip():
                    self.outputs.append(self.output_buffer)
                    has_content = True
        finally:
            self.output_is_done.set()

    async def wait_done(self, timeout: float | None = None) -> None:
        await asyncio.wait_for(self.output_is_done.wait(), timeout=timeout)

    async def _output_play_and_wait(self) -> None:
        await self._play()
        await self.wait_done()

    def as_command_task(self) -> Optional[CommandTask]:
        command = PyCommand(self._output_play_and_wait)
        return BaseCommandTask.from_command(command)


class ArrOutput(Output):

    def __init__(self):
        self._streams: dict[str, ArrOutputStream] = {}
        self._outputs: Dict[str, List[str]] = {}

    def new_stream(self, *, batch_id: Optional[str] = None) -> OutputStream:
        stream_outputs = []
        stream = ArrOutputStream(stream_outputs, id=batch_id)
        stream_id = stream.id
        if stream_id in self._streams:
            existing_stream = self._streams[stream_id]
            existing_stream.stop()
        self._streams[stream_id] = stream
        self._outputs[stream_id] = stream_outputs
        return stream

    def clear(self) -> List[str]:
        outputs = []
        for stream in self._streams.values():
            stream.stop()
        for stream_output in self._outputs.values():
            outputs.append("".join(stream_output))
        self._streams.clear()
        self._outputs.clear()
        return outputs
