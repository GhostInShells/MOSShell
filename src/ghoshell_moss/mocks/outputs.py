import asyncio
from typing import Optional, List, Dict

from ghoshell_moss.concepts.command import CommandTask, BaseCommandTask, CommandMeta, PyCommand
from ghoshell_moss.concepts.shell import Output, OutputStream
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_common.helpers import uuid

import threading
from queue import Queue, Empty


class ArrOutputStream(OutputStream):

    def __init__(self, outputs: List[str], id: str = "", ):
        self.outputs = outputs
        self.id = id or uuid()
        self.output_queue = Queue()
        self.output_done_event = ThreadSafeEvent()
        self.output_buffer = ""
        self.output_started = False

    def close(self):
        if self.output_done_event.is_set():
            return
        self.output_done_event.set()

    def _buffer(self, text: str) -> None:
        self.output_queue.put_nowait(text)

    def _commit(self) -> None:
        self.output_queue.put_nowait(None)

    def start(self) -> None:
        if self.output_started:
            return
        self.output_started = True
        t = threading.Thread(target=self._output_loop, daemon=True)
        t.start()

    def _output_loop(self) -> None:
        try:
            content_is_not_empty = False
            while not self.output_done_event.is_set():
                try:
                    self.output_queue.empty()
                    item = self.output_queue.get(block=True, timeout=0.1)
                except Empty:
                    continue

                if item is None:
                    break
                elif not item:
                    continue
                self.output_buffer += item
                if content_is_not_empty:
                    self.outputs.append(item)
                elif self.output_buffer.strip():
                    self.outputs.append(self.output_buffer)
                    content_is_not_empty = True
        finally:
            if self.cmd_task is not None:
                self.cmd_task.tokens = self.output_buffer
            self.output_done_event.set()

    def buffered(self) -> str:
        return self.output_buffer

    async def wait(self) -> None:
        await self.output_done_event.wait()


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
            existing_stream.close()
        self._streams[stream_id] = stream
        self._outputs[stream_id] = stream_outputs
        return stream

    def outputted(self) -> List[str]:
        data = self._outputs.copy()
        result = []
        for contents in data.values():
            result.append("".join(contents))
        return result

    def clear(self) -> List[str]:
        outputs = []
        for stream in self._streams.values():
            stream.close()
        for stream_output in self._outputs.values():
            outputs.append("".join(stream_output))
        self._streams.clear()
        self._outputs.clear()
        return outputs
