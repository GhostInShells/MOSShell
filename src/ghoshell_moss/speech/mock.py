import threading
import time
from queue import Empty, Queue
from typing import Optional

from ghoshell_common.helpers import uuid

from ghoshell_moss.core.concepts.speech import Speech, SpeechStream
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent


class MockSpeechStream(SpeechStream):
    def __init__(
        self,
        speech_outputs: list[str],
        id: str = "",
        typing_sleep: float = 0.0,
        speech_id: str = "",
    ):
        super().__init__(id=id or uuid())
        self.speech_id = speech_id
        self.speech_outputs = speech_outputs
        self.outputs = []
        self.output_queue = Queue()
        self.output_done_event = ThreadSafeEvent()
        self.output_buffer = ""
        self.output_started = False
        self.typing_sleep = typing_sleep

    async def aclose(self):
        self.close()

    def close(self):
        if self.output_done_event.is_set():
            return
        self.output_done_event.set()

    def _buffer(self, text: str) -> None:
        self.output_queue.put_nowait(text)

    def _commit(self) -> None:
        self.output_queue.put_nowait(None)

    async def astart(self) -> None:
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
                if item.strip() and self.typing_sleep > 0.0:
                    time.sleep(self.typing_sleep)
        finally:
            if self.cmd_task is not None:
                self.cmd_task.tokens = self.output_buffer
            self.output_done_event.set()
            self.speech_outputs.append("".join(self.outputs))

    def buffered(self) -> str:
        return self.output_buffer

    async def wait(self) -> None:
        await self.output_done_event.wait()


class MockSpeech(Speech):
    def __init__(self, typing_sleep: float = 0.0):
        self._streams: dict[str, MockSpeechStream] = {}
        self._outputs = []
        self._closed = ThreadSafeEvent()
        self._typing_sleep = typing_sleep
        self._uid = uuid()

    def new_stream(self, *, batch_id: Optional[str] = None) -> SpeechStream:
        stream = MockSpeechStream(
            self._outputs,
            id=batch_id,
            typing_sleep=self._typing_sleep,
            speech_id=self._uid,
        )
        stream_id = stream.id
        if stream_id in self._streams:
            existing_stream = self._streams[stream_id]
            existing_stream.close()
        self._streams[stream_id] = stream
        return stream

    def is_running(self) -> bool:
        return True

    def outputted(self) -> list[str]:
        result = self._outputs.copy()
        return result

    async def clear(self) -> list[str]:
        outputs = []
        for stream in self._streams.values():
            await stream.aclose()
        for stream_output in self._outputs:
            outputs.append(stream_output)
        self._streams.clear()
        self._outputs.clear()
        return outputs

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        self._closed.set()

    async def wait_closed(self) -> None:
        await self._closed.wait()
