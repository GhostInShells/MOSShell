import asyncio
from typing import List, Dict, Optional, Callable, Any

from ghoshell_moss.depends import check_agent

if check_agent():
    from ghoshell_moss.agent.console import ChatRenderer
from ghoshell_moss.concepts.speech import Speech, SpeechStream
from ghoshell_common.helpers import uuid


class ChatRenderSpeechStream(SpeechStream):

    def __init__(
            self,
            batch_id: str,
            output: Callable[[str], None],
            *,
            on_start: asyncio.Event,
            close: asyncio.Event,
    ):
        super().__init__(id=batch_id)
        self._output = output
        self._buffered = ""
        self._input_queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._started = False
        self._on_start = on_start
        self._close_event = close
        self._main_loop_task: Optional[asyncio.Task] = None

    def _commit(self) -> None:
        self._input_queue.put_nowait(None)

    async def _main_loop(self):
        try:
            while not self._close_event.is_set():
                item = await self._input_queue.get()
                if item is None:
                    break
                self._output(item)

        except asyncio.CancelledError:
            pass
        finally:
            self._close_event.set()

    def _buffer(self, text: str) -> None:
        if self._started:
            self._input_queue.put_nowait(text)
        if self.cmd_task is not None:
            self.cmd_task.tokens = self._buffered

    async def astart(self) -> None:
        if self._started:
            return
        if len(self._buffered) > 0:
            self._input_queue.put_nowait(self._buffered)
        self._started = True
        self._on_start.set()
        self._main_loop_task = asyncio.create_task(self._main_loop())

    async def aclose(self):
        self.close()

    def close(self) -> None:
        self.commit()
        self._close_event.set()

    def buffered(self) -> str:
        return self._buffered

    async def wait(self) -> None:
        if self._main_loop_task:
            await self._main_loop_task


class ChatRenderSpeech(Speech):

    def __init__(self, render: ChatRenderer):
        self.render = render
        self.last_stream_close_event = asyncio.Event()
        self._outputted = {}
        self._closed_event = asyncio.Event()

    def new_stream(self, *, batch_id: Optional[str] = None) -> SpeechStream:
        batch_id = batch_id or uuid()
        last_stream_close_event = self.last_stream_close_event
        new_close_event = asyncio.Event()
        self.last_stream_close_event = new_close_event
        self._outputted[batch_id] = []

        def _output(item: str):
            self._outputted[batch_id].add_task_with_paths(item)
            self.render.update_ai_response(item)

        return ChatRenderSpeechStream(
            batch_id,
            _output,
            on_start=last_stream_close_event,
            close=new_close_event
        )

    def outputted(self) -> List[str]:
        return list(self._outputted.values())

    async def clear(self) -> List[str]:
        outputted = self.outputted()
        self._outputted.clear()
        self.last_stream_close_event = asyncio.Event()
        return outputted

    async def start(self) -> None:
        pass

    async def close(self) -> None:
        self._closed_event.set()

    async def wait_closed(self) -> None:
        await self._closed_event.wait()
