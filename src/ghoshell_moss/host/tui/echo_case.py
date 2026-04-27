from typing import Callable, Iterable, Self
from prompt_toolkit.completion import WordCompleter, Completer
from prompt_toolkit.widgets import TextArea, Frame
from prompt_toolkit.key_binding import KeyPressEvent
from ghoshell_moss.host.abcd.tui import TUIState, MossHostTUI, RUNTIME, Runtime
from ghoshell_moss.host.abcd import IHost
import asyncio
import contextlib


class EchoState(TUIState):
    def __init__(self, name: str):
        self._name = name
        self._is_alive = False
        self._completer = WordCompleter([f"hello_{name}", "echo", "status"])
        self._render_callback = None
        # 内部显示区
        self._display = TextArea(text=f"Welcome to {name} State\n", read_only=True)
        self._main_task: asyncio.Task | None = None
        self._last_input = None
        self._interrupted = False

    def name(self) -> str:
        return self._name

    def completer(self) -> Completer | None:
        return self._completer

    def on_switch(self, alive: bool) -> None:
        self._is_alive = alive

    def on_interrupt(self, event: KeyPressEvent) -> None:
        self._interrupted = True
        self.rprint("interrupted")

    def handle_input(self, console_input: str) -> None:
        self._last_input = console_input
        self._interrupted = False
        self.rprint("> echo: " + console_input)

    async def _echo_as_hell(self):
        while True:
            if self._is_alive and self._last_input and not self._interrupted:
                self.rprint("> " + self._last_input)
            await asyncio.sleep(1)

    async def __aenter__(self) -> Self:
        self._main_task = asyncio.create_task(self._echo_as_hell())

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._main_task and not self._main_task.done():
            self._main_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._main_task


class FakeRuntime(Runtime):

    async def __aenter__(self) -> Self:
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class EchoCase(MossHostTUI):

    @classmethod
    def _get_runtime(cls, host: IHost) -> RUNTIME:
        return FakeRuntime()

    def create_states(self) -> Iterable[TUIState]:
        return [
            EchoState("A"),
            EchoState("B"),
        ]


if __name__ == "__main__":
    repl = EchoCase()
    repl.run()
