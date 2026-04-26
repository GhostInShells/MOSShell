from abc import ABC, abstractmethod
from typing import Coroutine

from prompt_toolkit.completion import Completer
from typing_extensions import Self

from prompt_toolkit.key_binding import KeyPressEvent, KeyBindings

from ghoshell_moss.host.abcd.tui import TUIState
from ghoshell_moss.host.tui.repl_registrar import REPLRegistrar
from rich.traceback import Traceback
import asyncio
import contextlib

__all__ = ["REPLState"]


class REPLState(TUIState, ABC):
    """支持 repl 的测试界面"""

    def __init__(self, name: str):
        self._name = name
        self._is_alive_event = asyncio.Event()
        self._repl_operator: asyncio.Task | None = None
        self._operation_queue: asyncio.Queue[str] = asyncio.Queue()
        self._operation_task: asyncio.Task | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._operation_index: int = 0
        self._main_loop_task: asyncio.Task | None = None
        self._closed = False
        self._repl: REPLRegistrar | None = None

    def name(self) -> str:
        return self._name

    @abstractmethod
    def _create_repl_inspectors(self) -> dict[str, object]:
        """返回提供命令行使用的工具集. """
        pass

    def key_bindings(self) -> KeyBindings | None:
        return None

    def completer(self) -> Completer | None:
        return self._repl

    def on_switch(self, alive: bool) -> None:
        if alive:
            self._is_alive_event.set()
        else:
            self._is_alive_event.clear()

    def on_interrupt(self, event: KeyPressEvent) -> None:
        if self._event_loop and self._operation_task and not self._operation_task.done():
            self.console.hint("canceling operation")
            self._event_loop.call_soon_threadsafe(self._operation_task.cancel)

    def handle_input(self, console_input: str) -> None:
        if not self._is_alive_event.is_set():
            return None
        elif not self._repl or not self._event_loop:
            # can not process any command
            return None
        else:
            self._operation_queue.put_nowait(console_input)
            return None

    @abstractmethod
    async def _on_text_input(self, console_input: str) -> None:
        pass

    async def _operator_loop(self) -> None:
        while not self._closed:
            operator = await self._operation_queue.get()
            if not self._is_alive_event.is_set():
                continue
            try:
                operation = self._operation_task
                if operation is not None and not operation.done():
                    operation.cancel()
                    try:
                        with contextlib.suppress(asyncio.CancelledError):
                            await operation
                    except Exception:
                        tb = Traceback()
                        self.console.rprint(tb)

                if self._repl and self._repl.match(operator):
                    result = self._repl.eval_input(operator)
                    if asyncio.iscoroutine(result):
                        self._create_operation(result)
                        continue
                    else:
                        self._handle_operation_result(result)
                else:
                    self._create_operation(self._on_text_input(operator))
                    continue
            except Exception:
                tb = Traceback()
                self.console.rprint(tb)
                continue

    def _create_operation(self, cor: Coroutine) -> None:
        self._operation_task = self._event_loop.create_task(self._ensure_operation_done(cor))

    async def _ensure_operation_done(self, cor: Coroutine) -> None:
        self._operation_index += 1
        index = self._operation_index
        self.console.hint("operation {} started".format(index))
        try:
            r = await cor
            self._handle_operation_result(r)
            self.console.hint("operation {} done".format(index))
        except asyncio.CancelledError:
            self.console.hint("operation {} cancelled".format(index))
        except Exception:
            self.console.hint("operation {} failed".format(index))
            tb = Traceback()
            self.console.rprint(tb)

    def _handle_operation_result(self, result) -> None:
        if result is None:
            return
        if hasattr(result, "__rich__") or hasattr(result, "__rich_console__"):
            self.console.rprint(result)
        elif isinstance(result, str):
            self.console.rprint(result)
        # 增加对 dict/list 等复杂类型的 JSON 格式化支持
        elif isinstance(result, (dict, list)):
            try:
                self.console.json(result)
            except Exception:
                value = "%r" % result
                self.console.rprint(value)
            return
        else:
            self.console.rprint(str(result))

    async def __aenter__(self) -> Self:
        inspectors = self._create_repl_inspectors()
        if len(inspectors) > 0:
            self._repl: REPLRegistrar = REPLRegistrar(inspectors)
        self._event_loop = asyncio.get_running_loop()
        self._main_loop_task = self._event_loop.create_task(self._operator_loop())

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._closed = True
        if self._operation_task is not None and not self._operation_task.done():
            self._operation_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._operation_task
        if self._main_loop_task and not self._main_loop_task.done():
            self._main_loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._main_loop_task
