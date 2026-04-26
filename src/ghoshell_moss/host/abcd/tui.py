import threading
from abc import ABC, abstractmethod
from typing import Iterable, Generic, TypeVar, Callable, Protocol, TypeAlias

from prompt_toolkit import PromptSession
from typing_extensions import Self
from rich.console import Console, RenderableType
from rich.traceback import Traceback
from prompt_toolkit.key_binding import (
    KeyBindings, KeyPressEvent, ConditionalKeyBindings, merge_key_bindings,
    KeyBindingsBase,
)
from prompt_toolkit.completion import Completer, DummyCompleter
from prompt_toolkit.filters import Condition
from prompt_toolkit import patch_stdout
from ghoshell_moss.core.concepts.session import OutputItem
from ghoshell_moss.host.abcd import MossHost
from ghoshell_moss.core.helpers import ThreadSafeEvent
import asyncio
import uvloop
import contextlib
from queue import Queue, Empty

__all__ = ["TUIState", "MossHostTUI", 'Runtime', "RUNTIME"]


class Runtime(Protocol):

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


RUNTIME = TypeVar("RUNTIME", bound=Runtime)

Renderable: TypeAlias = RenderableType | OutputItem


class ConsoleOutput:
    """可以共享 output 能力的模块. """

    def __init__(
            self,
            name: str,
            alive: Callable[[], bool],
            queue: Queue[Renderable],
    ):
        self._name: str = name
        self._alive_fn = alive
        self._queue: Queue[Renderable] = queue

    def rprint(self, item: Renderable) -> None:
        if not self._alive_fn():
            return
        self._queue.put_nowait(item)


class TUIState(ABC):

    @abstractmethod
    def name(self) -> str:
        """返回 state 的名字. """
        pass

    def completer(self) -> Completer | None:
        """
        提供一个这个状态专属的补完.
        """
        return None

    def key_bindings(self) -> KeyBindings | None:
        return None

    _console_output = None

    def with_output(self, output: ConsoleOutput) -> None:
        """注册一个回调, 用来做渲染通知."""
        self._console_output = output

    def rprint(self, item: Renderable) -> None:
        if self._console_output:
            self._console_output.rprint(item)

    @abstractmethod
    def on_switch(self, alive: bool) -> None:
        """接受一个讯号标记进入活跃状态与否. 不一定要用. """
        pass

    @abstractmethod
    def on_interrupt(self, event: KeyPressEvent) -> None:
        pass

    @abstractmethod
    def handle_input(self, console_input: str) -> None:
        """执行输入. """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """允许为 state 建立运行周期. """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MossHostTUI(Generic[RUNTIME], ABC):

    def __init__(
            self,
            host: MossHost | None = None,
    ):
        self.kb: KeyBindingsBase | None = None
        self.host: MossHost | None = host or MossHost.discover()
        self.runtime: RUNTIME = self._get_runtime(self.host)
        self._closing_event = ThreadSafeEvent()
        self._exit_command = f"/exit"
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._main_loop_task: asyncio.Task | None = None
        # 用子线程实现 print.
        self._renderable_queue: Queue[Renderable] = Queue()
        self._console_print_thread = threading.Thread(target=self._main_print_loop, daemon=True)
        self._states: dict[str, TUIState] = {}
        self._main_console_output = ConsoleOutput("", lambda: True, self._renderable_queue)
        # 需要对应 states.
        self._current_state_name: str = ""
        self._input_field = None
        self._console = Console(
        )
        self._prompt_session = PromptSession()
        self._dummy_completer = DummyCompleter()

    @classmethod
    @abstractmethod
    def _get_runtime(cls, host: MossHost) -> RUNTIME:
        """从 host 上拿到 runtime 对象. """
        pass

    @abstractmethod
    def create_states(self) -> Iterable[TUIState]:
        """返回当前 repl 拥有的 states. 其中应该包含 default """
        pass

    def _input_completer(self) -> Completer:
        return self.current_state().completer() or self._dummy_completer

    def welcome(self) -> None:
        self._rprint("hello world")

    def farewell(self) -> None:
        """要在界面里输出告别信息. """
        self._rprint("good bye")

    def default_key_bindings(self) -> KeyBindings:
        """定义一个可以修改的函数注册不同的快捷键. """
        kb = KeyBindings()

        @kb.add('c-c')
        def graceful_exit(event) -> None:
            self.close()

        @kb.add('c-n')
        def switch_next_state(event) -> None:
            if self._event_loop:
                self._event_loop.call_soon_threadsafe(self.switch_to, True)

        @kb.add('c-b')
        def switch_previous_state(event) -> None:
            if self._event_loop:
                self._event_loop.call_soon_threadsafe(self.switch_to, False)

        @kb.add('escape')
        def interrupt(event) -> None:
            # notify interruption
            if self._event_loop:
                self._event_loop.call_soon_threadsafe(self.current_state().on_interrupt, event)

        @kb.add('enter')
        def accept(event) -> None:
            event.current_buffer.validate_and_handle()

        return kb

    def current_state(self) -> TUIState:
        return self._states[self._current_state_name]

    @property
    def console(self) -> ConsoleOutput:
        return self._main_console_output

    def _rprint(self, obj: Renderable) -> None:
        if isinstance(obj, OutputItem):
            obj = f"> {obj.role}\n\n" + "\n".join([msg.to_content_string() for msg in obj.messages])
        self._console.print(obj)

    def _main_print_loop(self) -> None:
        """一个独立的输出线程"""
        while not self._closing_event.is_set():
            while not self._renderable_queue.empty():
                item = self._renderable_queue.get_nowait()
                self._rprint(item)
            try:
                item = self._renderable_queue.get(block=True, timeout=0.5)
                self._rprint(item)
            except Empty:
                continue

    def switch_state(self, state_name: str) -> None:
        """切换当前状态. """
        current_state = self.current_state()
        if current_state.name() == state_name:
            return
        if self._closing_event.is_set():
            return
        if state_name is not None:
            if state_name not in self._states:
                raise RuntimeError(f"State {state_name} is not defined")
            current_state.on_switch(False)
            self._current_state_name = state_name
            new_state = self._states[state_name]
            new_state.on_switch(True)
            # add switch notice.
            notice = f"> from state {current_state.name()} to {state_name}"
            self.console.rprint(notice)
        return

    def switch_to(self, next_or_previous: bool = True) -> None:
        """切换状态，True 为向后循环，False 为向前循环。"""
        names = list(self._states.keys())
        if not names:
            return

        current_idx = names.index(self._current_state_name)
        # 计算新的索引 (支持循环)
        offset = 1 if next_or_previous else -1
        new_idx = (current_idx + offset) % len(names)
        self.switch_state(names[new_idx])
        return

    async def _main_loop(self) -> None:
        try:
            async with contextlib.AsyncExitStack() as stack:
                # 启动 runtime.
                await stack.enter_async_context(self.runtime)
                # 启动所有的 state.
                for state in self._states.values():
                    # 启动所有的状态面板.
                    await stack.enter_async_context(state)
                list(self._states.values())[0].on_switch(True)
                # 发送一个初始讯号.
                self.switch_state(self._current_state_name)
                await  self._input_loop()
        except asyncio.CancelledError:
            pass
        except Exception:
            tb = Traceback()
            self._console.print(tb)
        finally:
            self._closing_event.set()

    async def _input_loop(self) -> None:
        with patch_stdout.patch_stdout():
            while not self._closing_event.is_set():
                item = await self._prompt_session.prompt_async(
                    # 增加一个漂亮的底色分隔符或特殊的 prompt 符号
                    message=lambda: f' {self._current_state_name}  ❯ ',
                    multiline=False,
                    completer=self._input_completer(),
                    key_bindings=self.kb,
                )
                if item == self._exit_command:
                    self._closing_event.set()
                    return
                self.current_state().handle_input(item)

    async def _run_main(self) -> None:
        self._event_loop = asyncio.get_running_loop()
        # task 化, 方便 cancel.
        self._main_loop_task = self._event_loop.create_task(self._main_loop())
        with contextlib.suppress(asyncio.CancelledError):
            await self._main_loop_task

    def close(self) -> None:
        """关闭系统. 可能在运行中被调用. """
        if self._closing_event.is_set():
            return
        self._closing_event.set()
        if self._event_loop and self._main_loop_task:
            if not self._main_loop_task.done():
                # close soon
                self._event_loop.call_soon_threadsafe(self._main_loop_task.cancel)

    def _is_alive_func(self, state_name: str) -> Callable[[], bool]:
        def _is_alive() -> bool:
            nonlocal state_name
            return self._current_state_name == state_name

        return _is_alive

    def run(self) -> None:
        """运行到结束"""
        # 绑定快捷键.
        kb_list: list[KeyBindingsBase] = [self.default_key_bindings()]
        # 启动渲染循环.
        self._console_print_thread.start()
        # 准备 states.
        # 界面刚进入时, 可能需要有一个固定的 container.
        for state in self.create_states():
            # 注册第一个为 current state
            if not self._current_state_name:
                self._current_state_name = state.name()
            self._states[state.name()] = state
            # 注册管理回调.
            output = ConsoleOutput(
                state.name(),
                self._is_alive_func(state.name()),
                self._renderable_queue,
            )
            if kb := state.key_bindings():
                state_kb = ConditionalKeyBindings(
                    kb,
                    Condition(self._is_alive_func(state.name())),
                )
                kb_list.append(state_kb)
            #  注册回调.
            state.with_output(output)
        # 合并所有的 key bindings.
        self.kb = merge_key_bindings(kb_list)

        if self._current_state_name not in self._states:
            raise RuntimeError(f"Default State {self._current_state_name} is not defined")
        self.current_state().on_switch(True)
        # 创建 app.
        loop = uvloop.new_event_loop()
        try:
            self.welcome()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run_main())
            self.farewell()
        except KeyboardInterrupt:
            # 用来做退出?
            pass
        except Exception:
            tb = Traceback()
            self._console.print(tb)
        finally:
            loop.close()
            self._closing_event.set()
            self._console_print_thread.join()
            raise SystemExit(0)
