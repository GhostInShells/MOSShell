from abc import ABC, abstractmethod
from typing import Iterable, Generic, TypeVar, Callable, Protocol

from prompt_toolkit.key_binding.key_bindings import key_binding
from typing_extensions import Self
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import Completer, DynamicCompleter
from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout, Window, HSplit
from prompt_toolkit.widgets import TextArea
from prompt_toolkit.filters import Condition

from ghoshell_moss.host.abcd import MossHost
from ghoshell_moss.message import Message
from ghoshell_moss.core.helpers import ThreadSafeEvent
import asyncio
import uvloop
import contextlib
from prompt_toolkit.layout import ConditionalContainer, AnyContainer

__all__ = ["ReplState", "MossHostRepl", 'Runtime', "RUNTIME"]


class Runtime(Protocol):

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


RUNTIME = TypeVar("RUNTIME", bound=Runtime)


class ReplState(ABC):

    @abstractmethod
    def name(self) -> str:
        """返回 state 的名字. """
        pass

    @abstractmethod
    def as_container(self) -> AnyContainer:
        """
        提供 container 用来做渲染界面.
        """
        pass

    @abstractmethod
    def completer(self) -> Completer:
        """
        提供一个这个状态专属的补完.
        """
        pass

    @abstractmethod
    def on_render(self, callback: Callable[[], None]):
        """注册一个回调, 用来做渲染通知."""
        pass

    @abstractmethod
    def set_alive(self, alive: bool) -> None:
        """接受一个讯号标记进入活跃状态与否. 不一定要用. """
        pass

    @abstractmethod
    def on_input(self, repl_input: str) -> None:
        """执行输入. """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        """允许为 state 建立运行周期. """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class MossHostRepl(Generic[RUNTIME], ABC):

    def __init__(
            self,
            host: MossHost | None = None,
    ):
        self.kb: KeyBindings = KeyBindings()
        self.host: MossHost | None = host or MossHost.discover()
        self.runtime: RUNTIME = self._get_runtime(self.host)
        self._closing_event = ThreadSafeEvent()
        self._exit_command = f"/exit"
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._main_loop_task: asyncio.Task | None = None
        self._render_event = ThreadSafeEvent()
        self._states: dict[str, ReplState] = {}
        # 需要对应 states.
        self._current_state: str = ""
        self.app: Application | None = None

    @classmethod
    @abstractmethod
    def _get_runtime(cls, host: MossHost) -> RUNTIME:
        """从 host 上拿到 runtime 对象. """
        pass

    @abstractmethod
    def create_states(self) -> Iterable[ReplState]:
        """返回当前 repl 拥有的 states. 其中应该包含 default """
        pass

    @abstractmethod
    def farewell(self) -> None:
        """要在界面里输出告别信息. """
        pass

    def output(self, role: str, *messages: Message | str) -> None:
        """提供快速的输出打印"""
        self._check_running()
        # 输出.
        self.host.matrix().session.output(role, *messages)

    def _check_running(self):
        if not self.app:
            raise RuntimeError(f"Not running: {self}")

    def bootstrap(self) -> Application:
        # 1. 构建状态展示区：叠加所有 State 的 ConditionalContainer
        state_containers = [
            ConditionalContainer(
                content=state.as_container(),  # 这里确保 as_container() 返回的是无边框内容
                filter=Condition(lambda s=state: s.name() == self._current_state),
            ) for state in self._states.values()
        ]

        # 2. 动态输入框：设置高度范围
        from prompt_toolkit.layout import Dimension
        input_field = TextArea(
            height=Dimension(min=1, max=5),  # 自增高：最小1行，最大5行
            multiline=False,
            prompt="❯ ",
            accept_handler=lambda buff: self.on_console_input(buff.text),
        )

        # 3. 布局
        root_container = HSplit([
            HSplit(state_containers),  # 状态区，去掉多余包裹
            Window(height=1, char="─", style="class:line"),  # 仅保留一行极简分割
            input_field,
        ])

        return Application(
            layout=Layout(root_container, focused_element=input_field),
            key_bindings=self.kb,
            full_screen=True,
            mouse_support=True,
            erase_when_done=False,
        )

    def bind_keys(self) -> None:
        """定义一个可以修改的函数注册不同的快捷键. """
        kb = self.kb

        @kb.add('c-c')
        def graceful_exit(event) -> None:
            self.close()

        @kb.add('c-n')
        def switch_state(event) -> None:
            # 示意一下切换的绑定.
            state_name = self.switch_state(None)

    def current_state(self) -> ReplState:
        self._check_running()
        return self._states[self._current_state]

    def rerender(self) -> None:
        """立刻触发一次渲染"""
        self._check_running()
        if not self._closing_event.is_set():
            self.app.invalidate()
            self._render_event.clear()

    def switch_state(self, state: str | None) -> str:
        """切换当前状态. """
        current_state = self.current_state()
        if current_state.name() == state:
            return state
        if self._closing_event.is_set():
            return ""
        if state is not None:
            if state not in self._states:
                raise RuntimeError(f"State {state} is not defined")
            self._current_state = state
            current_state.set_alive(False)
            self._states[self._current_state].set_alive(True)
            # 渲染一下.
            self.rerender()
            return self._current_state
        found_current = False
        change_state_name = None
        first_state_name = None
        # 找到下一个.
        for name in self._states:
            if first_state_name is None:
                first_state_name = name
            if found_current:
                change_state_name = name
                break
            if name == self._current_state:
                found_current = True
        if change_state_name is None:
            change_state_name = first_state_name
        return self.switch_state(change_state_name)

    def on_console_input(self, input_line: str) -> bool:
        """提前定义好拿到 command 后的回调"""
        if input_line.rstrip() == self._exit_command:
            self.close()
        else:
            # 接受输入并处理.
            self.current_state().on_input(input_line)
        return False

    async def _main_loop(self) -> None:
        async with contextlib.AsyncExitStack() as stack:
            # 启动 runtime.
            await stack.enter_async_context(self.runtime)
            for state in self._states.values():
                # 启动所有的状态面板.
                await stack.enter_async_context(state)
            list(self._states.values())[0].set_alive(True)
            # 发送一个初始讯号.
            render_task = self._event_loop.create_task(self._render_loop())
            self.switch_state(self._current_state)
            await self.app.run_async()

    async def _render_loop(self) -> None:
        # 初始化第一次.
        while not self._closing_event.is_set():
            await self._render_event.wait()
            self._render_event.clear()
            if self.app:
                self.app.invalidate()
            await asyncio.sleep(0.01)

    async def _run_main(self) -> None:
        self._event_loop = asyncio.get_running_loop()
        self._main_loop_task = self._event_loop.create_task(self._main_loop())
        try:
            await self._main_loop_task
        except asyncio.CancelledError:
            pass

    def close(self) -> None:
        """关闭系统. 可能在运行中被调用. """
        if self._closing_event.is_set():
            return
        self._closing_event.set()
        if self._event_loop and self._main_loop_task:
            if not self._main_loop_task.done():
                # close soon
                self._event_loop.call_soon_threadsafe(self._main_loop_task.cancel)
        if self.app:
            self.app.exit()

    def run(self) -> None:
        """运行到结束"""
        # 绑定快捷键.
        uvloop.install()
        self.bind_keys()
        if self.app is not None:
            raise RuntimeError(f"Already running: {self}")
        # 准备 states.
        # 界面刚进入时, 可能需要有一个固定的 container.
        for state in self.create_states():
            # 注册第一个为 current state
            if not self._current_state:
                self._current_state = state.name()
            self._states[state.name()] = state
            # 注册渲染回调.
            state.on_render(self._render_event.set)
        if self._current_state not in self._states:
            raise RuntimeError(f"Default State {self._current_state} is not defined")
        # 创建 app.
        self.app = self.bootstrap()
        try:
            asyncio.run(self._run_main())
            self.farewell()
        except KeyboardInterrupt:
            # 用来做退出?
            pass
        raise SystemExit(0)
