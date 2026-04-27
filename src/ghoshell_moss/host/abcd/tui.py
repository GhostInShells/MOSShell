from abc import ABC, abstractmethod
from typing import Iterable, Generic, TypeVar, Callable, Protocol, TypeAlias, Any

from prompt_toolkit import PromptSession
from typing_extensions import Self
from rich.console import Console, RenderableType
from rich.traceback import Traceback
from rich.rule import Rule
from rich.text import Text
from rich.syntax import Syntax
from rich.markdown import Markdown
from prompt_toolkit.key_binding import (
    KeyBindings, KeyPressEvent, ConditionalKeyBindings, merge_key_bindings,
    KeyBindingsBase,
)
from prompt_toolkit.completion import Completer, DummyCompleter, DynamicCompleter, Completion, merge_completers
from prompt_toolkit.filters import Condition
from prompt_toolkit import patch_stdout
from ghoshell_moss.core.concepts.session import OutputItem
from ghoshell_moss.host.abcd import IHost
from ghoshell_moss.core.helpers import ThreadSafeEvent
import asyncio
import uvloop
import contextlib
import sys
import threading
import json
from queue import Queue, Empty
from rich.panel import Panel
from rich.table import Table
from rich.console import Group

__all__ = ["TUIState", "MossHostTUI", 'Runtime', "RUNTIME", "ConsoleOutput"]

from prompt_toolkit.styles import Style

DEFAULT_STYLE = Style.from_dict({
    # 提示符区域
    'prompt': 'fg:#61afef bold',  # 蓝色加粗
    'prompt.state': 'fg:#e5c07b bold',  # 黄色，显示状态名
    'prompt.arrow': 'fg:#98c379',  # 绿色箭头

    # 输入行（默认文本）
    '': 'fg:#abb2bf bg:#282c34',  # 主体背景深灰，文字浅灰

    # 多行编辑：行号
    'line-number': 'fg:#5c6370 bg:#1e222a',
    'line-number.current': 'fg:#e5c07b bg:#2c313a bold',

    # 选中文本
    'selected': 'bg:#3e4452',

    # 补全菜单
    'completion-menu': 'bg:#2c323c',
    'completion-menu.completion': 'bg:#2c323c fg:#abb2bf',
    'completion-menu.completion.current': 'bg:#3e4452 fg:#e5c07b bold',
    'completion-menu.meta': 'fg:#5c6370',
    'completion-menu.meta.current': 'fg:#61afef',

    # 滚动条
    'scrollbar': 'bg:#4b5263',
    'scrollbar.button': 'bg:#6c7a8a',

    # 自动建议（灰色斜体）
    'auto-suggestion': 'fg:#5c6370 italic',

    # 搜索高亮
    'search': 'bg:#3d4a5f',

    # 底部工具栏
    'bottom-toolbar': 'bg:#1e222a fg:#abb2bf',
})


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
            queue: asyncio.Queue[list[Renderable]],
    ):
        self._name: str = name
        self._alive_fn = alive
        self._queue = queue

    def rprint(self, *items: Renderable, spacing: bool = True) -> None:
        if not self._alive_fn():
            return
        got_items = list(items)
        if spacing:
            got_items.append('')
        self._queue.put_nowait(got_items)

    def output(self, item: OutputItem) -> None:
        r = self.format_output(item)
        self.rprint('', r)

    def format_output(self, item: OutputItem) -> RenderableType:
        title = Text(f" {item.role.upper()} ", style="bold cyan")

        # 2. 渲染消息体
        content = Text()
        for msg in item.messages:
            # 使用你的 to_content_string()，并添加一点边距感
            content.append(msg.to_content_string() + "\n", style="default")

        # 3. 如果有 log，将其放在最下方 dim 显示
        if item.log:
            # 使用复合样式: 'dim' (亮度调暗) + 'italic' (斜体)
            content.append(f"\nLog: {item.log}", style="dim italic green")

        # 4. 返回带边框的 Panel
        return Panel(
            content,
            title=title,
            title_align="left",
            border_style=f"dim cyan",
            padding=(0, 1),
        )

    def syntax(self, code: str, lexer: str) -> None:
        r = Syntax(
            code,
            lexer,
            theme="ansi_dark",
            background_color="default",  # 关键点：背景透明，不抢终端色
        )
        self.rprint("", r)

    def json(self, value: Any) -> None:
        """统一的 JSON 渲染工厂，使用 ansi_dark 以适配任意终端配色"""
        r = Syntax(
            json.dumps(value, indent=2, ensure_ascii=False),
            "json",
            theme="ansi_dark",
            background_color="default",  # 关键点：背景透明，不抢终端色
        )
        self.rprint("", r)

    def markdown(self, value: str) -> None:
        r = Markdown(value, code_theme="ansi_dark")
        self.rprint(r)

    def hint(self, text: str) -> None:
        """输出一行灰色斜体提示文本（适合辅助信息、帮助文本等）。"""
        hint_text = Text(text, style="dim italic")
        self.rprint(hint_text)

    def info(self, text: str) -> None:
        """输出信息（蓝色，可选信息图标 ℹ️）。"""
        self.rprint(Text(f"ℹ️  {text}", style="bold cyan"))

    def notice(self, text: str) -> None:
        """输出通知/成功消息（绿色，带勾选图标 ✅）。"""
        self.rprint(Text(f"✅  {text}", style="bold green"))

    def error(self, text: str) -> None:
        """输出错误消息（红色，带警告图标 ❌）。"""
        self.rprint(Text(f"❌  {text}", style="bold red"))


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

    @property
    def console(self) -> ConsoleOutput:
        if self._console_output is None:
            raise RuntimeError(f"console output not set")
        return self._console_output

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


class TUICompleter(Completer):
    """处理全局系统级指令"""

    def __init__(self, default_commands: dict[str, str], command_mark: str = '/') -> None:
        self.default_commands = default_commands
        self.command_mark = command_mark

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith(self.command_mark):
            return
        for cmd in self.default_commands:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text), display_meta=self.default_commands[cmd])


class MossHostTUI(Generic[RUNTIME], ABC):

    def __init__(
            self,
            host: IHost | None = None,
            style: Style = None,
    ):
        self.kb: KeyBindingsBase | None = None
        self._style = style or DEFAULT_STYLE
        self.host: IHost | None = host or IHost.discover()
        self.runtime: RUNTIME = self._get_runtime(self.host)
        self._closing_event = ThreadSafeEvent()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._main_loop_task: asyncio.Task | None = None
        # 用子线程实现 print.
        self._renderable_queue: Queue[list[Renderable] | None] = Queue()
        self._console_print_thread = threading.Thread(target=self._main_render_loop, daemon=True)
        self._states: dict[str, TUIState] = {}
        self._main_console_output = ConsoleOutput("", lambda: True, self._renderable_queue)
        # 需要对应 states.
        self._current_state_name: str = ""
        self._prompt_session = PromptSession()
        self._rich_console = Console(
            force_terminal=True,
            color_system='truecolor',
        )
        self._dummy_completer = DummyCompleter()

    def default_commands(self) -> dict[str, tuple[str, Callable[[], None]]]:
        return {
            "/exit": ("exit the tui", lambda: self.close())
        }

    @classmethod
    @abstractmethod
    def _get_runtime(cls, host: IHost) -> RUNTIME:
        """从 host 上拿到 runtime 对象. """
        pass

    @abstractmethod
    def create_states(self) -> Iterable[TUIState]:
        """返回当前 repl 拥有的 states. 其中应该包含 default """
        pass

    def _input_completer(self) -> Completer:
        return self.current_state().completer() or self._dummy_completer

    def welcome(self) -> None:
        # 1. MOSS Banner
        banner = Panel(
            "Welcome to MOSS (Model-Oriented Operating System Shell)\n"
            "[dim]May AI Ghost wondering in the Shells[/dim]",
            style="bold cyan",
            border_style="cyan",
            expand=False
        )

        # 2. Node & Cell Info (打印 Cell 的 to_dict)
        cell_data = self.host.matrix().this.to_dict()
        node_table = Table(title="Current Cell Info", expand=True, box=None)
        node_table.add_column("Property", style="bold yellow")
        node_table.add_column("Value")
        for k, v in cell_data.items():
            node_table.add_row(k, str(v))

        # 3. Environment Context
        env_info = self.host.env.dump_moss_env(with_os_env=False)
        env_table = Table(title="Environment Configuration", expand=True, box=None)
        env_table.add_column("Config", style="bold magenta")
        env_table.add_column("Setting")
        for k, v in env_info.items():
            env_table.add_row(k, str(v))

        # 3. 基础使用指南
        guide = Table(title="Quick Start", expand=True, box=None)
        guide.add_column("Action", style="green")
        guide.add_column("Key / Command")
        guide.add_row("Switch State (Next)", "Ctrl + P")
        guide.add_row("Switch State (Prev)", "Ctrl + B")
        guide.add_row("Interrupt Task", "Esc")
        guide.add_row("Exit System", "/exit")

        # 4. 运行时自定义介绍 (通过抽象方法留给子类实现)
        custom_intro = self._get_custom_intro()

        # 组合渲染
        content = Group(
            banner,
            Panel(env_table, title="[bold]System Info[/bold]", border_style="dim"),
            Panel(guide, title="[bold]Shortcuts[/bold]", border_style="dim"),
            custom_intro if custom_intro else ""
        )

        self._direct_print(content)

    def _get_custom_intro(self) -> RenderableType | None:
        """由子类实现，提供特定 Runtime 的业务介绍。"""
        return None

    def farewell(self) -> None:
        """要在界面里输出告别信息. """
        self._direct_print("good bye")

    def default_key_bindings(self) -> KeyBindings:
        """定义一个可以修改的函数注册不同的快捷键. """
        kb = KeyBindings()

        @kb.add('c-c')
        def graceful_exit(event) -> None:
            self.close()

        # 添加 Shift+Enter 换行逻辑
        @kb.add('c-j')
        def multi_line_enter(event) -> None:
            event.current_buffer.insert_text('\n')

        @kb.add('c-p')
        def switch_next_state(event) -> None:
            if self._event_loop:
                self._event_loop.call_soon_threadsafe(self._switch_to, True)

        @kb.add('c-b')
        def switch_previous_state(event) -> None:
            if self._event_loop:
                self._event_loop.call_soon_threadsafe(self._switch_to, False)

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

    def _direct_print(self, obj: Renderable) -> None:
        if isinstance(obj, OutputItem):
            obj = self.console.format_output(obj)
        self._rich_console.print(obj)

    def _main_render_loop(self) -> None:
        """一个独立的输出线程"""
        while not self._closing_event.is_set():
            while not self._renderable_queue.empty():
                items = self._renderable_queue.get_nowait()
                if items is None:
                    return
                for item in items:
                    self._direct_print(item)
            try:
                items = self._renderable_queue.get(timeout=0.5)
            except Empty:
                continue
            if items is None:
                return
            for item in items:
                self._direct_print(item)

    def _switch_state(self, state_name: str) -> None:
        """切换当前状态. """
        current_state = self.current_state()
        if current_state.name() == state_name:
            return
        if self._closing_event.is_set():
            return
        if state_name is not None:
            if state_name not in self._states:
                raise RuntimeError(f"State {state_name} is not defined")
            old_state_name = current_state.name()
            current_state.on_switch(False)
            self._current_state_name = state_name
            new_state = self._states[state_name]
            new_state_name = state_name
            new_state.on_switch(True)
            # add switch notice.
            notice = Rule(
                f"From State `{old_state_name}` Switch to `{new_state_name}`",
                style="cyan",
                align="center",
            )
            self.console.rprint(notice)
        return

    def _switch_to(self, next_or_previous: bool = True) -> None:
        """切换状态，True 为向后循环，False 为向前循环。"""
        names = list(self._states.keys())
        if not names:
            return
        if len(names) == 1:
            self.console.hint("Only `{}` state exists".format(names[0]))
            return

        current_idx = names.index(self._current_state_name)
        # 计算新的索引 (支持循环)
        offset = 1 if next_or_previous else -1
        new_idx = (current_idx + offset) % len(names)
        self._switch_state(names[new_idx])
        return

    async def _main_loop(self) -> None:
        try:
            self._event_loop = asyncio.get_running_loop()
            async with contextlib.AsyncExitStack() as stack:
                # 启动 runtime.
                await stack.enter_async_context(self.runtime)
                # 启动所有的 state.
                for state in self._states.values():
                    # 启动所有的状态面板.
                    await stack.enter_async_context(state)
                list(self._states.values())[0].on_switch(True)
                # 发送一个初始讯号.
                # render_loop_task = asyncio.create_task(self._main_render_loop())
                input_loop_task = asyncio.create_task(self._input_loop())
                self.current_state().on_switch(True)
                await input_loop_task
        except Exception:
            tb = Traceback()
            self._rich_console.print(tb)
        finally:
            self._closing_event.set()

    async def _input_loop(self) -> None:
        # 绑定快捷键.
        kb_list: list[KeyBindingsBase] = [self.default_key_bindings()]
        for state in self._states.values():
            if kb := state.key_bindings():
                state_kb = ConditionalKeyBindings(
                    kb,
                    Condition(self._is_alive_func(state.name())),
                )
                kb_list.append(state_kb)
            # 合并所有的 key bindings.
        self.kb = merge_key_bindings(kb_list)
        dynamic_completer = DynamicCompleter(self._input_completer)
        default_commands = self.default_commands()
        tui_level_completer = TUICompleter(
            {
                name: value[0]
                for name, value in default_commands.items()
            }
        )
        completer = merge_completers([tui_level_completer, dynamic_completer])

        while not self._closing_event.is_set():
            with patch_stdout.patch_stdout(raw=True):
                item = await self._prompt_session.prompt_async(
                    message=lambda: f' {self._current_state_name}  ❯ ',
                    style=self._style,
                    key_bindings=self.kb,
                    multiline=True,
                    completer=completer,
                    complete_while_typing=True,
                    complete_in_thread=True,
                )
            if not item:
                continue
            if item in default_commands:
                desc, action = default_commands[item]
                action()
                continue
            self.current_state().handle_input(item)

    def close(self) -> None:
        """关闭系统. 可能在运行中被调用. """
        if self._closing_event.is_set():
            return
        self._closing_event.set()
        if self._prompt_session and self._prompt_session.app:
            if self._prompt_session.app.is_running:
                self._prompt_session.app.exit()
        self._rich_console.print("graceful closing...", style="green")

    def _is_alive_func(self, state_name: str) -> Callable[[], bool]:
        def _is_alive() -> bool:
            nonlocal state_name
            return self._current_state_name == state_name

        return _is_alive

    def run(self) -> None:
        """运行到结束"""
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

            #  注册回调.
            state.with_output(output)

        if self._current_state_name not in self._states:
            raise RuntimeError(f"Default State {self._current_state_name} is not defined")
        # 创建 app.
        if sys.platform == 'win32':
            loop = asyncio.new_event_loop()
        else:
            loop = uvloop.new_event_loop()
        try:
            self.welcome()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._main_loop())
            # 等待运行结束
            self._closing_event.set()
            self._renderable_queue.put_nowait(None)
            self._console_print_thread.join()
            self._rich_console.print("closed", style="green")
            self.farewell()
        except KeyboardInterrupt:
            # 用来做退出?
            pass
        except Exception:
            tb = Traceback()
            self._rich_console.print(tb)
        finally:
            loop.close()
            self._closing_event.set()
            raise SystemExit(0)
