"""TUI 框架基类 — MossHostTUI, TUIState, TuiRender, LiveStreamSink."""
from abc import ABC, abstractmethod
from collections import deque
from typing import Iterable, Generic, TypeVar, Callable, Protocol, TypeAlias, Any, Optional
from prompt_toolkit import PromptSession
from typing_extensions import Self
from rich.console import Console, RenderableType
from rich.traceback import Traceback, Trace
from rich.rule import Rule
from rich.text import Text
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.theme import Theme
from prompt_toolkit.key_binding import (
    KeyBindings, KeyPressEvent, ConditionalKeyBindings, merge_key_bindings,
    KeyBindingsBase,
)
from prompt_toolkit.completion import Completer, DummyCompleter, DynamicCompleter, Completion, merge_completers
from prompt_toolkit.filters import Condition
from prompt_toolkit import patch_stdout
from ghoshell_moss.core.blueprint.session import OutputItem
from ghoshell_moss.core.blueprint.host import MossHost
from ghoshell_moss.core.helpers import ThreadSafeEvent
import asyncio
import sys
if sys.platform != "win32":
    import uvloop
import contextlib
import sys
import threading
import json
import os
import janus
from queue import Queue, Empty
from rich.panel import Panel
from rich.table import Table
from rich.console import Group

__all__ = [
    "TUIState", "MossHostTUI", 'Runtime', "RUNTIME", "TuiRender",
    "Renderable", "OutputItem", "LiveStreamSink",
    "ConsoleOutput",  # backward compatibility
]

from prompt_toolkit.styles import Style

DEFAULT_PROMPT_STYLE = Style.from_dict({
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


class LiveStreamSink:
    """跨 asyncio/sync 边界的流式文本输出槽.

    asyncio 侧: await send(delta) / send_nowait(delta) → janus async_q
    渲染线程: render(console) 被 _direct_print 通过 duck-type 调用

    首次 render 进入 live 模式: 实时消费 janus queue 并攒 Segment buffer,
    粘字符串聚合减少 console.print 调用次数.
    后续 render 直接回放 buffer (支持 re-render / state 切换后重建).
    """

    def __init__(
            self,
            rich_print_kwargs: dict[str, Any] | None = None,
    ):
        self._queue = janus.Queue[str | None]()
        self._rich_print_kwargs = rich_print_kwargs or {}
        self._committed = False
        self._render_count = 0
        self._buffer: list[str] = []

    async def __aenter__(self):
        return self

    async def send(self, delta: str) -> None:
        if self._committed:
            return
        try:
            await self._queue.async_q.put(delta)
        except janus.AsyncQueueShutDown:
            pass

    def send_nowait(self, delta: str) -> None:
        if self._committed:
            return
        try:
            self._queue.sync_q.put_nowait(delta)
        except janus.SyncQueueShutDown:
            pass

    def commit(self) -> None:
        """标记流结束，发送 None sentinel 通知渲染端."""
        if self._committed:
            return
        self._committed = True
        try:
            self._queue.sync_q.put_nowait(None)
        except janus.SyncQueueShutDown:
            pass

    async def close(self) -> None:
        """安全关闭: 确保 committed + sentinel 入队后排空."""
        if not self._committed:
            self._committed = True
            try:
                self._queue.sync_q.put_nowait(None)
            except janus.SyncQueueShutDown:
                pass
        self._queue.shutdown(immediate=False)

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    def render(self, console: Console) -> None:
        if self._render_count > 0:
            if self._buffer:
                console.print(Panel(
                    Text("".join(self._buffer)),
                    title=" RESPONSE ",
                    title_align="left",
                    border_style="cyan",
                ))
            return

        self._buffer: list[str] = []
        pending: list[str] = []
        rendered_lines = 0

        def _render_panel() -> None:
            nonlocal rendered_lines
            if rendered_lines > 0:
                # 光标上移到上一个 panel 的起始行 + 清屏
                console.file.write(f"\033[{rendered_lines}F")
                console.file.write("\033[J")
            panel = Panel(
                Text("".join(self._buffer)),
                title=" RESPONSE ",
                title_align="left",
                border_style="cyan",
            )
            with console.capture() as capture:
                console.print(panel)
            output = capture.get()
            rendered_lines = output.count('\n')
            console.file.write(output)
            console.file.flush()

        try:
            while True:
                item = self._queue.sync_q.get()
                if item is None:
                    break
                pending.append(item)
                if self._queue.sync_q.empty():
                    self._buffer.append("".join(pending))
                    pending.clear()
                    _render_panel()
            if pending:
                self._buffer.append("".join(pending))
                pending.clear()
            _render_panel()
        except janus.SyncQueueShutDown:
            pass
        finally:
            console.print("")
            self._render_count += 1


Renderable: TypeAlias = RenderableType | OutputItem


class TuiRender:
    """Per-state render output manager with ring buffer and global notification channel.

    Replaces ConsoleOutput. Adds:
    - Ring buffer (deque) so non-current-state output is not discarded
    - urgent parameter for approval-level global notifications
    - replay_buffer() for state-switch history replay
    """

    def __init__(
            self,
            name: str,
            alive: Callable[[], bool],
            queue: asyncio.Queue[list[Renderable]],
            clear_func: Callable[[], None],
            buffer_size: int = 200,
            buffer_enabled: bool = True,
            buffer_rprint: bool = True,
            on_urgent: Callable[[str], None] | None = None,
    ):
        self._name: str = name
        self._alive_fn = alive
        self._queue = queue
        self._clear_fn = clear_func
        self._buffer: deque[list[Renderable]] = deque(maxlen=buffer_size)
        self._buffer_enabled = buffer_enabled
        self._buffer_rprint = buffer_rprint
        self._on_urgent = on_urgent
        self._bottom_text: str = ""

    def clear(self) -> None:
        self._clear_fn()

    def rprint(self, *items: Renderable, spacing: bool = True, urgent: bool = False) -> None:
        got_items = list(items)
        if spacing:
            got_items.append('')

        if self._buffer_enabled and (urgent or self._buffer_rprint):
            self._buffer.append(got_items)

        if urgent and self._on_urgent:
            self._on_urgent(self._urgent_text(got_items))

        if not self._alive_fn():
            return

        self._queue.put_nowait(got_items)

    def _urgent_text(self, items: list[Renderable]) -> str:
        """Convert urgent items to a compact single-line text for bottom toolbar."""
        parts: list[str] = []
        for item in items:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Text):
                parts.append(item.plain)
            elif hasattr(item, '__rich__'):
                parts.append(str(item))
        return " | ".join(p for p in parts if p)

    def replay_buffer(self) -> None:
        """Replay all buffered items through the render queue."""
        if not self._buffer_enabled:
            return
        # Drain pending items before replay
        self._clear_fn()
        for items in self._buffer:
            self._queue.put_nowait(list(items))

    def buffer_clear(self) -> None:
        """Clear the ring buffer without replaying."""
        self._buffer.clear()

    @property
    def buffer_size(self) -> int:
        return self._buffer.maxlen  # type: ignore[return-value]

    def set_bottom(self, text: str) -> None:
        """Set the bottom toolbar text (for approval counts, alerts, etc.)."""
        self._bottom_text = text

    def get_bottom_toolbar(self) -> Callable[[], str]:
        """Return a callable for prompt_toolkit's bottom_toolbar parameter."""
        return lambda: self._bottom_text

    def output(self, item: OutputItem) -> None:
        for i in self.format_output(item):
            self.rprint('', i)

    def format_output(self, item: OutputItem) -> Iterable[RenderableType]:
        contents = []
        for msg in item.messages:
            contents.append(msg.to_content_string())

        title = Text(f" {item.role.upper()} ", style="bold cyan")

        message_content = Syntax(
            "\n".join(contents),
            'xml',
            theme="ansi_dark",
            background_color="default",  # 关键点：背景透明，不抢终端色
            word_wrap=True,
        )
        yield Panel(
            message_content,
            title=title,
            title_align="left",
            border_style=f"dim cyan",
            padding=(0, 1),
        )

        # 3. 如果有 log，将其放在最下方 dim 显示
        if item.log:
            # 使用复合样式: 'dim' (亮度调暗) + 'italic' (斜体)
            yield Text(f"Log: {item.log}", style="dim italic green")

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
        if not isinstance(value, str):
            value = json.dumps(value, indent=2, ensure_ascii=False)
        r = Syntax(
            value,
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

    def print_exception(
            self,
            trace: Trace | None = None,
            *,
            width: Optional[int] = 100,
            extra_lines: int = 3,
            max_frames: int = 10,
    ) -> None:
        """Prints a rich render of the last exception and traceback.
        """

        traceback = Traceback(
            trace,
            width=width,
            extra_lines=extra_lines,
            word_wrap=True,
            show_locals=True,
            max_frames=max_frames,
        )
        self.rprint(traceback)


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

    _console_output: "TuiRender | None" = None

    def with_output(self, output: "TuiRender") -> None:
        """注册一个回调, 用来做渲染通知."""
        self._console_output = output

    @property
    def console(self) -> "TuiRender":
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
        text = text[len(self.command_mark):]
        for cmd in self.default_commands:
            if cmd.startswith(text):
                yield Completion(cmd, start_position=-len(text), display_meta=self.default_commands[cmd])


class MossHostTUI(Generic[RUNTIME], ABC):

    def __init__(
            self,
            host: MossHost | None = None,
            prompt_style: Style = None,
    ):
        self.kb: KeyBindingsBase | None = None
        self._style = prompt_style or DEFAULT_PROMPT_STYLE
        self.host: MossHost | None = host or MossHost.discover()
        self.runtime: RUNTIME = self._get_runtime()
        self._closing_event = ThreadSafeEvent()
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._main_loop_task: asyncio.Task | None = None
        # 用子线程实现 print.
        self._renderable_queue: Queue[list[Renderable] | None] = Queue()
        self._console_print_thread = threading.Thread(target=self._main_render_loop, daemon=True)
        self._states: dict[str, TUIState] = {}
        # 需要对应 states.
        self._current_state_name: str = ""
        self._prompt_session = PromptSession()
        self._rich_console = Console(
            force_terminal=True,
            color_system='truecolor',
            theme=Theme({
                "traceback.border": "bright_black",
                "traceback.text": "white",
                "traceback.title": "bold red",
                "traceback.item": "cyan",
            })
        )
        self._main_console_output = TuiRender("", lambda: True, self._renderable_queue, self.clear_console)
        self._bottom_toolbar_text: str = ""
        self._dummy_completer = DummyCompleter()

    def clear_console(self) -> None:
        """clear rich console"""
        _queue = self._renderable_queue
        while not _queue.empty():
            try:
                _queue.get_nowait()
            except Empty:
                break
        self._rich_console.clear()
        _queue.put_nowait(None)

    def _drain_render_queue(self) -> None:
        """Drain all pending items from the render queue without clearing console."""
        _queue = self._renderable_queue
        while not _queue.empty():
            try:
                _queue.get_nowait()
            except Empty:
                break

    def default_commands(self) -> dict[str, tuple[str, Callable[[], None]]]:
        return {
            "exit": ("exit the tui", lambda: self.close())
        }

    @abstractmethod
    def _get_runtime(self) -> RUNTIME:
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
            "[dim]May model Ghost wandering in the Shells[/dim]",
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
        env_table.add_row("SELF_PID", str(os.getpid()))

        # 3. 基础使用指南
        guide = Table(title="Quick Start", expand=True, box=None)
        guide.add_column("Action", style="green")
        guide.add_column("Key / Command")
        guide.add_row("Switch State", "Ctrl + T")
        guide.add_row("Emergency Stop", "Ctrl + G")
        guide.add_row("Add New Line", "Ctrl + J")
        guide.add_row("Interrupt Task", "Esc")
        guide.add_row("REPL command", "Start with /")
        guide.add_row("REPL help", "Start with ?")
        guide.add_row("Exit System", "/exit")

        # 4. 运行时自定义介绍 (通过抽象方法留给子类实现)
        custom_intro = self._get_custom_intro()

        # 5. 系统告警 — 启动时非致命错误一览
        alerts = self._get_system_alerts()

        # 组合渲染
        content = Group(
            banner,
            Panel(node_table, title="[bold]Current Matrix Cell[/bold]", border_style="dim"),
            Panel(env_table, title="[bold]System Info[/bold]", border_style="dim"),
            Panel(guide, title="[bold]Shortcuts[/bold]", border_style="dim"),
            custom_intro if custom_intro else "",
            alerts if alerts else "",
        )

        self._direct_print(content)

    def _get_custom_intro(self) -> RenderableType | None:
        """由子类实现，提供特定 Runtime 的业务介绍。"""
        return None

    def _get_system_alerts(self) -> RenderableType | None:
        """启动时系统告警 — 默认展示 manifests scan errors。子类可追加。

        TODO: 未来 system alerts 应在 matrix 或 TopicService ringbuffer
        中作为原生模块实现，TUI 订阅而非直接读 host.scan_errors。
        """
        errors = self.host.scan_errors
        if not errors:
            return None

        err_table = Table(box=None, expand=True)
        err_table.add_column("Module", style="bold red")
        err_table.add_column("Stage")
        err_table.add_column("Error", style="yellow")
        for err in errors:
            err_table.add_row(
                err.module_path,
                err.stage,
                f"{type(err.exception).__name__}: {err.exception}",
            )
        return Panel(
            err_table,
            title=f"[bold yellow]! Scan Warnings ({len(errors)})[/bold yellow]",
            border_style="yellow",
        )

    def _on_emergency_pause(self) -> None:
        """急停 hook — 子类 override 实现具体 pause/resume 逻辑."""
        pass

    def _prompt_status(self) -> list[tuple[str, str]]:
        """返回 prompt 前的状态标记 (FormattedText 元组列表)。子类 override 追加."""
        return []

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

        @kb.add('c-t')
        def toggle_state(event) -> None:
            if self._event_loop:
                self._event_loop.call_soon_threadsafe(self._toggle_state)

        @kb.add('escape')
        def interrupt(event) -> None:
            # notify interruption
            if self._event_loop:
                self._event_loop.call_soon_threadsafe(self.current_state().on_interrupt, event)

        @kb.add('enter')
        def accept(event) -> None:
            event.current_buffer.validate_and_handle()

        @kb.add('c-g')
        def emergency_pause(event) -> None:
            if self._event_loop:
                self._event_loop.call_soon_threadsafe(self._on_emergency_pause)

        return kb

    def current_state(self) -> TUIState:
        return self._states[self._current_state_name]

    @property
    def console(self) -> TuiRender:
        return self._main_console_output

    def set_bottom_toolbar(self, text: str) -> None:
        """Set the bottom toolbar text (e.g. approval count, system alerts)."""
        self._bottom_toolbar_text = text

    def get_bottom_toolbar(self) -> Callable[[], str]:
        """Return a callable for prompt_toolkit's bottom_toolbar parameter.

        动态拼接: state 切换提示 + urgent 通知.
        """
        def _build() -> str:
            parts: list[str] = []
            names = list(self._states.keys())
            if len(names) > 1:
                current = self._current_state_name
                state_parts = []
                for name in names:
                    marker = "●" if name == current else "○"
                    state_parts.append(f"{marker} {name}")
                parts.append("  C-t  ".join(state_parts))
            if self._bottom_toolbar_text:
                parts.append(self._bottom_toolbar_text)
            return "  ▏ ".join(parts) if parts else ""
        return _build

    def _direct_print(self, obj: Renderable) -> None:
        try:
            if isinstance(obj, OutputItem):
                for item in self.console.format_output(obj):
                    self._rich_console.print(item)
            elif isinstance(obj, LiveStreamSink):
                obj.render(self._rich_console)
            else:
                self._rich_console.print(obj)
        except Exception:
            try:
                self._rich_console.print_exception()
            except Exception:
                pass

    def _main_render_loop(self) -> None:
        """一个独立的输出线程"""
        while not self._closing_event.is_set():
            # non-blocking drain
            while not self._renderable_queue.empty():
                items = self._renderable_queue.get_nowait()
                if items is None:
                    continue
                for item in items:
                    self._direct_print(item)
            if self._closing_event.is_set():
                break
            # blocking wait with timeout
            try:
                items = self._renderable_queue.get(timeout=0.5)
            except Empty:
                continue
            if items is None:
                continue
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
            # replay buffered history from the new state
            if new_state.console is not None:
                new_state.console.replay_buffer()
            # add switch notice after replay so it sits at the bottom
            notice = Rule(
                f"From State `{old_state_name}` Switch to `{new_state_name}`",
                style="cyan",
                align="center",
            )
            self.console.rprint(notice)
        return

    def _toggle_state(self) -> None:
        """C-t: 循环切到下一个 state."""
        names = list(self._states.keys())
        if len(names) <= 1:
            return
        current_idx = names.index(self._current_state_name) if self._current_state_name in names else 0
        next_idx = (current_idx + 1) % len(names)
        self._switch_state(names[next_idx])


    async def _main_loop(self) -> None:
        try:
            self._event_loop = asyncio.get_running_loop()
            async with contextlib.AsyncExitStack() as stack:
                # 启动 runtime.
                await stack.enter_async_context(self.runtime)
                # welcome after runtime initialized.
                self.welcome()
                # 启动所有的 state.
                for state in self._states.values():
                    # 启动所有的状态面板.
                    await stack.enter_async_context(state)
                list(self._states.values())[0].on_switch(True)
                # 发送一个初始讯号.
                input_loop_task = asyncio.create_task(self._input_loop())
                self.current_state().on_switch(True)
                await input_loop_task
        except Exception:
            self.console.print_exception()
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
                    message=lambda: self._prompt_status() + [
                        ("class:prompt.state", self._current_state_name),
                        ("", "  "),
                        ("class:prompt.arrow", "❯ "),
                    ],
                    style=self._style,
                    key_bindings=self.kb,
                    multiline=True,
                    completer=completer,
                    complete_while_typing=True,
                    complete_in_thread=True,
                    bottom_toolbar=self.get_bottom_toolbar(),
                )
            if not item:
                continue
            # default command check
            command_line = item.lstrip('/')
            if command_line in default_commands:
                desc, action = default_commands[command_line]
                try:
                    action()
                except Exception:
                    self.console.print_exception()
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
                self.console.info("current state is %s" % self._current_state_name)
            self._states[state.name()] = state
            # 注册管理回调.
            output = TuiRender(
                state.name(),
                self._is_alive_func(state.name()),
                self._renderable_queue,
                self._drain_render_queue,
                on_urgent=self.set_bottom_toolbar,
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

            loop.run_until_complete(self._main_loop())
            loop.set_exception_handler(self.tui_exception_handler)
            # 等待运行结束
            self._closing_event.set()
            self._console_print_thread.join()
            self._rich_console.print("closed", style="green")
            self.farewell()
        except KeyboardInterrupt:
            # 用来做退出?
            pass
        except Exception:
            self._rich_console.print_exception()
        finally:
            loop.close()
            self._closing_event.set()
            raise SystemExit(0)

    def tui_exception_handler(self, loop: asyncio.AbstractEventLoop, context: dict):
        # 1. 提取异常对象
        exception = context.get("exception")
        message = context.get("message", "Unhandled exception in event loop")
        self.console.error(message)
        if self.host.matrix().is_running():
            self.host.matrix().logger.exception("%s: %s", message, exception)


# backward compatibility
ConsoleOutput = TuiRender
