from typing import Callable, Iterable, Self
from prompt_toolkit.completion import WordCompleter, Completer
from prompt_toolkit.layout import AnyContainer, Window
from prompt_toolkit.widgets import TextArea, Frame
from ghoshell_moss.host.repl.abcd import ReplState, MossHostRepl, RUNTIME, Runtime
from ghoshell_moss.host.abcd import MossHost
from prompt_toolkit.layout import Dimension


class EchoState(ReplState):
    def __init__(self, name: str, repl: MossHostRepl):
        self._name = name
        self._repl = repl
        self._is_alive = False
        self._completer = WordCompleter([f"hello_{name}", "echo", "status"])
        self._render_callback = None

        # 内部显示区
        self._display = TextArea(text=f"Welcome to {name} State\n", read_only=True)

    def name(self) -> str: return self._name

    def as_container(self) -> AnyContainer:
        return Window(
            content=self._display.control,  # 直接使用 control，不要包裹在组件里
            height=Dimension(weight=1),  # weight=1 表示占用分配的所有剩余空间
            wrap_lines=True  # 如果内容太长，自动换行
        )

    def completer(self) -> Completer: return self._completer

    def on_render(self, callback: Callable[[], None]): self._render_callback = callback

    def set_alive(self, alive: bool) -> None: self._is_alive = alive

    def on_input(self, repl_input: str) -> None:
        # 回显逻辑
        self._display.text += f"> {repl_input}\n"
        # self._repl.output("system", f"[{self._name}] received: {repl_input}")
        # if self._render_callback: self._render_callback()

    async def __aenter__(self): return self

    async def __aexit__(self, exc, val, tb): pass


class FakeRuntime(Runtime):

    async def __aenter__(self) -> Self:
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class EchoCase(MossHostRepl):

    @classmethod
    def _get_runtime(cls, host: MossHost) -> RUNTIME:
        return FakeRuntime()

    def create_states(self) -> Iterable[ReplState]:
        return [
            EchoState("A", self),
            EchoState("B", self),
        ]

    def farewell(self):
        print("Goodbye!")


if __name__ == "__main__":
    repl = EchoCase()
    repl.run()
