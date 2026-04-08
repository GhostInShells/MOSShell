import sys
import subprocess
import asyncio
import importlib
from typing import Iterable, Optional, List, Tuple, Type, Any, cast

from click import Group, Command
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import Completer, Completion, CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from ghoshell_moss.moss.environment import Environment
from rich.console import Console
from rich.text import Text
from rich.rule import Rule
from typer import Typer

__all__ = ["TyperAppConsole", "TyperAppCompleter", "main"]


class TyperAppCompleter(Completer):
    """
    基于 Typer/Click 树的自动补全器。
    """

    def __init__(self, app: Typer, command_mark: str = "/", help_mark: str = "?") -> None:
        self.app: Typer = app
        self.command_mark: str = command_mark
        self.help_mark: str = help_mark

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        text: str = document.text_before_cursor

        # 识别前缀
        is_cmd: bool = text.startswith(self.command_mark)
        is_help: bool = text.startswith(self.help_mark)

        if not (is_cmd or is_help):
            return

        prefix: str = self.command_mark if is_cmd else self.help_mark

        # 特殊处理 exit 补全
        exit_cmd: str = f"{self.command_mark}exit"
        if is_cmd and exit_cmd.startswith(text):
            yield Completion(exit_cmd, start_position=-len(text), display_meta="exit console")

        # 提取命令路径
        parts: List[str] = text[len(prefix):].lstrip().split()
        if text.endswith(" ") and text.strip() != prefix:
            parts.append("")

        import typer.main
        try:
            # 获取根 Group
            current_click_obj: Any = typer.main.get_group(self.app)

            # 1. 递归查找到当前输入的父层级
            for i in range(len(parts) - 1):
                part: str = parts[i]
                if isinstance(current_click_obj, Group):
                    next_obj: Optional[Command] = current_click_obj.commands.get(part)
                    if next_obj:
                        current_click_obj = next_obj
                    else:
                        return
                else:
                    return

            last_part: str = parts[-1] if parts else ""

            # 2. 如果当前层级是 Group (有子命令)
            if isinstance(current_click_obj, Group):
                sub_commands: List[str] = list(current_click_obj.commands.keys())
                for cmd_name in sub_commands:
                    if cmd_name.startswith(last_part):
                        # 获取子命令对象以提取帮助文本
                        cmd_obj: Optional[Command] = current_click_obj.commands.get(cmd_name)
                        # short_help 通常是 Docstring 的第一行
                        help_text: str = (cmd_obj.short_help if cmd_obj else "") or ""

                        yield Completion(
                            cmd_name,
                            start_position=-len(last_part),
                            display_meta=help_text
                        )

            # 3. 如果当前层级是 Command (补全参数/选项)
            elif isinstance(current_click_obj, Command):
                for param in current_click_obj.params:
                    # 只补全以 -- 或 - 开头的选项
                    for opt in param.opts:
                        if opt.startswith(last_part):
                            yield Completion(
                                opt,
                                start_position=-len(last_part),
                                display_meta=param.help or "Option"
                            )
        except Exception:
            pass


class TyperAppConsole:
    COMMAND_MARK: str = "/"
    HELP_MARK: str = "?"
    EXIT_COMMAND: str = "/exit"

    def __init__(
            self,
            *,
            typer_module_name: str,
            typer_app_name: str = 'app',
            exit_command: Optional[str] = None,
            env: Environment | None = None,
    ) -> None:
        self.app_module: str = typer_module_name
        self.console: Console = Console()
        self.kb: KeyBindings = KeyBindings()
        self.env: Environment | None = env
        self._setup_bindings()
        self.exit_command: str = exit_command or self.EXIT_COMMAND

        self.app: Typer = self._load_app(typer_module_name, typer_app_name)
        self._completer: TyperAppCompleter = TyperAppCompleter(self.app, self.COMMAND_MARK, self.HELP_MARK)

        import typer.main
        click_group: Group = typer.main.get_group(self.app)
        self.display_name: str = click_group.name if click_group.name else "Typer-App"

    def _load_app(self, module_name: str, app_name: str) -> Typer:
        module: Any = importlib.import_module(module_name)
        app: Any = getattr(module, app_name)
        if not isinstance(app, Typer):
            raise ImportError(f"{module_name}:{app_name} is not a Typer instance")
        return app

    def _setup_bindings(self) -> None:
        @self.kb.add('escape')
        def _(event: Any) -> None:
            event.current_buffer.reset()

    def _get_bottom_toolbar(self) -> StyleAndTextTuples:
        """
        使用显式的元组定义样式，避免 HTML 解析错误。
        格式: (style_str, text_str)
        """
        return [
            ("class:toolbar.label", " App: "),
            ("class:toolbar.name", f" {self.display_name} "),
            ("", " | "),
            ("class:toolbar.key", " / "),
            ("", " Exec "),
            ("class:toolbar.key", " ? "),
            ("", " Help "),
            ("class:toolbar.key", f" {self.exit_command} "),
            ("", " Exit "),
        ]

    def run_command_sync(self, command_str: str, is_help: bool = False) -> None:
        """
        同步执行子进程。
        """
        actual_cmd_body: str = f"{command_str} --help" if is_help else command_str

        prefix_list: List[str] = [sys.executable, "-m", "typer", self.app_module, "run"]
        cmd_list: List[str] = prefix_list + actual_cmd_body.split()

        self.console.print("\n")
        title: str = f" [bold yellow]Help:[/] {self.display_name} {command_str}" if is_help \
            else f"🚀 [bold cyan]Exec:[/] {self.display_name} {command_str}"
        self.console.print(Rule(title=Text.from_markup(title), style="cyan"))

        try:
            subprocess.run(cmd_list, check=False, env=self.env.dump_moss_env(for_child_process=True))
        except KeyboardInterrupt:
            self.console.print(Text("\n[Aborted by User]", style="bold red"))
        finally:
            self.console.print(Rule(style="dim"))
            self.console.print("\n")

    async def _main_loop(self) -> None:
        # 使用自定义样式表来渲染 toolbar 和 prompt
        session: PromptSession = PromptSession(
            key_bindings=self.kb,
            bottom_toolbar=self._get_bottom_toolbar
        )

        while True:
            try:
                # Prompt 同样使用 Tuple 列表，保证 100% 正确渲染
                prompt_content: StyleAndTextTuples = [
                    ("class:prompt.name", self.display_name),
                    ("", " > "),
                ]

                user_input: str = await session.prompt_async(
                    prompt_content,
                    completer=self._completer
                )

                stripped_input: str = user_input.strip()
                if not stripped_input:
                    continue

                if stripped_input == self.exit_command:
                    break

                if stripped_input.startswith(self.HELP_MARK):
                    body: str = stripped_input[len(self.HELP_MARK):].strip()
                    self.run_command_sync(body, is_help=True)
                elif stripped_input.startswith(self.COMMAND_MARK):
                    body: str = stripped_input[len(self.COMMAND_MARK):].strip()
                    self.run_command_sync(body, is_help=False)
                else:
                    await self.handle_text_input(stripped_input)

            except (EOFError, KeyboardInterrupt):
                break

    async def handle_text_input(self, text: str) -> None:
        self.console.print(f"[bold white][Echo][/] {text}")

    def on_start(self) -> None:
        self.console.clear()
        self.console.print(Rule(title="[bold green] TYPER REPL CONSOLE [/]", style="green"))
        self.console.print(
            f"Welcome! Use [bold yellow]{self.COMMAND_MARK}[/] for commands and [bold yellow]{self.HELP_MARK}[/] for help.\n")

    def on_quit(self) -> None:
        self.console.print(Text("Bye!", style="bold magenta"))

    def run(self) -> None:
        self.on_start()
        try:
            asyncio.run(self._main_loop())
        finally:
            self.on_quit()


def main() -> None:
    # 这里的模块路径请根据实际情况修改
    console = TyperAppConsole(
        typer_module_name="ghoshell_moss.cli.main",
        typer_app_name="app",
        env=Environment.discover(),
    )
    console.run()


if __name__ == "__main__":
    main()
