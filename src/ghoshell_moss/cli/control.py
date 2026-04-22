import sys
import subprocess
import asyncio
import importlib
from typing import Iterable, Optional, List, Any

import typer.main
from click import Group, Command
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import Completer, Completion, CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from ghoshell_moss.host.environment import Environment
from rich.console import Console
from rich.text import Text
from rich.rule import Rule
from typer import Typer

__all__ = ["TyperAppController", "TyperAppCompleter", "main"]


class TyperAppCompleter(Completer):
    """
    基于 Typer/Click 树的自动补全器。
    默认状态下尝试补全命令，若以 help_mark 开头则尝试补全帮助路径。
    """

    def __init__(self, app: Typer, help_mark: str = "?") -> None:
        self.app: Typer = app
        self.help_mark: str = help_mark

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        text: str = document.text_before_cursor

        # 识别是否处于帮助模式
        is_help: bool = text.startswith(self.help_mark)

        # 提取用于解析的清理后的文本
        if is_help:
            # 去掉 ? 前缀并左去空格
            clean_text = text[len(self.help_mark):].lstrip()
            prefix = self.help_mark
        else:
            clean_text = text.lstrip()
            prefix = ""

        # 分割路径
        parts: List[str] = clean_text.split()
        if text.endswith(" ") and clean_text != "":
            parts.append("")

        # 处理退出命令的特殊补全
        if not is_help and "exit".startswith(clean_text):
            yield Completion("exit", start_position=-len(clean_text), display_meta="exit console")

        try:
            # 获取 Typer 对应的 Click 根 Group
            current_click_obj: Any = typer.main.get_group(self.app)

            # 1. 递归查找到当前输入路径的父层级
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

            # 2. 如果当前层级是 Group (展示子命令)
            if isinstance(current_click_obj, Group):
                sub_commands: List[str] = list(current_click_obj.commands.keys())
                for cmd_name in sub_commands:
                    if cmd_name.startswith(last_part):
                        cmd_obj: Optional[Command] = current_click_obj.commands.get(cmd_name)
                        help_text: str = (cmd_obj.short_help if cmd_obj else "") or ""
                        yield Completion(
                            cmd_name,
                            start_position=-len(last_part),
                            display_meta=help_text
                        )

            # 3. 如果当前层级是 Command (展示选项)
            elif isinstance(current_click_obj, Command):
                for param in current_click_obj.params:
                    for opt in param.opts:
                        if opt.startswith(last_part):
                            yield Completion(
                                opt,
                                start_position=-len(last_part),
                                display_meta=param.help or "Option"
                            )
        except Exception:
            pass


class TyperAppController:
    HELP_MARK: str = "?"
    EXIT_WORD: str = "exit"

    def __init__(
            self,
            *,
            typer_module_name: str,
            typer_app_name: str = 'app',
            env: Environment | None = None,
    ) -> None:
        self.app_module: str = typer_module_name
        self.console: Console = Console()
        self.kb: KeyBindings = KeyBindings()
        self.env: Environment | None = env
        self._setup_bindings()

        self.app: Typer = self._load_app(typer_module_name, typer_app_name)
        # 初始化不带 / 前缀限制的补全器
        self._completer: TyperAppCompleter = TyperAppCompleter(self.app, self.HELP_MARK)

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
        return [
            ("class:toolbar.label", " App: "),
            ("class:toolbar.name", f" {self.display_name} "),
            ("", " | "),
            ("class:toolbar.key", " [Enter] "),
            ("", " Exec "),
            ("class:toolbar.key", f" {self.HELP_MARK} "),
            ("", " Help "),
            ("class:toolbar.key", f" {self.EXIT_WORD} "),
            ("", " Exit "),
        ]

    def run_command_sync(self, command_str: str, is_help: bool = False) -> None:
        """
        同步执行子进程命令。
        """
        parts = command_str.split()
        if not is_help and parts:
            try:
                current_click_obj: Any = typer.main.get_group(self.app)
                for part in parts:
                    if isinstance(current_click_obj, Group):
                        next_obj = current_click_obj.commands.get(part)
                        if next_obj:
                            current_click_obj = next_obj
                        else:
                            break
                # 如果停在 Group 级且无后续，强制触发 --help
                if isinstance(current_click_obj, Group):
                    is_help = True
            except Exception:
                pass

        actual_cmd_body: str = f"{command_str} --help" if is_help else command_str
        prefix_list: List[str] = [sys.executable, "-m", "typer", self.app_module, "run"]
        cmd_list: List[str] = prefix_list + actual_cmd_body.split()

        self.console.print("\n")
        title: str = f" [bold yellow]Help:[/] {self.display_name} {command_str}" if is_help \
            else f"🚀 [bold cyan]Exec:[/] {self.display_name} {command_str}"
        self.console.print(Rule(title=Text.from_markup(title), style="cyan"))

        try:
            # 注入环境变量，确保子进程环境一致
            child_env = self.env.dump_moss_env(for_child_process=True) if self.env else None
            subprocess.run(cmd_list, check=False, env=child_env)
        except KeyboardInterrupt:
            self.console.print(Text("\n[Aborted by User]", style="bold red"))
        finally:
            self.console.print(Rule(style="dim"))
            self.console.print("\n")

    async def _main_loop(self) -> None:
        session: PromptSession = PromptSession(
            key_bindings=self.kb,
            bottom_toolbar=self._get_bottom_toolbar
        )

        while True:
            try:
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

                # 退出逻辑
                if stripped_input == self.EXIT_WORD:
                    break

                # 路由：判断是否为帮助请求
                if stripped_input.startswith(self.HELP_MARK):
                    body: str = stripped_input[len(self.HELP_MARK):].strip()
                    self.run_command_sync(body, is_help=True)
                else:
                    # 默认全部作为命令执行
                    self.run_command_sync(stripped_input, is_help=False)

            except (EOFError, KeyboardInterrupt):
                break

    def on_start(self) -> None:
        self.console.clear()
        self.console.print(Rule(title="[bold green] MOSS TYPER SHELL [/]", style="green"))
        self.console.print(
            f"Welcome! Direct input commands, or use [bold yellow]{self.HELP_MARK}[/] for help.\n")

    def on_quit(self) -> None:
        self.console.print(Text("Bye!", style="bold magenta"))

    def run(self) -> None:
        self.on_start()
        try:
            asyncio.run(self._main_loop())
        finally:
            self.on_quit()


def main() -> None:
    # 模块路径保持你原始的配置
    controller = TyperAppController(
        typer_module_name="ghoshell_moss.cli.main",
        typer_app_name="app",
        env=Environment.discover(),
    )
    controller.run()


if __name__ == "__main__":
    main()