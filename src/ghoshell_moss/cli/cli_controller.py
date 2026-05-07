import sys
import subprocess
import asyncio
import importlib
import click
from prompt_toolkit.shortcuts import prompt
from prompt_toolkit.styles import Style
from typing import Iterable, Optional, List, Any
from prompt_toolkit.completion import WordCompleter
from pygments.lexer import default
from rich.table import Table

import typer.main
from click import Group, Command
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.completion import Completer, Completion, CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from ghoshell_moss.host import Host, Environment
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

    def __init__(self, app: Typer, *, command_mark: str = '/', help_mark: str = "?") -> None:
        self.app: Typer = app
        self.help_mark: str = help_mark
        self.command_mark: str = command_mark

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        text: str = document.text_before_cursor

        # 识别是否处于帮助模式
        is_help: bool = text.startswith(self.help_mark)
        is_cmd: bool = text.startswith(self.command_mark)

        # 提取用于解析的清理后的文本
        if is_help:
            # 去掉 ? 前缀并左去空格
            clean_text = text[len(self.help_mark):].lstrip()
            prefix = self.help_mark
        elif is_cmd:
            clean_text = text[len(self.command_mark):].lstrip()
            prefix = ''
        else:
            return

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
    CMD_MARK: str = "/"
    EXIT_WORD: str = "exit"

    def __init__(
            self,
            *,
            typer_module_name: str,
            typer_app_name: str = 'app',
            env: Environment | None = None,
            console: Console | None = None,
    ) -> None:
        self.app_module: str = typer_module_name
        self.console: Console = console or Console()
        self.kb: KeyBindings = KeyBindings()
        self.env: Environment | None = env
        self._setup_bindings()

        self.app: Typer = self._load_app(typer_module_name, typer_app_name)
        # 初始化不带 / 前缀限制的补全器
        self._completer: TyperAppCompleter = TyperAppCompleter(
            self.app,
            command_mark=self.CMD_MARK,
            help_mark=self.HELP_MARK,
        )

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

                # 路由：判断是否为帮助请求
                if stripped_input.startswith(self.HELP_MARK):
                    body: str = stripped_input[len(self.HELP_MARK):].strip()
                    self.run_command_sync(body, is_help=True)
                elif stripped_input.startswith(self.CMD_MARK):
                    # 默认全部作为命令执行
                    stripped_cmd = stripped_input[len(self.CMD_MARK):]
                    # 退出逻辑
                    if stripped_cmd == self.EXIT_WORD:
                        break
                    self.run_command_sync(stripped_cmd, is_help=False)
                else:
                    self.console.print("press `/` or `?` to issue command")

            except (EOFError, KeyboardInterrupt):
                break

    def on_start(self) -> None:
        self.console.clear()
        self.console.print(Rule(title="[bold green] MOSS TYPER SHELL [/]", style="green"))

        # 新增环境状态看板
        if self.env:
            status_table = f"[dim]Mode:[/] [cyan]{self.env.moss_mode_name}[/] | [dim]Scope:[/] [cyan]{self.env.session_scope}[/]"
            self.console.print(status_table, justify="center")

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


def interactive_config(host: Host, console: Console, current_mode: str | None, current_scope: str | None):
    style = Style.from_dict({
        'question': '#ansiyellow bold',
    })

    # 1. 自动发现所有模式
    all_modes = host.all_modes()  # dict[str, MossMode]

    if not current_mode:
        console.print(Rule(title="Step 1: Select Environment Mode", style="yellow"))

        # 使用 Rich Table 展示所有可用模式的信息
        table = Table(show_header=True, header_style="bold magenta", box=None)
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Description", style="green")
        table.add_column("Source", style="dim", overflow="ellipsis")

        for name, mode_obj in all_modes.items():
            # 简化的文件路径，只显示最后两级以便阅读
            short_file = "/".join(mode_obj.file.split("/")[-2:]) if mode_obj.file else "internal"
            table.add_row(name, mode_obj.description, short_file)

        console.print(table)

        # 准备自动补全器
        mode_completer = WordCompleter(list(all_modes.keys()), ignore_case=True)

        # 交互输入
        default_mode = host.env.moss_mode_name
        current_mode = prompt([
            ('class:question', "❯ Select Mode "),
            ('', f"(default: {default_mode}): ")
        ], style=style, completer=mode_completer).strip() or default_mode

    # 2. 配置 Session Scope
    if not current_scope:
        console.print(Rule(title="Step 2: Define Session Scope", style="yellow"))
        console.print(f"[dim]Scope determines the isolation boundary for your session data.[/]")

        default_scope = host.env.session_scope
        current_scope = prompt([
            ('class:question', "❯ Enter Session Scope "),
            ('', f"(default: {default_scope}): ")
        ], style=style).strip() or default_scope

    return current_mode, current_scope


@click.command()
@click.option('--mode', '-m', help='运行模式 (如: dev, prod)', default=None)
@click.option('--session-scope', '-s', help='Session 作用域', default=None)
@click.option('--interactive', '-i', is_flag=True, help='强制进入交互式配置', default=True)
def main_entry(mode, session_scope, interactive):
    host = Host.discover()
    env = host.env
    console = Console()

    # 2. 交互式补全逻辑
    # 如果参数缺失，或者用户明确要求交互配置
    if interactive or (mode is None and session_scope is None):
        mode, session_scope = interactive_config(host, console, mode, session_scope)

    # 3. 应用配置
    if mode:
        env.set_mode(mode)
    if session_scope:
        env.set_session_scope(session_scope)
    host = host.reboot()

    # 4. 启动控制器
    controller = TyperAppController(
        typer_module_name="ghoshell_moss.cli.main",
        typer_app_name="app",
        env=host.env,
        console=console,
    )
    controller.run()


def main():
    """
    MOSS 命令行统一入口
    """

    # 1. 使用 Typer 或 Click 快速解析启动参数
    # 这里为了简单直接用 click 的装饰器风格，或者手动解析 sys.argv
    main_entry()


if __name__ == "__main__":
    main()
