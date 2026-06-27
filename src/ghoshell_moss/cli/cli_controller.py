"""
moss-cli — human-facing interactive shell with tab completion.

Each command executes in a subprocess (python -m typer ... run <cmd>)
for process isolation.  Global context (mode, ghost, network, scope)
is injected via environment variables.

Startup flow:
  1. Project.discover() — workspace + env
  2. Interactive config — mode / ghost / network / scope, each with explanation
  3. TyperAppController REPL — / cmd, ? help, Tab complete
"""

import asyncio
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Any

import click
import typer
from click import Group, Command
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion, CompleteEvent
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.project import Project, HostModeMeta

__all__ = ["TyperAppController", "TyperAppCompleter", "main"]


# ---------------------------------------------------------------------------
# tab completer — walks the Typer → Click command tree
# ---------------------------------------------------------------------------

class TyperAppCompleter(Completer):
    """Tab completion from a Typer app's Click command tree."""

    def __init__(self, app: typer.Typer, *, command_mark: str = "/", help_mark: str = "?") -> None:
        self.app = app
        self.help_mark = help_mark
        self.command_mark = command_mark

    def get_completions(self, document: Document, complete_event: CompleteEvent) -> Iterable[Completion]:
        text = document.text_before_cursor

        is_help = text.startswith(self.help_mark)
        is_cmd = text.startswith(self.command_mark)
        if not (is_help or is_cmd):
            return

        prefix_len = len(self.help_mark) if is_help else len(self.command_mark)
        clean_text = text[prefix_len:].lstrip()

        parts = clean_text.split()
        if text.endswith(" ") and clean_text:
            parts.append("")

        # exit shortcut
        if not is_help and "exit".startswith(clean_text):
            yield Completion("exit", start_position=-len(clean_text), display_meta="exit console")

        try:
            current = typer.main.get_group(self.app)
            for i in range(len(parts) - 1):
                if hasattr(current, 'commands'):
                    nxt = current.commands.get(parts[i])
                    if nxt is None:
                        return
                    current = nxt
                else:
                    return

            last_part = parts[-1] if parts else ""

            if hasattr(current, 'commands'):
                for cmd_name in sorted(current.commands):
                    if cmd_name.startswith(last_part):
                        cmd_obj = current.commands.get(cmd_name)
                        help_text = (cmd_obj.short_help if cmd_obj else "") or ""
                        yield Completion(
                            cmd_name, start_position=-len(last_part),
                            display_meta=help_text[:80],
                        )
            elif isinstance(current, Command):
                for param in current.params:
                    for opt in param.opts:
                        if opt.startswith(last_part):
                            yield Completion(
                                opt, start_position=-len(last_part),
                                display_meta=param.help or "option",
                            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# REPL controller — prompt_toolkit + subprocess execution
# ---------------------------------------------------------------------------

class TyperAppController:
    HELP_MARK = "?"
    CMD_MARK = "/"
    EXIT_WORD = "exit"

    def __init__(
        self,
        *,
        typer_module_name: str,
        typer_app_name: str = "app",
        env: Environment | None = None,
        console: Console | None = None,
    ):
        self.app_module = typer_module_name
        self.console = console or Console()
        self.kb = KeyBindings()
        self.env = env
        self._setup_bindings()

        self.app = self._load_app(typer_module_name, typer_app_name)
        self._completer = TyperAppCompleter(self.app, command_mark=self.CMD_MARK, help_mark=self.HELP_MARK)

        click_group = typer.main.get_group(self.app)
        self.display_name = click_group.name if click_group.name else "moss"

    def _load_app(self, module_name: str, app_name: str) -> typer.Typer:
        module = importlib.import_module(module_name)
        app = getattr(module, app_name)
        if not isinstance(app, typer.Typer):
            raise ImportError(f"{module_name}:{app_name} is not a Typer instance")
        return app

    def _setup_bindings(self) -> None:
        @self.kb.add("escape")
        def _(event: Any) -> None:
            event.current_buffer.reset()

    def _get_bottom_toolbar(self) -> StyleAndTextTuples:
        parts: StyleAndTextTuples = [
            ("class:toolbar.label", " " + self.display_name + " "),
        ]
        if self.env:
            ctx_parts = []
            if self.env.mode_name and self.env.mode_name != "none":
                ctx_parts.append(f"mode={self.env.mode_name}")
            if self.env.ghost_name and self.env.ghost_name != "none":
                ctx_parts.append(f"ghost={self.env.ghost_name}")
            if self.env.network and self.env.network != "default":
                ctx_parts.append(f"net={self.env.network}")
            scope = self.env.network_scope
            if scope and scope != "default":
                ctx_parts.append(f"scope={scope}")
            if ctx_parts:
                parts.append(("", " | "))
                parts.append(("class:toolbar.context", " ".join(ctx_parts)))
        parts.extend([
            ("", " | "),
            ("class:toolbar.key", " [Tab] "),
            ("", "complete "),
            ("class:toolbar.key", f" {self.HELP_MARK} "),
            ("", "help "),
            ("class:toolbar.key", f" {self.EXIT_WORD} "),
            ("", "quit"),
        ])
        return parts

    # -- execution --------------------------------------------------

    def run_command_sync(self, command_str: str, is_help: bool = False) -> None:
        """Execute a single command in a subprocess with MOSS env injected."""
        parts = command_str.split()
        if not is_help and parts:
            try:
                current = typer.main.get_group(self.app)
                for part in parts:
                    if hasattr(current, 'commands'):
                        nxt = current.commands.get(part)
                        if nxt:
                            current = nxt
                        else:
                            break
                if hasattr(current, 'commands'):
                    is_help = True  # partial path → show help
            except Exception:
                pass

        actual_body = f"{command_str} --help" if is_help else command_str
        cmd_list = [sys.executable, "-m", "typer", self.app_module, "run"] + actual_body.split()

        self.console.print("")
        title = (
            f" [bold yellow]Help:[/] {self.display_name} {command_str}" if is_help
            else f" [bold cyan]Exec:[/] {self.display_name} {command_str}"
        )
        self.console.print(Rule(title=Text.from_markup(title), style="cyan"))

        try:
            child_env = None
            if self.env:
                child_env = self.env.dump_cell_env(with_os_env=True)
            subprocess.run(cmd_list, check=False, env=child_env)
        except KeyboardInterrupt:
            self.console.print(Text("\n[Aborted by User]", style="bold red"))
        finally:
            self.console.print(Rule(style="dim"))
            self.console.print("")

    # -- main loop --------------------------------------------------

    async def _main_loop(self) -> None:
        session = PromptSession(key_bindings=self.kb, bottom_toolbar=self._get_bottom_toolbar)

        while True:
            try:
                prompt_content: StyleAndTextTuples = [
                    ("class:prompt.name", self.display_name),
                    ("", " > "),
                ]
                user_input = await session.prompt_async(prompt_content, completer=self._completer)
                stripped = user_input.strip()
                if not stripped:
                    continue

                if stripped.startswith(self.HELP_MARK):
                    body = stripped[len(self.HELP_MARK):].strip()
                    self.run_command_sync(body, is_help=True)
                elif stripped.startswith(self.CMD_MARK):
                    cmd = stripped[len(self.CMD_MARK):].strip()
                    if cmd == self.EXIT_WORD:
                        break
                    self.run_command_sync(cmd, is_help=False)
                else:
                    self.console.print(
                        f"Type [bold]{self.CMD_MARK}[/] before a command, "
                        f"or [bold]{self.HELP_MARK}[/] for help."
                    )
            except (EOFError, KeyboardInterrupt):
                break

    # -- lifecycle --------------------------------------------------

    def on_start(self) -> None:
        self.console.clear()
        self.console.print(Rule(title="[bold green] MOSS Shell [/]", style="green"))

        if self.env:
            rows = [
                ["mode", self.env.mode_name or "none"],
                ["ghost", self.env.ghost_name or "none"],
                ["network", f"{self.env.network} / {self.env.network_scope}"],
                ["workspace", str(self.env.workspace_path)],
            ]
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="dim")
            table.add_column(style="cyan")
            for label, value in rows:
                table.add_row(label, value)
            self.console.print(table)
        self.console.print(
            f"\nType [bold yellow]{self.CMD_MARK}[/] before a command "
            f"([bold yellow]{self.CMD_MARK}help[/] for reference)."
        )

    def on_quit(self) -> None:
        self.console.print(Text("Bye!", style="bold magenta"))

    def run(self) -> None:
        self.on_start()
        try:
            asyncio.run(self._main_loop())
        finally:
            self.on_quit()


# ---------------------------------------------------------------------------
# interactive config — walk through each context parameter with rationale
# ---------------------------------------------------------------------------

def _select_mode(project: Project) -> str:
    """Interactive mode selection.

    Options:
      0 — none (no mode isolation)
      1..N — discovered modes
      Enter (default) — keep current env value
    """
    console = Console()
    current = project.env.mode_name or "none"
    modes = list(project.list_modes())

    console.print(Rule(title="[bold yellow]Step 1: Select Mode[/]", style="yellow"))
    console.print(
        "[dim]Mode 隔离能力视图[/] — 不同 mode 提供不同的 providers / channels / configs.\n"
        "[dim]切换 mode 会改变模型看到的可用能力集合.[/]"
    )

    # build index: 0=none, 1..N=modes
    idx_to_name: dict[int, str] = {}
    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")

    none_label = "none [current]" if current == "none" else "none"
    table.add_row("0", none_label, "No mode — use project defaults only")
    idx_to_name[0] = "none"

    default_idx = 0
    for i, (_, manifest) in enumerate(modes, 1):
        if manifest.is_error():
            table.add_row(str(i), manifest.name(), f"[red]{manifest.error()}[/]")
            continue
        meta = manifest.value()
        label = f"{meta.name} [current]" if meta.name == current else meta.name
        table.add_row(str(i), label, (meta.description or "—")[:100])
        idx_to_name[i] = meta.name
        if meta.name == current:
            default_idx = i
    console.print(table)
    console.print()

    choice = click.prompt(
        f"Select mode — currently {current}  (Enter=confirm, 0=none)",
        default=str(default_idx),
        show_default=False,
    ).strip()
    idx = int(choice) if choice.isdigit() else default_idx
    return idx_to_name.get(idx, "none")


def _select_ghost(project: Project) -> str:
    """Interactive ghost selection.

    Options:
      0 — none (no ghost)
      1..N — discovered ghosts
      Enter (default) — keep current env value
    """
    console = Console()
    current = project.env.ghost_name or "none"
    ghosts = list(project.ghosts())

    console.print(Rule(title="[bold yellow]Step 2: Select Ghost[/]", style="yellow"))
    console.print(
        "[dim]Ghost 是持久化智能体运行时[/] — 不同 ghost 有不同的 soul / memory / 行为模式.\n"
        "[dim]选 none 则当前会话无 ghost 参与.[/]"
    )

    idx_to_name: dict[int, str] = {}
    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="cyan")
    table.add_column("Description", style="green")

    none_label = "none [current]" if current == "none" else "none"
    table.add_row("0", none_label, "No ghost — shell-only session")
    idx_to_name[0] = "none"

    default_idx = 0
    ghost_list = list(ghosts)
    for i, (_, meta) in enumerate(ghost_list, 1):
        if isinstance(meta, Exception):
            table.add_row(str(i), "?", f"[red]{meta}[/]")
            continue
        label = f"{meta.name()} [current]" if meta.name() == current else meta.name()
        table.add_row(str(i), label, (meta.description() or "—")[:100])
        idx_to_name[i] = meta.name()
        if meta.name() == current:
            default_idx = i
    console.print(table)
    console.print()

    choice = click.prompt(
        f"Select ghost — currently {current}  (Enter=confirm, 0=none)",
        default=str(default_idx),
        show_default=False,
    ).strip()
    idx = int(choice) if choice.isdigit() else default_idx
    return idx_to_name.get(idx, "none")


def _select_network(project: Project, default_network: str) -> str:
    """Interactive network selection with explanation."""
    console = Console()
    metas = project.network_metas()
    if not metas:
        console.print("[dim]No network configs found. Using 'default'.[/dim]")
        return default_network

    console.print(Rule(title="[bold yellow]Step 3: Select Network[/]", style="yellow"))
    console.print(
        "[dim]Network 决定通讯驱动和传输参数[/] — zenoh / mqtt / ws 等.\n"
        "[dim]不同 network 对应不同的连接方式和发现范围.[/]"
    )

    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="cyan")
    table.add_column("Driver", style="green")
    table.add_column("Scope", style="dim")
    table.add_column("Description", style="green")
    for i, (name, meta) in enumerate(sorted(metas.items()), 1):
        table.add_row(str(i), name, meta.driver, meta.scope, (meta.description or "—")[:80])
    console.print(table)
    console.print()

    names = sorted(metas.keys())
    default_idx = None
    for i, name in enumerate(names, 1):
        if name == default_network:
            default_idx = i
            break

    default_str = str(default_idx) if default_idx else "1"
    choice = click.prompt(
        f"Select network — currently {default_network}  (Enter=confirm)",
        default=default_str,
        show_default=False,
    ).strip()
    if choice and choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(names):
            return names[idx]
    return default_network


def _select_scope(default_scope: str) -> str:
    """Interactive scope input with explanation."""
    console = Console()
    console.print(Rule(title="[bold yellow]Step 4: Network Scope[/]", style="yellow"))
    console.print(
        "[dim]Scope 是通讯子空间标识[/] — 同一 scope 下的 cell 可以互相发现.\n"
        "[dim]多组并行 session 用不同 scope 隔离. 一般情况保留默认值即可.[/]"
    )
    console.print()
    scope = click.prompt(
        f"Enter scope — currently {default_scope}  (Enter=confirm)",
        default=default_scope,
        show_default=False,
    ).strip()
    return scope or default_scope


def interactive_config(project: Project) -> None:
    """Walk through mode / ghost / network / scope, explaining each choice.

    Modifies the project's Environment in-place so that downstream
    subprocesses inherit the selected context via env vars.
    """
    env = project.env
    console = Console()

    console.print(Rule(title="[bold green] MOSS Session Config [/]", style="green"))
    console.print(
        "[dim]Each choice shapes what the running MOSS instance can see and do.\n"
        "Press Enter to accept the default (from MOSS.md or environment).[/]"
    )

    # mode
    mode_name = _select_mode(project)
    if mode_name and mode_name != "none":
        env.set_mode(mode_name)
        console.print(f"[green]Mode set to:[/] [cyan]{mode_name}[/]")

    # ghost
    ghost_name = _select_ghost(project)
    if ghost_name and ghost_name != "none":
        env.set_ghost_name(ghost_name)
        console.print(f"[green]Ghost set to:[/] [cyan]{ghost_name}[/]")

    # network
    default_network = env.network or "default"
    network = _select_network(project, default_network)
    if network and network != "default":
        import os
        os.environ["MOSS_NETWORK"] = network
        console.print(f"[green]Network set to:[/] [cyan]{network}[/]")

    # scope
    default_scope = env.network_scope or "default"
    scope = _select_scope(default_scope)
    if scope and scope != "default":
        env.set_network_scope(scope)
        console.print(f"[green]Scope set to:[/] [cyan]{scope}[/]")

    console.print(Rule(style="dim"))


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

@click.command()
@click.option("--mode", "-m", default=None, help="MOSS mode name (skips interactive selection).")
@click.option("--ghost", "-g", default=None, help="Ghost name (skips interactive selection).")
@click.option("--network", default=None, help="Network driver (skips interactive selection).")
@click.option("--scope", "-s", default=None, help="Network scope (skips interactive selection).")
@click.option("--no-interactive", "-I", is_flag=True, help="Skip interactive config entirely.")
def main_entry(mode: str | None, ghost: str | None, network: str | None, scope: str | None, no_interactive: bool):
    """MOSS interactive shell — human-facing REPL with tab completion."""

    # 1. discover project
    project = Project.discover()
    env = project.env

    # 2. apply CLI overrides immediately
    if mode:
        env.set_mode(mode)
    if ghost:
        env.set_ghost_name(ghost)
    if network:
        import os
        os.environ["MOSS_NETWORK"] = network
    if scope:
        env.set_network_scope(scope)

    # 3. interactive config for any remaining unset values
    if not no_interactive:
        interactive_config(project)

    # 4. launch REPL
    controller = TyperAppController(
        typer_module_name="ghoshell_moss.cli.main",
        typer_app_name="app",
        env=env,
    )
    controller.run()


def main():
    """Entry point for moss-cli console script."""
    main_entry()


if __name__ == "__main__":
    main()
