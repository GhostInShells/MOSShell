"""
ghoshell_cli utility functions
"""

import click
import json as _json
from contextlib import contextmanager
from typing import Optional, List, Any, Union
from rich.console import Console as RichConsole, Group
from rich.text import Text
from rich.panel import Panel
from rich.syntax import Syntax as RichSyntax
from rich.table import Table
from rich.box import ROUNDED, DOUBLE, HEAVY, SIMPLE
from rich.style import Style

from ghoshell_moss.host import Host

__all__ = [
    'console',
    'print_host_mode_info',
    'echo',
    'print_success',
    'print_error',
    'print_warning',
    'print_info',
    'print_code',
    'print_panel',
    'print_simple_table',
    'print_simple_panel',
    'set_ai_mode',
    'is_ai_mode',
    'show_status',
]

_ai_mode = False

# real console for human users
_real_console = RichConsole(force_terminal=True, color_system="auto")
_pale_console = RichConsole(no_color=True, force_terminal=False, color_system=None)


def _strip_markup(text: str) -> str:
    """Strip rich markup tags from a string, e.g. [bold cyan]text[/bold cyan] -> text."""
    try:
        return Text.from_markup(text).plain
    except Exception:
        return text


def _renderable_to_plain(renderable) -> str:
    """Convert a rich renderable (Syntax, Panel, Table) to plain text."""
    try:
        with _pale_console.capture() as capture:
            _pale_console.print(renderable)
        return capture.get()
    except Exception:
        return str(renderable)


def _ai_print(*args, **kwargs) -> None:
    """AI mode: strip markup and print plain text."""
    parts = []
    for arg in args:
        if isinstance(arg, str):
            parts.append(_strip_markup(arg))
        elif isinstance(arg, RichSyntax):
            parts.append(arg.code.rstrip())
        elif isinstance(arg, (Panel, Table)):
            parts.append(_renderable_to_plain(arg))
        else:
            parts.append(str(arg))

    text = "".join(parts).rstrip()
    if text:
        click.echo(text)


def _ai_print_exception(**kwargs) -> None:
    import traceback
    traceback.print_exc()


def _ai_json(**kwargs) -> None:
    data = kwargs.pop('data', None)
    if data is not None:
        click.echo(_json.dumps(data, ensure_ascii=False, indent=2))


class _ConsoleProxy:
    """
    Proxy object for console that delegates to either RichConsole (human mode)
    or plain text output (AI mode).  Using a proxy instead of replacing the
    module-level variable means `from utils import console` always works
    regardless of when AI mode is toggled.
    """

    # --- methods called directly on console ---
    def print(self, *args, **kwargs):
        if _ai_mode:
            return _ai_print(*args, **kwargs)
        return _real_console.print(*args, **kwargs)

    def json(self, **kwargs):
        if _ai_mode:
            return _ai_json(**kwargs)
        return _real_console.json(**kwargs)

    def print_exception(self, **kwargs):
        if _ai_mode:
            return _ai_print_exception(**kwargs)
        return _real_console.print_exception(**kwargs)

    # --- everything else delegates to the real console ---
    def __getattr__(self, name):
        return getattr(_real_console, name)

    def __rich_console__(self, *args, **kwargs):
        return _real_console.__rich_console__(*args, **kwargs)


# single console instance — never replaced, always delegates
console = _ConsoleProxy()

# convenience: expose internal consoles for edge cases
_rich = _real_console


def set_ai_mode(enabled: bool) -> None:
    """Enable AI-friendly plain output mode, stripping all rich visual formatting."""
    global _ai_mode
    _ai_mode = enabled


def is_ai_mode() -> bool:
    """Check if AI output mode is active."""
    return _ai_mode


@contextmanager
def show_status(message: str):
    """
    Show a rich spinner status while a long-running operation is in progress.
    In AI mode (--ai), this is a transparent no-op that yields immediately.
    """
    if _ai_mode:
        yield
    else:
        with _real_console.status(f"[dim]{message}[/dim]") as status:
            yield status


def print_host_mode_info(host: Host) -> None:
    if _ai_mode:
        click.echo(f"MODE: {host.mode.name}")
        click.echo(f"workspace: {host.env.workspace_path}")
        if host.mode.import_path:
            click.echo(f"mode package: {host.mode.import_path}")
        if host.mode.file:
            click.echo(f"mode file: {host.mode.file}")
        click.echo("—" * 40)
        return

    console.print(f"[bold cyan]MODE:[/bold cyan] [green]{host.mode.name}[/green]")
    style = "dim italic"
    console.print(f"[{style}]workspace: {host.env.workspace_path}[/{style}]")
    if host.mode.import_path:
        console.print(f"[{style}]mode package: {host.mode.import_path}[/{style}]")
    if host.mode.file:
        console.print(f"[{style}]mode file: {host.mode.file}[/{style}]")
    console.print("[dim]" + "—" * 40 + "[/dim]")


def echo(message: str):
    click.echo(message)


def print_success(message: str):
    if _ai_mode:
        click.echo(f"[OK] {_strip_markup(message)}")
        return
    console.print(f"[bold green]✓ {message}[/bold green]")


def print_error(message: str):
    if _ai_mode:
        click.echo(f"[ERROR] {_strip_markup(message)}")
        return
    console.print(f"[bold red]✗ {message}[/bold red]")


def print_warning(message: str):
    if _ai_mode:
        click.echo(f"[WARN] {_strip_markup(message)}")
        return
    console.print(f"[bold yellow]⚠ {message}[/bold yellow]")


def print_info(message: str):
    if _ai_mode:
        click.echo(f"[INFO] {_strip_markup(message)}")
        return
    console.print(f"[bold bright_blue]ℹ[/bold bright_blue] [bright_blue]{message}[/bright_blue]")


def print_code(code: str, language: str = "python"):
    if _ai_mode:
        click.echo(code)
        return
    click.secho(f"# --- {language} code ---", fg="cyan", dim=True)
    click.echo(code)
    click.secho("# -----------------------", fg="cyan", dim=True)


def print_panel(content: str, title: Optional[str] = None):
    if _ai_mode:
        _print_ai_panel(content, title)
        return
    renderable = Text.from_markup(content)
    title_renderable = None
    if title:
        title_renderable = Text(title, style="bold bright_cyan")
    panel = Panel(
        renderable,
        title=title_renderable,
        box=DOUBLE,
        border_style="bright_cyan",
        title_align="center",
        padding=(1, 2),
    )
    console.print(panel)


def print_simple_panel(content: Union[str, Text], title: Optional[str] = None) -> None:
    if _ai_mode:
        if isinstance(content, Text):
            content = content.plain
        else:
            content = _strip_markup(content)
        _print_ai_panel(content, title)
        return

    if isinstance(content, str):
        renderable = Text.from_markup(content)
    else:
        renderable = content

    panel = Panel(
        renderable,
        title=title,
        box=SIMPLE,
        border_style="white",
        padding=(0, 1),
    )
    console.print(panel)


def _print_ai_panel(content: str, title: Optional[str] = None) -> None:
    """AI mode: plain text representation of a panel."""
    content = _strip_markup(content)
    if title:
        click.echo(f"## {title}")
        click.echo(content)
    else:
        click.echo(content)


def print_simple_table(
        data: List[List[Any]],
        headers: List[str],
        title: Optional[str] = None,
        header_style: str = "bold cyan",
        column_styles: Optional[List[str]] = None,
        title_style: str = "bold underline",
        column_ratios: Optional[List[int]] = None,
        show_header: bool = True,
) -> None:
    """
    Print a table. In AI mode, outputs as a markdown table.
    """
    if _ai_mode:
        _print_ai_table(data, headers, title, show_header)
        return

    styled_title = f"[{title_style}]{title}[/{title_style}]" if title else None

    table = Table(
        title=styled_title,
        box=SIMPLE,
        border_style="dim",
        header_style=header_style,
        show_header=show_header,
        show_edge=False,
        pad_edge=False,
        padding=(0, 2),
        collapse_padding=True,
    )

    for i, header in enumerate(headers):
        style = column_styles[i] if column_styles and i < len(column_styles) else None
        ratio = column_ratios[i] if column_ratios and i < len(column_ratios) else None
        table.add_column(header, style=style, ratio=ratio)

    for row in data:
        table.add_row(*[str(cell) for cell in row])

    console.print(table)


def _print_ai_table(
        data: List[List[Any]],
        headers: List[str],
        title: Optional[str] = None,
        show_header: bool = True,
) -> None:
    """AI mode: output table as markdown."""
    if title:
        click.echo(f"## {title}")
        click.echo("")

    # strip markup from all cells
    clean_data = []
    for row in data:
        clean_data.append([_strip_markup(str(cell)) for cell in row])
    clean_headers = [_strip_markup(h) for h in headers]

    if show_header:
        # markdown header row
        header_line = "| " + " | ".join(clean_headers) + " |"
        separator = "|" + "|".join([" --- " for _ in clean_headers]) + "|"
        click.echo(header_line)
        click.echo(separator)

    for row in clean_data:
        row_line = "| " + " | ".join(row) + " |"
        click.echo(row_line)
