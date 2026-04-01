"""
ghoshell_cli utility functions
"""

import click
from typing import Optional, Any

try:
    from rich import print as rprint
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich.syntax import Syntax

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def get_console() -> Optional[Any]:
    """Get rich console instance, returns None if rich is not available"""
    if RICH_AVAILABLE:
        return Console()
    return None


def print_success(message: str):
    """Print success message"""
    if RICH_AVAILABLE:
        console = Console()
        console.print(f"[green]✓[/green] {message}")
    else:
        click.echo(f"✓ {message}")


def print_error(message: str):
    """Print error message"""
    if RICH_AVAILABLE:
        console = Console()
        console.print(f"[red]✗[/red] {message}")
    else:
        click.echo(f"✗ {message}")


def print_warning(message: str):
    """Print warning message"""
    if RICH_AVAILABLE:
        console = Console()
        console.print(f"[yellow]⚠[/yellow] {message}")
    else:
        click.echo(f"⚠ {message}")


def print_info(message: str):
    """Print info message"""
    if RICH_AVAILABLE:
        console = Console()
        console.print(f"[blue]ℹ[/blue] {message}")
    else:
        click.echo(f"ℹ {message}")


def print_code(code: str, language: str = "python"):
    """Print code block with syntax highlighting"""
    if RICH_AVAILABLE:
        console = Console()
        syntax = Syntax(code, language, theme="monokai", line_numbers=True)
        console.print(syntax)
    else:
        click.echo(code)


def print_table(headers: list, rows: list):
    """Print table"""
    if RICH_AVAILABLE:
        console = Console()
        table = Table(*headers)
        for row in rows:
            table.add_row(*[str(cell) for cell in row])
        console.print(table)
    else:
        # Simple table output
        col_widths = [len(str(h)) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Print header
        header_line = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
        click.echo(header_line)
        click.echo("-" * len(header_line))

        # Print rows
        for row in rows:
            row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
            click.echo(row_line)


def print_panel(content: str, title: Optional[str] = None):
    """Print content in a panel"""
    if RICH_AVAILABLE:
        console = Console()
        panel = Panel(content, title=title, border_style="blue")
        console.print(panel)
    else:
        if title:
            click.echo(f"=== {title} ===")
        click.echo(content)
        if title:
            click.echo("=" * (len(title) + 8))
