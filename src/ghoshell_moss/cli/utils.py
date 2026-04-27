"""
ghoshell_cli utility functions
"""

import click
from typing import Optional, List, Any, Union
from rich.console import Console, Group
from rich.text import Text
from rich.panel import Panel
from rich.box import ROUNDED, DOUBLE, HEAVY, SIMPLE
from rich.table import Table
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
]

console = Console(force_terminal=True, color_system="auto")


# 在你现有的代码逻辑里，可以考虑这样写样式
def print_host_mode_info(host: Host) -> None:
    # 使用 Rich 的渲染
    console.print(f"[bold cyan]MODE:[/bold cyan] [green]{host.mode.name}[/green]")

    # 路径类信息，由于很长，用 dim 弱化
    style = "dim italic"
    console.print(f"[{style}]workspace: {host.env.workspace_path}[/{style}]")
    if host.mode.import_path:
        console.print(f"[{style}]mode package: {host.mode.import_path}[/{style}]")
    if host.mode.file:
        console.print(f"[{style}]mode file: {host.mode.file}[/{style}]")

    # 分隔线也可以用 dim
    console.print("[dim]" + "—" * 40 + "[/dim]")


def echo(message: str):
    """方便未来统一替换."""
    click.echo(message)


def print_success(message: str):
    """打印成功消息 - 绿色"""
    console.print(f"[bold green]✓ {message}[/bold green]")


def print_error(message: str):
    """打印错误消息 - 红色"""
    console.print(f"[bold red]✗ {message}[/bold red]")


def print_warning(message: str):
    """打印警告消息 - 黄色"""
    console.print(f"[bold yellow]⚠ {message}[/bold yellow]")


def print_info(message: str):
    """打印提示消息 - 亮蓝色，图标加粗"""
    console.print(f"[bold bright_blue]ℹ[/bold bright_blue] [bright_blue]{message}[/bright_blue]")


def print_code(code: str, language: str = "python"):
    """
    打印代码块。
    由于去掉了 rich，无法实现复杂的语法高亮，
    这里通过加深背景颜色或改变前景色来区分代码区域。
    """
    click.secho(f"# --- {language} code ---", fg="cyan", dim=True)
    click.echo(code)
    click.secho("# -----------------------", fg="cyan", dim=True)


def print_panel(content: str, title: Optional[str] = None):
    """打印面板效果"""
    # 使用 Text.from_markup 解析内容中的富文本标记
    renderable = Text.from_markup(content)
    # 标题样式：加粗亮青色
    title_renderable = None
    if title:
        title_renderable = Text(title, style="bold bright_cyan")
    # 更美观的样式：双线边框，标题居中，边框亮青色
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
    """
    打印简洁风格面板，使用简单的白线边框
    """
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


def print_simple_table(
    data: List[List[Any]],
    headers: List[str],
    title: Optional[str] = None,
    header_style: str = "bold",
    column_styles: Optional[List[str]] = None,
    title_style: str = "bold underline",
) -> None:
    """
    打印简洁风格表格，使用简单的白线边框
    """
    # 应用标题样式
    styled_title = f"[{title_style}]{title}[/{title_style}]" if title else None

    table = Table(
        title=styled_title,
        box=SIMPLE,
        border_style="white",
        header_style=header_style,
        show_header=True,
        show_edge=True,
        pad_edge=True,
        padding=(0, 1),
    )

    # 添加列
    for i, header in enumerate(headers):
        style = column_styles[i] if column_styles and i < len(column_styles) else None
        table.add_column(header, style=style)

    # 添加行
    for row in data:
        table.add_row(*[str(cell) for cell in row])

    console.print(table)
