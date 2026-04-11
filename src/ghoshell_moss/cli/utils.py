"""
ghoshell_cli utility functions
"""

import click
from typing import Optional
from rich.console import Console, Group
from rich.text import Text

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
]

console = Console()


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
    # 使用 secho 打印绿色的勾号和消息
    click.secho(f"✓ {message}", fg="green", bold=True)


def print_error(message: str):
    """打印错误消息 - 红色"""
    click.secho(f"✗ {message}", fg="red", bold=True)


def print_warning(message: str):
    """打印警告消息 - 黄色"""
    click.secho(f"⚠ {message}", fg="yellow", bold=True)


def print_info(message: str):
    """打印提示消息 - 蓝色"""
    click.secho(f"ℹ {message}", fg="blue")


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
    if title:
        # 标题用青色加粗
        click.secho(f"┏━ {title} ━┓", fg="cyan", bold=True)

    # 内容稍稍缩进
    for line in content.splitlines():
        click.echo(f"  {line}")

    if title:
        click.secho(f"┗━" + "━" * (len(title) + 2) + "━┛", fg="cyan", bold=True)
    else:
        click.secho("━" * 20, fg="cyan")
