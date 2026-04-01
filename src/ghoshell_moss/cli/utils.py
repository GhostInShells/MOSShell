"""
ghoshell_cli utility functions
"""

import click
from typing import Optional


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


def print_table(headers: list, rows: list):
    """打印简易表格"""
    # 计算列宽
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # 打印表头（黄色加粗）
    header_line = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    click.secho(header_line, fg="yellow", bold=True)

    # 打印分割线
    click.echo("-" * (sum(col_widths) + (len(headers) - 1) * 3))

    # 打印行
    for row in rows:
        row_line = " | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row))
        click.echo(row_line)


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
