import typer
import sys
from typing import Optional
from ghoshell_moss.cli.utils import (
    print_error, print_info,
    print_panel, echo
)
from ghoshell_moss.cli import codex
from ghoshell_moss.cli import concepts
from ghoshell_moss.cli import workspace

__version__ = "0.1.0-beta"

# 创建 app 对象
# help_option_names 依然有效
app = typer.Typer(
    name="moss",
    help="MOSS - command line tool for managing and operating the MOSShell system.",
    rich_markup_mode=None,  # 如果你将来想用 rich，可以改为 "rich"
    no_args_is_help=True  # 没传子命令时自动显示帮助
)

app.add_typer(codex.app, name="codex", short_help="Python runtime inspect tools")
app.add_typer(workspace.app, name="ws", short_help="MOSS Workspace tools")
app.command(name='concepts', short_help="show concepts of MOSS")(concepts.show_concepts)


@app.callback(invoke_without_command=True)
def main(
        ctx: typer.Context,
        version: Optional[bool] = typer.Option(
            None, "--version", "-V", help="Show version information", is_eager=True
        ),
):
    """
    MOSS - command line tool

    This is a command line tool for MOSS (Model-oriented Operating System Shell).
    """
    if version:
        print_panel(
            f"MOSS CLI v{__version__}\n"
            f"MOSS (Model-oriented Operating System Shell)\n"
            f"Python: {sys.version.split()[0]}",
            title="Version Information"
        )
        raise typer.Exit()  # 显式退出，防止继续执行子命令

    # 如果没有子命令，typer 会因为 no_args_is_help=True 自动处理
    # 如果你想自定义处理逻辑，可以保留 ctx.invoked_subcommand 判断


@app.command("help", short_help="Show help information")
def cli_help(ctx: typer.Context):
    """
    Show complete help information
    """
    # Typer 获取父级帮助的方式与 Click 一致
    echo(ctx.get_help())


def main_entry():
    """Command line entry point"""
    try:
        # Typer 的启动方式
        app()
    except Exception as e:
        print_error(f"Command execution failed: {str(e)}")
        sys.exit(1)
