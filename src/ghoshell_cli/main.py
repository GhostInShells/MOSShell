"""
ghoshell CLI - main entry point
Command line tool for Ghost In Shells
"""

import click
import sys
from typing import Optional

from ghoshell_cli.utils import (
    print_success, print_error, print_warning, print_info,
    print_panel, get_console
)

__version__ = "0.1.0-alpha"


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True
)
@click.option(
    "--version", "-V",
    is_flag=True,
    help="Show version information"
)
@click.pass_context
def main(ctx: click.Context, version: bool):
    """
    ghoshell - Ghost In Shells command line tool

    This is a command line tool for AI Operating System Shell, used for
    managing and operating the MOSShell system.

    Use ghoshell <command> --help to see help for specific commands.
    """
    if version:
        print_panel(
            f"ghoshell CLI v{__version__}\n"
            f"MOS-Shell (Model-oriented Operating System Shell)\n"
            f"Python: {sys.version.split()[0]}",
            title="Version Information"
        )
        return

    # Show help if no subcommand provided
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        print_info("Use ghoshell <command> --help for command-specific help.")


@main.command("help")
@click.pass_context
def ghoshell_help(ctx):
    """
    Show complete help information
    """
    # Show detailed help information
    click.echo(ctx.parent.get_help())

    # Show additional tips if console is available
    console = get_console()
    if console:
        console.print("\n[yellow]Tips:[/yellow]")
        console.print("  • Use [bold]ghoshell --version[/bold] to show version")
        console.print("  • Use [bold]ghoshell <command> --help[/bold] for command help")


def main_entry():
    """Command line entry point"""
    try:
        main(prog_name="ghoshell")
    except Exception as e:
        print_error(f"Command execution failed: {str(e)}")
        sys.exit(1)


