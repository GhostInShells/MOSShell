"""
moss CLI - main entry point
Command line tool for Ghost In Shells
"""

import click
import sys
from typing import Optional

from ghoshell_moss.cli.utils import (
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
    MOSS - command line tool

    This is a command line tool for MOSS (Model-oriented Operating System Shell), used for
    managing and operating the MOSShell system.

    Use moss <command> --help to see help for specific commands.
    """
    if version:
        print_panel(
            f"MOSS CLI v{__version__}\n"
            f"MOSS (Model-oriented Operating System Shell)\n"
            f"Python: {sys.version.split()[0]}",
            title="Version Information"
        )
        return

    # Show help if no subcommand provided
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        print_info("Use moss <command> --help for command-specific help.")


@main.command("help")
@click.pass_context
def moss_help(ctx):
    """
    Show complete help information
    """
    # Show detailed help information
    click.echo(ctx.parent.get_help())

    # Show additional tips if console is available
    console = get_console()
    if console:
        console.print("\n[yellow]Tips:[/yellow]")
        console.print("  • Use [bold]moss --version[/bold] to show version")
        console.print("  • Use [bold]moss <command> --help[/bold] for command help")


def main_entry():
    """Command line entry point"""
    try:
        main(prog_name="moss")
    except Exception as e:
        print_error(f"Command execution failed: {str(e)}")
        sys.exit(1)


