"""
moss CLI - main entry point
Command line tool for Ghost In Shells
"""

import click
import sys

from ghoshell_moss.cli.utils import (
    print_error, print_info,
    print_panel, echo
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
        echo(ctx.get_help())
        print_info("Use moss <command> --help for command-specific help.")


@main.command("help")
@click.pass_context
def cli_help(ctx: click.Context):
    """
    Show complete help information
    """
    # Show detailed help information
    echo(ctx.parent.get_help())


def main_entry():
    """Command line entry point"""
    try:
        main(prog_name="moss")
    except Exception as e:
        print_error(f"Command execution failed: {str(e)}")
        sys.exit(1)
