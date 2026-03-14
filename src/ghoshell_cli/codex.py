"""
Codex command group - code reflection and viewing tools
"""

import click
import inspect
import importlib
import sys

from ghoshell_cli.main import main
from ghoshell_cli.utils import (
    print_success, print_error, print_info, print_code, print_panel
)


@main.group("codex")
def codex():
    """
    Code reflection and viewing tools

    Provides Python code reflection, viewing and analysis functions.
    """
    pass


@codex.command("get-source")
@click.argument("module_path")
@click.option(
    "--language", "-l",
    default="python",
    help="Code language for syntax highlighting (default: python)"
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=False, writable=True),
    help="Output to file instead of console"
)
def get_source(module_path: str, language: str, output: str):
    """
    Reflect a Python module and read its source code

    \b
    MODULE_PATH: Python module import path, e.g.:
      - foo.bar
      - ghoshell_cli.main
      - click

    \b
    Examples:
      ghoshell codex get-source click
      ghoshell codex get-source ghoshell_cli.codex --language python
      ghoshell codex get-source os.path --output path.py
    """
    try:
        print_info(f"Importing module: {module_path}")
        module = importlib.import_module(module_path)

        print_info(f"Getting source code...")
        source_code = inspect.getsource(module)

        if output:
            with open(output, "w", encoding="utf-8") as f:
                f.write(source_code)
            print_success(f"Source code saved to: {output}")
        else:
            print_panel(
                f"Module: {module_path}\n"
                f"File: {inspect.getfile(module)}\n"
                f"Length: {len(source_code)} characters",
                title="Source Code Information"
            )
            print_code(source_code, language=language)

    except ImportError as e:
        print_error(f"Failed to import module '{module_path}': {str(e)}")
        sys.exit(1)
    except OSError as e:
        print_error(f"Failed to read module source: {str(e)}")
        print_info("Note: Some built-in modules or C extension modules may not have Python source code")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unknown error: {str(e)}")
        sys.exit(1)


@codex.command("info")
@click.argument("module_path")
def module_info(module_path: str):
    """
    Show detailed information about a module

    \b
    Displays:
      - File path
      - Docstring
      - Contained classes, functions and variables
      - Import dependencies
    """
    try:
        print_info(f"Analyzing module: {module_path}")
        module = importlib.import_module(module_path)

        info = []
        info.append(f"Module: {module_path}")
        info.append(f"File: {inspect.getfile(module)}")

        if module.__doc__:
            info.append(f"\nDocstring:\n{module.__doc__}")

        # Collect member information
        members = inspect.getmembers(module)
        classes = [name for name, obj in members if inspect.isclass(obj)]
        functions = [name for name, obj in members if inspect.isfunction(obj)]
        variables = [
            name for name, obj in members
            if not name.startswith("_") and
               not inspect.isclass(obj) and
               not inspect.isfunction(obj)
        ]

        info.append(f"\nClasses ({len(classes)}): {', '.join(sorted(classes))}")
        info.append(f"\nFunctions ({len(functions)}): {', '.join(sorted(functions))}")
        info.append(f"\nVariables ({len(variables)}): {', '.join(sorted(variables))}")

        print_panel("\n".join(info), title="Module Information")

    except ImportError as e:
        print_error(f"Failed to import module '{module_path}': {str(e)}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unknown error: {str(e)}")
        sys.exit(1)
