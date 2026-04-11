"""
Codex command group - code reflection and viewing tools
"""

import typer
import inspect
import importlib
from pathlib import Path
from typing import Optional

# 假设你的 app 定义在 main.py 中
# 注意：在 Typer 中，我们通常使用 app.add_typer 来组合模块
codex_cli = typer.Typer(
    short_help="Code reflection, viewing and analysis tools.",
    help="Code reflection, viewing and analysis tools.",
    no_args_is_help=True,
)

from ghoshell_moss.cli.utils import (
    print_success, print_error, print_info, print_code, print_panel, echo
)


@codex_cli.command("get-interface")
def get_interface(
        import_path: str = typer.Argument(..., help="Python import path e.g.: [module.path][:attribute]")
):
    """
    Reflect a Python module and read its interface with detail body of class or functions.
    """
    from ghoshell_moss.core.codex import reflect_any_by_import_path
    result = reflect_any_by_import_path(import_path)
    echo(result)


@codex_cli.command("get-source")
def get_source(
        module_path: str = typer.Argument(..., help="Python module import path, e.g.: foo.bar"),
        language: str = typer.Option("python", "--language", "-l", help="Code language for syntax highlighting"),
        output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output to file instead of console",
                                              writable=True)
):
    """
    Reflect a Python module and read its source code.
    """
    try:
        print_info(f"Importing module: {module_path}")
        module = importlib.import_module(module_path)

        print_info(f"Getting source code...")
        source_code = inspect.getsource(module)

        if output:
            output.write_text(source_code, encoding="utf-8")
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
        print_error(f"Failed to import module '{module_path}': {e}")
        raise typer.Exit(code=1)
    except OSError as e:
        print_error(f"Failed to read module source: {e}")
        print_info("Note: Some built-in modules or C extension modules may not have Python source code")
        raise typer.Exit(code=1)
    except Exception as e:
        print_error(f"Unknown error: {e}")
        raise typer.Exit(code=1)


@codex_cli.command("info")
def module_info(
        module_path: str = typer.Argument(..., help="Module path to analyze")
):
    """
    Show detailed information about a module (File path, Docstring, Classes, etc.)
    """
    try:
        print_info(f"Analyzing module: {module_path}")
        module = importlib.import_module(module_path)

        # 构建信息文本
        info_lines = [
            f"Module: {module_path}",
            f"File: {inspect.getfile(module)}"
        ]

        if module.__doc__:
            info_lines.append(f"\nDocstring:\n{module.__doc__.strip()}")

        members = inspect.getmembers(module)
        classes = sorted([name for name, obj in members if inspect.isclass(obj)])
        functions = sorted([name for name, obj in members if inspect.isfunction(obj)])
        variables = sorted([
            name for name, obj in members
            if not name.startswith("_") and not inspect.isclass(obj) and not inspect.isfunction(obj)
        ])

        info_lines.append(f"\nClasses ({len(classes)}): {', '.join(classes) if classes else 'None'}")
        info_lines.append(f"\nFunctions ({len(functions)}): {', '.join(functions) if functions else 'None'}")
        info_lines.append(f"\nVariables ({len(variables)}): {', '.join(variables) if variables else 'None'}")

        print_panel("\n".join(info_lines), title="Module Information")

    except ImportError as e:
        print_error(f"Failed to import module '{module_path}': {e}")
        raise typer.Exit(code=1)
