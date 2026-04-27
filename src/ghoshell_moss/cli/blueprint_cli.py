"""
MOSS command group - Blueprint related commands
By: Deepseek v3.2
"""

import click
import pkgutil
import importlib
import sys

from ghoshell_moss.cli.main import main
from ghoshell_moss.cli.utils import (
    print_error, print_info, print_panel, echo,
    print_simple_table, console
)


def _get_blueprint_modules():
    """
    Get list of blueprint modules from ghoshell_moss.core.blueprint
    Returns list of module names without .py extension
    """
    blueprint_package = "ghoshell_moss.core.blueprint"
    try:
        package = importlib.import_module(blueprint_package)
    except ImportError as e:
        print_error(f"Failed to import blueprint package '{blueprint_package}': {str(e)}")
        return []

    modules = []
    try:
        # Some packages may not have __path__ attribute (e.g., namespace packages)
        if not hasattr(package, '__path__'):
            return []

        for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
            if not is_pkg and name != "__init__":
                modules.append(name)
    except Exception as e:
        print_error(f"Failed to list modules in '{blueprint_package}': {str(e)}")
        return []

    return sorted(modules)


@main.command("blueprint")
@click.argument("module_name", required=False)
def blueprint(module_name: str = None):
    """
    Reflect blueprint modules from ghoshell_moss.core.blueprint

    \b
    Usage:
      ghoshell moss blueprint              # List all available blueprint modules
      ghoshell moss blueprint <module>     # Reflect a specific blueprint module

    \b
    Examples:
      ghoshell moss blueprint
      ghoshell moss blueprint builder
      ghoshell moss blueprint provider
    """
    modules = _get_blueprint_modules()

    if module_name is None:
        # No module specified, show list
        if not modules:
            print_info("No blueprint modules found.")
            return

        # 准备表格数据
        table_data = []
        for module in modules:
            table_data.append([f"[cyan]{module}[/cyan]"])

        # 使用简洁表格显示
        print_simple_table(
            data=table_data,
            headers=["Blueprint Module"],
            title="Available Blueprint Modules",
            column_styles=["cyan"],
            title_style="bold cyan",
        )

        console.print(f"\n[dim]Total: {len(modules)} modules[/dim]")
        console.print(f"[dim]Use [bold]moss blueprint <module_name>[/bold] to reflect a specific module.[/dim]")
        return

    # Module specified, reflect it
    if module_name not in modules:
        print_error(f"Blueprint module '{module_name}' not found. Available modules:")
        for mod in modules:
            print_info(f"  • {mod}")
        sys.exit(1)

    from ghoshell_moss.core.codex import reflect_any_by_import_path
    import_path = f"ghoshell_moss.core.blueprint.{module_name}"
    try:
        result = reflect_any_by_import_path(import_path)
        echo(result)
    except Exception as e:
        print_error(f"Failed to reflect module '{import_path}': {str(e)}")
        sys.exit(1)