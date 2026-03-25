"""
MOSS command group - MOSShell related commands
"""

import click
import pkgutil
import importlib
import sys

from ghoshell_cli.main import main
from ghoshell_cli.utils import (
    print_error, print_info, print_panel
)


def _get_concept_modules():
    """
    Get list of concept modules from ghoshell_moss.core.concepts
    Returns list of module names without .py extension
    """
    concept_package = "ghoshell_moss.core.concepts"
    try:
        package = importlib.import_module(concept_package)
    except ImportError as e:
        print_error(f"Failed to import concept package '{concept_package}': {str(e)}")
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
        print_error(f"Failed to list modules in '{concept_package}': {str(e)}")
        return []

    return sorted(modules)


@main.group("moss")
def moss():
    """
    MOSShell related commands

    Commands for interacting with MOSShell system and concepts.
    """
    pass


@moss.command("concepts")
@click.argument("module_name", required=False)
def concepts(module_name: str = None):
    """
    Reflect concept modules from ghoshell_moss.core.concepts

    \b
    Usage:
      ghoshell moss concepts              # List all available concept modules
      ghoshell moss concepts <module>     # Reflect a specific concept module

    \b
    Examples:
      ghoshell moss concepts
      ghoshell moss concepts command
      ghoshell moss concepts channel
    """
    modules = _get_concept_modules()

    if module_name is None:
        # No module specified, show list
        if not modules:
            print_info("No concept modules found.")
            return

        print_panel(
            "\n".join([f"• {module}" for module in modules]),
            title="Available Concept Modules"
        )
        print_info(f"Total: {len(modules)} modules")
        print_info("Use 'ghoshell moss concepts <module_name>' to reflect a specific module.")
        return

    # Module specified, reflect it
    if module_name not in modules:
        print_error(f"Concept module '{module_name}' not found. Available modules:")
        for mod in modules:
            print_info(f"  • {mod}")
        sys.exit(1)

    from ghoshell_codex import reflect_any_by_import_path
    import_path = f"ghoshell_moss.core.concepts.{module_name}"
    try:
        result = reflect_any_by_import_path(import_path)
        click.echo(result)
    except Exception as e:
        print_error(f"Failed to reflect module '{import_path}': {str(e)}")
        sys.exit(1)