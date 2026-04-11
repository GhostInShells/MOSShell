"""
MOSS command group - MOSShell related commands
"""

import typer
import pkgutil
import importlib
from typing import Optional, List
from ghoshell_moss.cli.utils import (
    print_error, print_info, print_panel, echo
)

__all__ = ['show_concepts']
# 假设这是挂载在主 app 下的子 typer

CONCEPT_PACKAGE = "ghoshell_moss.core.concepts"


def _get_concept_modules() -> List[str]:
    """
    获取 ghoshell_moss.core.concepts 下的模块列表
    """
    try:
        package = importlib.import_module(CONCEPT_PACKAGE)
        if not hasattr(package, '__path__'):
            return []

        modules = [
            name for _, name, is_pkg in pkgutil.iter_modules(package.__path__)
            if not is_pkg and name != "__init__"
        ]
        return sorted(modules)
    except (ImportError, Exception) as e:
        # 在 CLI 工具中，这种内部错误建议用 print_error
        print_error(f"Failed to access concept package: {e}")
        return []


def show_concepts(
        module_name: Optional[str] = typer.Argument(
            None,
            help="Specific concept module to reflect. If omitted, lists all available modules."
        )
):
    """
    Reflect concept modules from ghoshell_moss.core.concepts.

    If MODULE_NAME is provided, reflects that specific module.
    Otherwise, lists all available concept modules.
    """
    modules = _get_concept_modules()

    # 情况 A: 用户没有输入模块名，展示列表
    if module_name is None:
        if not modules:
            print_info("No concept modules found.")
            return

        formatted_list = "\n".join([f"• [bold cyan]{mod}[/bold cyan]" for mod in modules])
        print_panel(
            formatted_list,
            title="Available Concept Modules"
        )
        print_info(f"Total: {len(modules)} modules")
        print_info(f"\nTip: Run [bold]moss concepts <name>[/bold] to see details.")
        return

    # 情况 B: 用户输入了模块名，进行校验
    if module_name not in modules:
        print_error(f"Concept module '{module_name}' not found.")
        print_info("Available modules:")
        for mod in modules:
            print_info(f"  • {mod}")
        raise typer.Exit(code=1)

    # 情况 C: 校验通过，执行反射逻辑
    from ghoshell_moss.core.codex import reflect_any_by_import_path
    import_path = f"{CONCEPT_PACKAGE}.{module_name}"

    try:
        print_info(f"Reflecting concept: {import_path}...")
        result = reflect_any_by_import_path(import_path)
        echo(result)
    except Exception as e:
        print_error(f"Failed to reflect module '{import_path}': {e}")
        raise typer.Exit(code=1)
