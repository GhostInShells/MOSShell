"""
MOSS command group - MOSShell related commands
"""

import typer
import pkgutil
import importlib
import ast
import os
from typing import Optional, List, Tuple
from ghoshell_moss.cli.utils import (
    print_error, print_info, print_panel, echo,
    print_simple_panel, print_simple_table, console
)
from ghoshell_moss.core.codex.discover import scan_package
from ghoshell_moss.core.codex import reflect_any_by_import_path

__all__ = ['show_core_concepts']
# 假设这是挂载在主 app 下的子 typer

CONCEPT_PACKAGE = "ghoshell_moss.core.concepts"
BLUEPRINT_PACKAGE = "ghoshell_moss.core.blueprint"
CONTRACTS_PACKAGE = "ghoshell_moss.contracts"

codex_app = typer.Typer(
    short_help="Show moss concepts",
    help="Show moss concepts by reflect source code, follow `code as prompt` principle",
    no_args_is_help=True,
)


@codex_app.command(
    name='core',
    help="list or show detail of the core concepts of moss structure",
)
def show_core_concepts(
        module_name: Optional[str] = typer.Argument(
            None,
            help="Specific core concept module to reflect. If omitted, lists all available modules."
        )
):
    _show_package_module(CONCEPT_PACKAGE, module_name)


@codex_app.command(
    name='blueprint',
    help="list or show detail of the blueprint of building model-oriented operating system from MOSS",
)
def show_blueprint(
        module_name: Optional[str] = typer.Argument(
            None,
            help="Specific blueprint module to reflect. If omitted, lists all available modules."
        )
):
    _show_package_module(BLUEPRINT_PACKAGE, module_name)


@codex_app.command(
    name='contracts',
    help="list or show detail of the basic abstract dependencies of the MOSS.",
)
def show_contracts(
        module_name: Optional[str] = typer.Argument(
            None,
            help="Specific contracts module to reflect. If omitted, lists all available modules."
        )
):
    _show_package_module(CONTRACTS_PACKAGE, module_name)


def _show_package_module(package: str, module_name: Optional[str] = None) -> None:
    """
    Reflect concept modules from ghoshell_moss.core.concepts.

    If MODULE_NAME is provided, reflects that specific module.
    Otherwise, lists all available concept modules.
    """
    console.print(f"\n[dim]Found modules in package '{package}'[/dim]")
    modules = list(scan_package(package, parse=lambda x: not x.is_package))

    # 情况 A: 用户没有输入模块名，展示列表
    if module_name is None:
        if not modules:
            print_info("No concept modules found.")
            return

        module_descriptions = []
        for mod in modules:
            desc = mod.short_doc
            module_descriptions.append((mod.module_name, desc))

        # 准备表格数据
        table_data = []
        for mod, desc in module_descriptions:
            if desc:
                # 如果描述有多行，合并为单行
                desc_single_line = ' '.join(desc.split())
                table_data.append([f"[bold cyan]{mod}[/bold cyan]", desc_single_line])
            else:
                table_data.append([f"[bold cyan]{mod}[/bold cyan]", ""])

        # 使用简洁表格显示
        print_simple_table(
            data=table_data,
            headers=["Module", "Description"],
            title="Available Concept Modules",
            column_styles=["bold cyan", ""],
            title_style="bold bright_cyan",
            column_ratios=[1, 3],  # 模块名列占1份，描述列占3份
        )

        console.print(f"\n[dim]Total: {len(modules)} modules[/dim]")
        console.print(f"[dim]Tip: Run [bold]moss concepts <name>[/bold] to see details.[/dim]")
        return

    modules_map = {mod.module_name: mod for mod in modules}

    # 情况 B: 用户输入了模块名，进行校验
    if module_name not in modules_map:
        print_error(f"Concept module '{module_name}' not found.")
        print_info("Available modules:")
        for mod in modules:
            print_info(f"  • {mod}")
        raise typer.Exit(code=1)

    import_path = modules_map[module_name].module_path

    try:
        print_info(f"Reflecting concept: {import_path}...")
        result = reflect_any_by_import_path(import_path)
        echo(result)
    except Exception as e:
        print_error(f"Failed to reflect module '{import_path}': {e}")
        raise typer.Exit(code=1)
