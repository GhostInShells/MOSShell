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


def _get_concept_description(module_name: str) -> str:
    """
    获取概念模块的描述（第一个无主的字符串）
    """
    try:
        # 构建模块文件路径
        package = importlib.import_module(CONCEPT_PACKAGE)
        if not hasattr(package, '__path__'):
            return ""

        package_path = package.__path__[0] if isinstance(package.__path__, list) else package.__path__
        module_file = os.path.join(package_path, f"{module_name}.py")

        if not os.path.exists(module_file):
            return ""

        # 读取文件并解析 AST
        with open(module_file, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        # 方法1: 使用 ast.get_docstring 获取模块文档字符串
        desc = ast.get_docstring(tree)
        if desc:
            # 清理多余的空白和换行，合并为单行
            desc = ' '.join(desc.split())
            return desc

        # 方法2: 遍历模块的语句，找到第一个字符串表达式
        for node in tree.body:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    desc = node.value.value.strip()
                    if desc:
                        desc = ' '.join(desc.split())
                        return desc
    except Exception:
        # 任何错误都返回空字符串
        pass

    return ""


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

        # 获取每个模块的描述
        module_descriptions = []
        for mod in modules:
            desc = _get_concept_description(mod)
            module_descriptions.append((mod, desc))

        # 准备表格数据
        table_data = []
        for mod, desc in module_descriptions:
            if desc:
                # 如果描述有多行，合并为单行
                desc_single_line = ' '.join(desc.split())
                # 显示前100个字符，如果超过则添加提示
                if len(desc_single_line) > 100:
                    display_desc = desc_single_line[:97] + "..."
                else:
                    display_desc = desc_single_line
                table_data.append([f"[bold cyan]{mod}[/bold cyan]", display_desc])
            else:
                table_data.append([f"[bold cyan]{mod}[/bold cyan]", ""])

        # 使用简洁表格显示
        print_simple_table(
            data=table_data,
            headers=["Module", "Description"],
            title="Available Concept Modules",
            column_styles=["bold cyan", ""],
            title_style="bold cyan underline",
        )

        console.print(f"\n[dim]Total: {len(modules)} modules[/dim]")
        console.print(f"[dim]Tip: Run [bold]moss concepts <name>[/bold] to see details.[/dim]")
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
