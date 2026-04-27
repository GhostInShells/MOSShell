"""
Codex command group - code reflection and viewing tools
"""

import typer
import inspect
import importlib
import pkgutil
import os
import ast
from pathlib import Path
from typing import Optional, List, Tuple

# 假设你的 app 定义在 main.py 中
# 注意：在 Typer 中，我们通常使用 app.add_typer 来组合模块
codex_app = typer.Typer(
    short_help="Code reflection, viewing and analysis tools.",
    help="Code reflection, viewing and analysis tools.",
    no_args_is_help=True,
)

from ghoshell_moss.cli.utils import (
    print_success, print_error, print_info, print_code, print_panel, echo,
    print_simple_panel, print_simple_table, console
)


@codex_app.command("get-interface")
def get_interface(
        import_path: str = typer.Argument(..., help="Python import path e.g.: [module.path][:attribute]")
):
    """
    Reflect a Python module and read its interface with detail body of class or functions.
    """
    from ghoshell_moss.core.codex import reflect_any_by_import_path
    result = reflect_any_by_import_path(import_path)
    echo(result)


@codex_app.command("get-source")
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
            print_simple_panel(
                f"Module: [dim]{module_path}[/dim]\n"
                f"File: [dim]{inspect.getfile(module)}[/dim]\n"
                f"Length: [dim]{len(source_code)} characters[/dim]",
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


@codex_app.command("info")
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

        print_simple_panel("\n".join(info_lines), title="Module Information")

    except ImportError as e:
        print_error(f"Failed to import module '{module_path}': {e}")
        raise typer.Exit(code=1)


def _get_package_modules(package_path: str, recursive: bool = False, include_packages: bool = True) -> List[Tuple[str, str, str]]:
    """
    获取指定包下的模块和包列表

    Args:
        package_path: 包路径
        recursive: 是否递归查找子包中的模块
        include_packages: 是否在结果中包含包本身

    Returns:
        列表，每个元素是 (完整导入路径, 名称, 类型)
        类型: "package" 或 "module"
    """
    try:
        package = importlib.import_module(package_path)
        if not hasattr(package, '__path__'):
            return []

        result = []

        for _, name, is_pkg in pkgutil.iter_modules(package.__path__):
            if name == "__init__":
                continue

            full_path = f"{package_path}.{name}"

            if is_pkg:
                # 这是一个包
                if include_packages:
                    result.append((full_path, name, "package"))

                if recursive:
                    # 递归查找子包中的模块
                    sub_items = _get_package_modules(full_path, recursive, include_packages)
                    result.extend(sub_items)
            else:
                # 这是一个模块
                result.append((full_path, name, "module"))

        return sorted(result, key=lambda x: x[0])  # 按完整路径排序
    except (ImportError, Exception) as e:
        print_error(f"Failed to access package '{package_path}': {e}")
        return []


def _get_item_description(full_path: str, item_type: str) -> str:
    """
    获取模块或包的描述（第一个无主的字符串）

    Args:
        full_path: 完整的导入路径，如 ghoshell_moss.core.concepts.channel
        item_type: 类型，"package" 或 "module"
    """
    try:
        # 解析完整路径，获取名称和父包路径
        parts = full_path.split('.')
        if len(parts) < 2:
            return ""

        item_name = parts[-1]
        parent_path = '.'.join(parts[:-1])

        # 获取父包
        parent_package = importlib.import_module(parent_path)
        if not hasattr(parent_package, '__path__'):
            return ""

        parent_dir = parent_package.__path__[0] if isinstance(parent_package.__path__, list) else parent_package.__path__

        # 确定要读取的文件
        if item_type == "module":
            file_path = os.path.join(parent_dir, f"{item_name}.py")
        else:  # package
            file_path = os.path.join(parent_dir, item_name, "__init__.py")
            # 如果包没有 __init__.py 文件，尝试读取包目录下的同名 .py 文件
            if not os.path.exists(file_path):
                alt_file_path = os.path.join(parent_dir, f"{item_name}.py")
                if os.path.exists(alt_file_path):
                    file_path = alt_file_path
                else:
                    return ""

        if not os.path.exists(file_path):
            return ""

        # 读取文件并解析 AST
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content)

        # 方法1: 使用 ast.get_docstring 获取文档字符串
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


@codex_app.command("list")
def list_modules(
    package_path: str = typer.Argument(
        "ghoshell_moss",
        help="Python package path to list modules and packages from, e.g.: ghoshell_moss.core.concepts"
    ),
    recursive: bool = typer.Option(
        False,
        "--recursive", "-r",
        help="Recursively list items in subpackages"
    ),
):
    """
    List all modules and packages in a Python package with their descriptions.
    """
    # 获取模块和包列表
    items = _get_package_modules(package_path, recursive, include_packages=True)

    if not items:
        print_info(f"No modules or packages found in package '{package_path}'.")
        return

    # 获取每个项目的描述
    item_descriptions = []
    for full_path, name, item_type in items:
        desc = _get_item_description(full_path, item_type)
        item_descriptions.append((full_path, name, item_type, desc))

    # 准备表格数据
    table_data = []
    for full_path, name, item_type, desc in item_descriptions:
        # 根据类型格式化名称：包显示为 "包名/"，模块显示为 "模块名"
        if item_type == "package":
            display_name = f"[bold magenta]{name}/[/bold magenta]"
        else:  # module
            display_name = f"[bold cyan]{name}[/bold cyan]"

        if desc:
            # 如果描述有多行，合并为单行
            desc_single_line = ' '.join(desc.split())
            table_data.append([display_name, f"[dim]{full_path}[/dim]", desc_single_line])
        else:
            table_data.append([display_name, f"[dim]{full_path}[/dim]", ""])

    # 使用简洁表格显示
    title = f"Items in {package_path}"
    if recursive:
        title += " (recursive)"

    print_simple_table(
        data=table_data,
        headers=["Name", "Full Path", "Description"],
        title=title,
        column_styles=["", "dim", ""],  # 名称列样式由 display_name 决定
        title_style="bold bright_cyan",
        column_ratios=[1, 2, 2],  # 名称:完整路径:描述 = 1:2:2
    )

    # 统计信息
    package_count = sum(1 for _, _, item_type, _ in item_descriptions if item_type == "package")
    module_count = sum(1 for _, _, item_type, _ in item_descriptions if item_type == "module")

    console.print(f"\n[dim]Total: {len(items)} items ({module_count} modules, {package_count} packages)[/dim]")

    # 动态提示信息
    tips = []

    if package_count > 0:
        # 如果有包，提示可以进一步列出包内容
        if recursive:
            # 递归模式下，提示可以查看具体包的内容
            tips.append(f"To explore a package, run [bold]moss codex list {package_path}.<package_name>[/bold]")
        else:
            # 非递归模式下，提示可以递归查看或查看具体包
            tips.append(f"To explore packages recursively, run [bold]moss codex list --recursive[/bold]")
            tips.append(f"To explore a specific package, run [bold]moss codex list {package_path}.<package_name>[/bold]")

    if module_count > 0:
        # 如果有模块，提示可以查看模块详情
        tips.append(f"To see module details, run [bold]moss codex info {package_path}.<module_name>[/bold]")

    # 显示提示信息
    for i, tip in enumerate(tips):
        prefix = "• " if i > 0 else "🛈 "
        console.print(f"[dim]{prefix}{tip}[/dim]")
