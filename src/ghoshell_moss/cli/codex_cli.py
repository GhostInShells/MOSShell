"""
Codex command group — runtime introspection tools for AI and human developers.
"""

import typer
import inspect
import importlib
import pkgutil
import os
import ast
import json
import sys
import subprocess
from pathlib import Path
from typing import Optional, List, Tuple

codex_app = typer.Typer(
    short_help="Reflect, locate, and explore Python modules at runtime.",
    help="Reflect, locate, and explore Python modules at runtime.",
    no_args_is_help=True,
)

from ghoshell_moss.cli.utils import (
    print_success, print_error, print_info, print_code, echo,
    print_simple_panel, print_simple_table, console
)


def _resolve_import_path(import_path: str):
    """Import from path, trying colon then dot-as-separator fallback."""
    from ghoshell_common.helpers import import_from_path
    try:
        return import_from_path(import_path)
    except ImportError:
        if '.' in import_path:
            parts = import_path.rsplit('.', 1)
            return import_from_path(f"{parts[0]}:{parts[1]}")
        raise


@codex_app.command("get-interface")
def get_interface(
        import_path: str = typer.Argument(
            ...,
            help="Python import path, e.g.: module.path or module.path:attr (dot also accepted as fallback)",
        )
):
    """
    Reflect a Python module or attribute and display its interface.

    For modules, recursively includes interfaces of dependency types so you
    get the full picture in one shot.  Prefer whole-module reflection — use
    :attr only when you need a single class/function to keep output small.
    """
    from ghoshell_moss.core.codex import reflect_any_by_import_path
    from ghoshell_common.helpers import generate_import_path

    try:
        value = _resolve_import_path(import_path)
    except ImportError as e:
        print_error(f"Failed to import '{import_path}': {e}")
        raise typer.Exit(code=1)

    canonical = generate_import_path(value)
    try:
        source_file = inspect.getfile(value)
    except TypeError:
        source_file = "<unknown>"

    print_info(f"Resolved: {canonical}")
    print_info(f"Source: {source_file}")

    result = reflect_any_by_import_path(canonical)
    print_code(result, language="python")


@codex_app.command("get-source")
def get_source(
        module_path: str = typer.Argument(
            ...,
            help="Python import path, e.g.: module.path or module.path:attr (dot also accepted as fallback)",
        ),
        language: str = typer.Option("python", "--language", "-l", help="Code language for syntax highlighting"),
        output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output to file instead of console",
                                              writable=True)
):
    """Read the source code of a Python module or attribute."""
    try:
        print_info(f"Importing: {module_path}")
        obj = _resolve_import_path(module_path)

        print_info(f"Getting source code...")
        source_code = inspect.getsource(obj)

        label = module_path
        try:
            source_file = inspect.getfile(obj)
        except TypeError:
            source_file = "<unknown>"

        if output:
            output.write_text(source_code, encoding="utf-8")
            print_success(f"Source code saved to: {output}")
        else:
            print_simple_panel(
                f"Target: [dim]{label}[/dim]\n"
                f"File: [dim]{source_file}[/dim]\n"
                f"Length: [dim]{len(source_code)} characters[/dim]",
                title="Source Code Information"
            )
            print_code(source_code, language=language)

    except ImportError as e:
        print_error(f"Failed to import '{module_path}': {e}")
        raise typer.Exit(code=1)
    except OSError as e:
        print_error(f"Failed to read source: {e}")
        print_info("Note: Some built-in modules or C extension modules may not have Python source code")
        raise typer.Exit(code=1)
    except Exception as e:
        print_error(f"Unknown error: {e}")
        raise typer.Exit(code=1)


@codex_app.command("where")
def where_cmd(
        import_path: str = typer.Argument(
            ...,
            help="Python import path, e.g.: module.path:attr (dot also accepted as fallback)",
        )
):
    """Show the canonical definition path of a module or attribute."""
    from ghoshell_common.helpers import generate_import_path

    try:
        value = _resolve_import_path(import_path)
    except ImportError as e:
        print_error(f"Failed to import '{import_path}': {e}")
        raise typer.Exit(code=1)

    canonical = generate_import_path(value)
    try:
        source_file = inspect.getfile(value)
    except TypeError:
        source_file = "<unknown>"

    print_success(f"Canonical: {canonical}")
    print_info(f"Source: {source_file}")


def _get_package_modules(package_path: str, recursive: bool = False, include_packages: bool = True) -> List[
    Tuple[str, str, str]]:
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

        parent_dir = parent_package.__path__[0] if isinstance(parent_package.__path__,
                                                              list) else parent_package.__path__

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


def _is_package(import_path: str) -> bool:
    """Check if an import path points to a package (has __path__), not a module."""
    try:
        obj = importlib.import_module(import_path)
        return hasattr(obj, '__path__')
    except ImportError:
        return False


def _list_module_members(module_path: str):
    """List members (classes, functions, variables) of a module for list output."""
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        print_error(f"Failed to import module '{module_path}': {e}")
        return

    members = inspect.getmembers(module)
    classes = sorted(
        (name, obj) for name, obj in members
        if inspect.isclass(obj) and obj.__module__ == module.__name__
    )
    functions = sorted(
        (name, obj) for name, obj in members
        if inspect.isfunction(obj) and obj.__module__ == module.__name__
    )
    variables = sorted(
        (name, obj) for name, obj in members
        if not name.startswith("_")
        and not inspect.isclass(obj)
        and not inspect.isfunction(obj)
        and not inspect.ismodule(obj)
    )

    table_data = []
    for name, _ in classes:
        table_data.append(["class", f"[bold cyan]{name}[/bold cyan]", f"[dim]{module_path}.{name}[/dim]", ""])
    for name, _ in functions:
        table_data.append(["function", f"[bold green]{name}[/bold green]", f"[dim]{module_path}.{name}[/dim]", ""])
    for name, _ in variables:
        table_data.append(["variable", f"[bold yellow]{name}[/bold yellow]", f"[dim]{module_path}.{name}[/dim]", ""])

    print_simple_table(
        data=table_data,
        headers=["Type", "Name", "Full Path", ""],
        title=f"Members of {module_path}",
        column_styles=["dim", "", "dim", ""],
        title_style="bold bright_cyan",
        column_ratios=[1, 1, 3, 0],
    )

    console.print(
        f"\n[dim]Total: {len(table_data)} members "
        f"({len(classes)} classes, {len(functions)} functions, {len(variables)} variables)[/dim]"
    )
    console.print(f"[dim]To see the source of a member, run [bold]moss codex get-source {module_path}:<member>[/bold][/dim]")
    console.print(f"[dim]To see member interfaces, run [bold]moss codex get-interface {module_path}:<member>[/bold][/dim]")


@codex_app.command("list")
def list_modules(
        package_path: str = typer.Argument(
            "ghoshell_moss",
            help="Python package or module path, e.g.: ghoshell_moss.core.concepts"
        ),
        recursive: bool = typer.Option(
            False,
            "--recursive", "-r",
            help="Recursively list items in subpackages"
        ),
):
    """
    List modules and packages in a Python package, or members of a module.
    """
    # 如果是 module 而非 package，列出模块成员
    if not _is_package(package_path):
        _list_module_members(package_path)
        return

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
            tips.append(
                f"To explore a specific package, run [bold]moss codex list {package_path}.<package_name>[/bold]")

    if module_count > 0:
        # 如果有模块，提示可以查看模块详情
        tips.append(f"To see module members, run [bold]moss codex list {package_path}.<module_name>[/bold]")
        tips.append(f"To see module interface, run [bold]moss codex get-interface {package_path}.<module_name>[/bold]")

    # 显示提示信息
    for i, tip in enumerate(tips):
        prefix = "• " if i > 0 else "🛈 "
        console.print(f"[dim]{prefix}{tip}[/dim]")


@codex_app.command("eval")
def eval_code(
    code: str = typer.Argument(default="", help="Python code to execute"),
    module: Optional[str] = typer.Option(
        None, "--module", "-m",
        help="Import path of module to use as execution context (types and imports available)",
    ),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f",
        help="Read code from file instead of argument",
    ),
):
    """
    Execute arbitrary Python code in the live MOSS runtime for introspection.

    Use this when you need to inspect runtime state that static reflection
    (get-interface, get-source, list) cannot reach — live object graphs,
    async coroutine state, multi-process topology, or dynamic attributes.

    Output is JSON with two fields:
      {"returns": <value>, "std_output": "<captured print() output>"}

    Assign to __result__ to return a structured value. Use print() for
    free-form output. Both work simultaneously — print() goes to std_output,
    __result__ goes to returns. Non-JSON-serializable returns are repr()'d.

    --module runs code in the context of an already-imported module, so
    its types and imports are directly available without re-importing.
    This is the "second reflection" pattern: read a module with get-interface
    or get-source, then poke its runtime state with eval --module.

    Examples:
      moss codex eval "__result__ = type(42)"
      moss codex eval -m ghoshell_moss.host.moss_runtime "print(dir())"
      moss codex eval -m ghoshell_moss "print(Channel.__mro__)"
      moss codex eval -f debug_script.py
    """
    if file:
        try:
            code = file.read_text()
        except FileNotFoundError:
            print_error(f"File not found: {file}")
            raise typer.Exit(code=1)

    if not code.strip():
        print_error("No code to execute. Provide code as argument or via --file.")
        raise typer.Exit(code=1)

    request = json.dumps({"code": code, "module": module})

    child = subprocess.run(
        [sys.executable, "-m", "ghoshell_moss.cli._eval_child"],
        input=request,
        capture_output=True,
        text=True,
        timeout=30,
    )

    if child.returncode != 0:
        print_error(f"Code execution failed:\n{child.stderr.rstrip()}")
        raise typer.Exit(code=1)

    console.print(child.stdout.rstrip())


# ---------------------------------------------------------------------------
# Knowledge index commands — reflect-generated concept explorers
# ---------------------------------------------------------------------------

CONCEPT_PACKAGE = "ghoshell_moss.core.concepts"
BLUEPRINT_PACKAGE = "ghoshell_moss.core.blueprint"
CONTRACTS_PACKAGE = "ghoshell_moss.contracts"
CHANNELS_PACKAGE = "ghoshell_moss.channels"


def _show_package_module(
        package: str,
        module_name: str | None = None,
        *,
        cmd_name: str = "",
        deps: bool = False,
) -> None:
    """Reflect concept modules from a package. Lists all if no module_name given."""
    from ghoshell_moss.core.codex.discover import scan_package
    from ghoshell_moss.core.codex import reflect_any_by_import_path

    modules = list(scan_package(package, parse=lambda x: not x.is_package))

    if module_name is None:
        if not modules:
            print_info("No concept modules found.")
            return

        table_data = []
        for mod in modules:
            desc = ' '.join(mod.short_doc.split()) if mod.short_doc else ""
            table_data.append([f"[bold cyan]{mod.module_name}[/bold cyan]", desc])

        print_simple_table(
            data=table_data,
            headers=["Module", "Description"],
            title=f"Available Modules in {package}",
            column_styles=["bold cyan", ""],
            title_style="bold bright_cyan",
            column_ratios=[1, 3],
        )

        console.print(f"\n[dim]Total: {len(modules)} modules[/dim]")
        console.print(f"[dim]Tip: Run [bold]moss codex {cmd_name} <name>[/bold] for source, "
                      f"add [bold]--deps[/bold] for full dependency reflection.[/dim]")
        return

    modules_map = {mod.module_name: mod for mod in modules}

    if module_name not in modules_map:
        print_error(f"Module '{module_name}' not found in {package}.")
        print_info("Available modules:")
        for mod in modules:
            print_info(f"  * {mod.module_name}")
        raise typer.Exit(code=1)

    import_path = modules_map[module_name].module_path

    try:
        print_info(f"Reflecting: {import_path}...")
        result = reflect_any_by_import_path(import_path, deps=deps)
        echo(result)
        if not deps:
            console.print(f"\n[dim]Use [bold]--deps[/bold] to include dependency interfaces.[/dim]")
    except Exception as e:
        print_error(f"Failed to reflect module '{import_path}': {e}")
        raise typer.Exit(code=1)


@codex_app.command(
    name='concepts',
    help="List or show detail of the core concepts of MOSS structure",
)
def codex_concepts(
        module_name: str | None = typer.Argument(
            None,
            help="Specific core concept module to reflect. If omitted, lists all available modules."
        ),
        deps: bool = typer.Option(
            False, "--deps", "-d",
            help="Include reflected dependency interfaces in the output.",
        ),
):
    _show_package_module(CONCEPT_PACKAGE, module_name, cmd_name="concepts", deps=deps)


@codex_app.command(
    name='blueprint',
    help="List or show detail of the blueprint for building model-oriented operating system from MOSS",
)
def codex_blueprint(
        module_name: str | None = typer.Argument(
            None,
            help="Specific blueprint module to reflect. If omitted, lists all available modules."
        ),
        deps: bool = typer.Option(
            False, "--deps", "-d",
            help="Include reflected dependency interfaces in the output.",
        ),
):
    _show_package_module(BLUEPRINT_PACKAGE, module_name, cmd_name="blueprint", deps=deps)


@codex_app.command(
    name='contracts',
    help="List or show detail of the basic abstract dependencies of MOSS",
)
def codex_contracts(
        module_name: str | None = typer.Argument(
            None,
            help="Specific contracts module to reflect. If omitted, lists all available modules."
        ),
        deps: bool = typer.Option(
            False, "--deps", "-d",
            help="Include reflected dependency interfaces in the output.",
        ),
):
    _show_package_module(CONTRACTS_PACKAGE, module_name, cmd_name="contracts", deps=deps)


@codex_app.command(
    name='channeltypes',
    help="List or show detail of the official channel types bundled with MOSS",
)
def codex_channeltypes(
        module_name: str | None = typer.Argument(
            None,
            help="Specific channel type module to reflect. If omitted, lists all available.",
        ),
        deps: bool = typer.Option(
            False, "--deps", "-d",
            help="Include reflected dependency interfaces in the output.",
        ),
):
    _show_package_module(CHANNELS_PACKAGE, module_name, cmd_name="channeltypes", deps=deps)


@codex_app.command(
    name='architecture',
    help="Show the core architecture map — all key modules and their roles",
)
def codex_architecture():
    """Display the curated architecture map of MOSS core abstractions.

    Imports ghoshell_moss.architecture and reflects every module in its
    __dict__ — each one is a manually curated entry in the architecture map.
    Output includes type (module/package), name, and first-line docstring.
    """
    import ghoshell_moss.architecture as arch

    table_data = []
    packages = 0
    modules = 0

    for name, value in sorted(arch.__dict__.items(), key=lambda x: x[0]):
        if name.startswith('_'):
            continue
        if not inspect.ismodule(value):
            continue

        short_doc = (value.__doc__ or '').strip().split('\n')[0]
        is_pkg = hasattr(value, '__path__')

        if is_pkg:
            kind = "[dim]package[/dim]"
            display = f"[bold magenta]{name}[/bold magenta]"
            packages += 1
        else:
            kind = "[dim]module[/dim]"
            display = f"[bold cyan]{name}[/bold cyan]"
            modules += 1

        table_data.append([kind, display, short_doc])

    print_simple_table(
        data=table_data,
        headers=["Type", "Name", "Description"],
        title="MOSS Architecture Map",
        column_styles=["", "", ""],
        title_style="bold bright_cyan",
        column_ratios=[1, 2, 5],
    )

    console.print(
        f"\n[dim]Total: {len(table_data)} entries "
        f"({modules} modules, {packages} packages)[/dim]"
    )
    console.print(
        f"[dim]Tip: [bold]moss codex get-source <name>[/bold] "
        f"for full source of a module[/dim]"
    )
    console.print(
        f"[dim]Tip: [bold]moss codex list <name>[/bold] "
        f"for members of a module or package[/dim]"
    )
