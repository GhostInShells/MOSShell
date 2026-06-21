import typer
from rich.syntax import Syntax
from .utils import console, print_simple_table, print_info
from ghoshell_moss.host import Environment  # 确保导入路径正确
from ghoshell_moss.core.ctml.versions import CTML_VERSION

ctml_app = typer.Typer(
    help="MOSS CTML Manager: Manage and introspect CTML versions and meta-instructions.",
    no_args_is_help=True
)


@ctml_app.command(name="list")
def list_ctml_versions(
        json_out: bool = typer.Option(False, "--json", help="Output raw JSON for AI consumption."),
):
    """
    List all available CTML versions discovered in MOSS environment.
    """
    # 发现环境
    env = Environment.discover()
    default_version = env.moss_meta.ctml_version
    print_info(f"Current CTML versions: {default_version}")
    versions = env.ctml_versions()

    if not versions:
        console.print("[yellow]No CTML versions found in environment.[/yellow]")
        return

    if json_out:
        data = {v: str(p.absolute()) for v, p in versions.items()}
        console.json(data=data)
        return

    # 获取系统默认路径以便区分来源
    from ghoshell_moss.core.ctml.versions import default_moss_ctml_meta_instruction_directory
    builtin_dir = default_moss_ctml_meta_instruction_directory()

    # 准备数据行
    rows = []
    for version, path in sorted(versions.items()):
        is_builtin = str(path).startswith(str(builtin_dir))
        source_label = "Built-in" if is_builtin else "Workspace"
        rows.append([version, str(path.absolute()), source_label])

    # 使用极致简洁风格打印
    print_simple_table(
        data=rows,
        headers=["Version", "Location", "Source"],
        title="MOSS CTML Versions",
        column_styles=["cyan", "green", "dim"],
        header_style="bold magenta"
    )


@ctml_app.command(name="read")
def read_ctml_version(
        version: str = typer.Argument(
            CTML_VERSION,
            help=f"The CTML version name to read (e.g., '1.0.0') default: {CTML_VERSION}"
        ),
        raw: bool = typer.Option(False, "--raw", help="Output raw content without syntax highlighting.")
):
    """
    Read the content of a specific CTML version.
    """
    env = Environment.discover()
    versions = env.ctml_versions()

    if version not in versions:
        console.print(f"[red]Error: CTML version '{version}' not found.[/red]")
        # 顺便展示可用的版本
        available = ", ".join(versions.keys())
        console.print(f"[dim]Available versions: {available}[/dim]")
        raise typer.Exit(code=1)

    file_path = versions[version]

    try:
        content = file_path.read_text(encoding="utf-8")

        if raw:
            print(content)
        else:
            console.print(f"[bold blue]Source: {file_path.absolute()}[/bold blue]\n")
            # CTML 通常是 Markdown 格式的提示词
            syntax = Syntax(content, "markdown", theme="monokai", line_numbers=True)
            console.print(syntax)

    except Exception as e:
        console.print(f"[red]Error reading file: {e}[/red]")
        raise typer.Exit(code=1)
