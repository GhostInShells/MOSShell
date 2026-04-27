from typing import List
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from .utils import console, print_simple_table, print_simple_panel
import typer

from ghoshell_moss.host import Host

# by gemini 3
mode_app = typer.Typer(help="Manage MOSS Host Modes (Environment Isolation).", no_args_is_help=True)


@mode_app.command(name="list")
def list_modes():
    """
    List all discovered modes in the current MOSS workspace.
    """
    host = Host()
    modes = host.all_modes()

    # 准备表格数据
    table_data = []
    for name, m in modes.items():
        # 处理显示逻辑，如果是 * 则显示 ALL
        apps_str = ", ".join(m.apps) if m.apps != ["*"] else "[dim]ALL[/dim]"
        up_str = ", ".join(m.bringup) if m.bringup else "[dim]None[/dim]"

        table_data.append([
            f"[green]{name}[/green]",
            apps_str,
            up_str,
            m.description
        ])

    # 使用简洁表格显示
    print_simple_table(
        data=table_data,
        headers=["Name", "Apps (Allowed)", "Bring-up", "Description"],
        title="MOSS Discovered Modes",
        column_styles=["green", "cyan", "magenta", ""],
        title_style="bold yellow",
    )

    console.print(f"\n[dim]Total: {len(modes)} modes found.[/dim]")
    console.print(f"[dim]Use [bold]moss modes show <name>[/bold] to see instructions.[/dim]")


@mode_app.command(name="show")
def show_mode(name: str):
    """
    Show detailed information and instructions for a specific mode.
    """
    host = Host()
    modes = host.all_modes()

    if name not in modes:
        console.print(f"[red]Error: Mode '{name}' not found.[/red]")
        raise typer.Exit(1)

    m = modes[name]

    # 使用简洁面板显示模式基本信息
    content = (
        f"File Path: [dim]{m.file}[/dim]\n"
        f"Import Path: [dim]{m.import_path or 'N/A (Markdown Only)'}[/dim]\n"
        f"Description: [dim]{m.description}[/dim]"
    )
    print_simple_panel(content, title=f"Mode: {m.name}")

    # 打印指令内容
    if m.instruction:
        console.print("\n[bold cyan]Instruction (MODE.md):[/bold cyan]")
        console.print(Syntax(m.instruction, "markdown", theme="monokai", background_color="default"))
    else:
        console.print("\n[yellow]No custom instruction defined for this mode.[/yellow]")


@mode_app.command(name="create")
def create_mode(
        name: str = typer.Argument(..., help="Unique name for the new mode."),
        description: str = typer.Option("", "--desc", "-d", help="One-line description."),
        apps: List[str] = typer.Option(["*"], "--app", "-a", help="Allowed app patterns (can repeat)."),
        up: List[str] = typer.Option([], "--up", "-u", help="Bring-up app patterns (can repeat)."),
):
    """
    Create a new MOSS Mode with a MODE.md file.
    """
    host = Host()
    try:
        host.new_mode(
            name=name,
            apps=apps,
            bring_up_apps=up,
            description=description
        )
        console.print(f"[green]Successfully created mode '{name}'.[/green]")
        console.print(f"[dim]You can now edit the MODE.md in your modes directory to add instructions.[/dim]")
    except NameError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to create mode:[/red] {e}")
        raise typer.Exit(1)

# 最后在主 app 中注册
# app.add_typer(mode_app, name="modes")
