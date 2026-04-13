from typing import List
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from .utils import console
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

    table = Table(title="[bold yellow]MOSS Discovered Modes[/bold yellow]", box=None)
    table.add_column("Name", style="green", no_wrap=True)
    table.add_column("Apps (Allowed)", style="cyan")
    table.add_column("Bring-up", style="magenta")
    table.add_column("Description", ratio=1)

    for name, m in modes.items():
        # 处理显示逻辑，如果是 * 则显示 ALL
        apps_str = ", ".join(m.apps) if m.apps != ["*"] else "[dim]ALL[/dim]"
        up_str = ", ".join(m.bringup) if m.bringup else "[dim]None[/dim]"

        table.add_row(
            name,
            apps_str,
            up_str,
            m.description
        )

    console.print(table)
    console.print(f"\n[dim]Total: {len(modes)} modes found.[/dim]")
    console.print("[dim]Use 'moss-cli modes show <name>' to see instructions.[/dim]")


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
    console.print(Panel(f"[bold green]Mode: {m.name}[/bold green]", border_style="cyan"))

    # 打印基础元数据
    meta_table = Table.grid(padding=(0, 2))
    meta_table.add_row("[bold]File Path:[/bold]", m.file)
    meta_table.add_row("[bold]Import Path:[/bold]", m.import_path or "[dim]N/A (Markdown Only)[/dim]")
    meta_table.add_row("[bold]Description:[/bold]", m.description)
    console.print(meta_table)

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
