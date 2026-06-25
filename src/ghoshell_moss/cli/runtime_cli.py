"""moss runtime — inspect and manage the running cell fleet."""
import typer

from ghoshell_moss.host import Host
from .utils import console, print_simple_table

# todo: 彻底删除
runtime_app = typer.Typer(
    help="Runtime inspection and management — list cells, kill cells.",
    no_args_is_help=True,
)


@runtime_app.command(name="list-cells")
def list_cells(
    mode: str = typer.Option(None, "-m", "--mode", help="MOSS mode name."),
    session_scope: str = typer.Option(None, "-s", "--session-scope", help="Session scope."),
    show_dead: bool = typer.Option(False, "--show-dead", help="Include dead cells (PID gone, file残留)."),
):
    """List all cells registered in this scope with liveness status."""
    host = Host(mode=mode, session_scope=session_scope)

    alive_metas = host.list_scope_cells(alive_only=True)
    alive_addresses = {m.address for m in alive_metas}

    if show_dead:
        all_metas = host.list_scope_cells(alive_only=False)
    else:
        all_metas = alive_metas

    if not all_metas:
        console.print("[yellow]No cells found in this scope.[/yellow]")
        return

    table_data = []
    for m in all_metas:
        status = "[green]ALIVE[/green]" if m.address in alive_addresses else "[red]DEAD[/red]"
        ghost = m.ghost_name or "-"
        table_data.append([status, m.address, str(m.pid), ghost, m.mode_name])

    print_simple_table(
        data=table_data,
        headers=["Status", "Address", "PID", "Ghost", "Mode"],
        title=f"Cells in scope '{host.env.network_scope}'",
        column_styles=["", "cyan", "", "", ""],
        title_style="bold green",
    )
    console.print(f"\n[dim]Total: {len(all_metas)} cells ("
                  f"{len(alive_metas)} alive, "
                  f"{len(all_metas) - len(alive_metas)} dead).[/dim]")


@runtime_app.command(name="kill")
def kill_cell(
    address: str = typer.Argument(..., help="Cell address to kill, e.g. 'app/mcp/xxx'."),
    mode: str = typer.Option(None, "-m", "--mode", help="MOSS mode name."),
    session_scope: str = typer.Option(None, "-s", "--session-scope", help="Session scope."),
):
    """Kill a cell by address — terminate, wait 3s, force-kill if needed."""
    host = Host(mode=mode, session_scope=session_scope)

    meta = host.env.read_cell_meta(address, alive_only=True)
    if meta is None:
        console.print(f"[yellow]Cell '{address}' not found or already dead.[/yellow]")
        raise typer.Exit(1)

    console.print(f"Killing cell [cyan]{address}[/cyan] (PID {meta.pid})...")
    if host.kill_cell(address):
        console.print(f"[green]Cell '{address}' killed.[/green]")
    else:
        console.print(f"[yellow]Cell '{address}' PID {meta.pid} was already gone.[/yellow]")


@runtime_app.command(name="kill-all")
def kill_all(
    mode: str = typer.Option(None, "-m", "--mode", help="MOSS mode name."),
    session_scope: str = typer.Option(None, "-s", "--session-scope", help="Session scope."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
):
    """Kill all cells in this scope (except self)."""
    host = Host(mode=mode, session_scope=session_scope)

    cells = host.list_scope_cells(alive_only=True)
    if not cells:
        console.print("[yellow]No alive cells in this scope.[/yellow]")
        return

    # 过滤掉自己（CLI 进程）
    import os
    my_pid = os.getpid()
    targets = [c for c in cells if c.pid != my_pid]

    if not targets:
        console.print("[yellow]No other cells to kill (only this CLI process).[/yellow]")
        return

    console.print(f"About to kill [bold red]{len(targets)}[/bold red] cells in scope "
                  f"'[cyan]{host.env.network_scope}[/cyan]':")
    for c in targets:
        ghost_tag = f" [{c.ghost_name}]" if c.ghost_name else ""
        console.print(f"  - [cyan]{c.address}[/cyan] (PID {c.pid}){ghost_tag}")

    if not yes:
        ok = typer.confirm("\nProceed?")
        if not ok:
            console.print("[dim]Aborted.[/dim]")
            raise typer.Exit(0)

    killed = host.kill_all_cells()
    console.print(f"\n[green]Killed {len(killed)} cells:[/green]")
    for addr in killed:
        console.print(f"  - {addr}")
