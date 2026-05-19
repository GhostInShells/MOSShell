"""moss script — one-shot dev-time scripts that connect to the running matrix."""
import subprocess
import sys
from pathlib import Path

import typer

from ghoshell_moss.host import Host
from .utils import console, print_simple_table, print_simple_panel, print_host_mode_info

script_app = typer.Typer(
    help="One-shot dev-time scripts for debugging and probing the running matrix.",
    no_args_is_help=True,
)


def _find_script_dir(host: Host, name: str) -> Path | None:
    """Find a script directory by name under the workspace scripts/ tree."""
    scripts_root = host.env.workspace_path / "scripts"
    if not scripts_root.is_dir():
        return None
    candidate = scripts_root / name
    if candidate.is_dir():
        return candidate
    return None


def _read_script_md(script_dir: Path) -> dict:
    """Parse SCRIPT.md frontmatter. Returns empty dict if missing or unparseable."""
    md_path = script_dir / "SCRIPT.md"
    if not md_path.is_file():
        return {}
    try:
        import frontmatter
        post = frontmatter.load(str(md_path))
        return dict(post.metadata) if post.metadata else {}
    except Exception:
        return {}


@script_app.command(name="list")
def list_scripts(
        json_out: bool = typer.Option(False, "--json", help="Output raw JSON."),
        mode: str = typer.Option(None, "-m", "--mode", help="MOSS mode name"),
):
    """List all scripts discovered in the workspace."""
    host = Host(mode=mode)
    scripts_root = host.env.workspace_path / "scripts"
    if not scripts_root.is_dir():
        if json_out:
            console.json(data=[])
        else:
            console.print("[yellow]No scripts/ directory found in workspace.[/yellow]")
        return

    results = []
    for entry in sorted(scripts_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("_") or entry.name.startswith("."):
            continue
        meta = _read_script_md(entry)
        results.append({
            "name": entry.name,
            "path": str(entry),
            "description": meta.get("description", ""),
        })

    if json_out:
        console.json(data=results)
        return

    if not results:
        console.print("[yellow]No scripts found.[/yellow]")
        return

    table_data = [
        [f"[cyan]{r['name']}[/cyan]", r["description"]]
        for r in results
    ]
    print_simple_table(
        data=table_data,
        headers=["Name", "Description"],
        title="MOSS Scripts",
        column_styles=["cyan", ""],
        title_style="bold green",
    )
    console.print(f"\n[dim]Total: {len(results)} scripts.[/dim]")
    console.print("[dim]Hint: Use [bold]moss script run <name>[/bold] to execute.[/dim]")


@script_app.command(name="run")
def run_script(
        name: str = typer.Argument(..., help="Script name (subdirectory under scripts/)."),
        mode: str | None = typer.Option(None, "-m", "--mode", help="MOSS mode name."),
        session_scope: str | None = typer.Option(None, "-s", "--session-scope"),
):
    """Run a one-shot script as a foreground subprocess.

    The script connects to the running matrix network via Zenoh,
    performs its task (e.g. sends a signal to a ghost), and exits.

    Scripts reuse the moss Python runtime — no pyproject.toml isolation.
    """
    host = Host(mode=mode, session_scope=session_scope)

    script_dir = _find_script_dir(host, name)
    if script_dir is None:
        console.print(f"[red]Error: Script '{name}' not found.[/red]")
        raise typer.Exit(1)

    meta = _read_script_md(script_dir)
    script_file = script_dir / "main.py"
    if not script_file.is_file():
        console.print(f"[red]Error: No main.py found in {script_dir}[/red]")
        raise typer.Exit(1)

    if not script_dir.samefile(host.env.workspace_path / "scripts" / name):
        console.print(f"[red]Error: Script path mismatch.[/red]")
        raise typer.Exit(1)

    address = f"script/{name}"
    description = meta.get("description", "")

    console.print(
        f"[bold green]Script:[/bold green] {name}\n"
        f"[bold blue]Directory:[/bold blue] {script_dir}\n"
        f"[bold blue]Address:[/bold blue] {address}\n"
        f"[bold blue]Description:[/bold blue] {description}\n"
    )

    env = host.env.dump_moss_env(
        cell_address=address,
        for_child_process=True,
        with_os_env=False,
    )

    try:
        console.print("[dim]—— Script Started (Ctrl+C to stop) ——[/dim]\n")
        subprocess.run(
            args=[sys.executable, str(script_file)],
            cwd=script_dir,
            env=env,
            check=False,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Script interrupted by user.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Failed to run script: {e}[/red]")
        raise typer.Exit(1)
    finally:
        console.print("\n[dim]—— Script Session Ended ——[/dim]")


@script_app.command(name="init")
def init_script(
        name: str = typer.Argument(..., help="Script name (creates scripts/<name>/)."),
        description: str = typer.Option("", "-d", "--description", help="Script description."),
        mode: str = typer.Option(None, "-m", "--mode", help="MOSS mode name."),
):
    """Create a new script from the template.

    Creates scripts/<name>/ with SCRIPT.md and main.py.
    Scripts that start with '_' are excluded from discovery (examples/internal).
    """
    host = Host(mode=mode)
    scripts_root = host.env.workspace_path / "scripts"

    if not scripts_root.is_dir():
        console.print("[yellow]Creating scripts/ directory in workspace.[/yellow]")
        scripts_root.mkdir(parents=True, exist_ok=True)

    script_dir = scripts_root / name
    if script_dir.exists():
        console.print(f"[red]Error: Script '{name}' already exists at {script_dir}[/red]")
        raise typer.Exit(1)

    script_dir.mkdir(parents=True)

    script_md = script_dir / "SCRIPT.md"
    script_md.write_text(
        f"---\ndescription: {description or 'One-shot dev-time script.'}\n---\n"
    )

    main_py = script_dir / "main.py"
    main_py.write_text(
        '"""%s"""\n'
        "from ghoshell_moss.core.blueprint.mindflow import Signal\n"
        "from ghoshell_moss.core.blueprint.matrix import Matrix\n"
        "\n"
        "\n"
        "async def main():\n"
        '    async with Matrix.discover() as matrix:\n'
        '        signal = Signal.new(\n'
        '            name="input",\n'
        '            description="hello from script cell",\n'
        "        )\n"
        "        matrix.session.add_signal(signal)\n"
        '        print(f"Signal sent: {signal.id}")\n'
        "\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    import asyncio\n"
        "    asyncio.run(main())\n"
        % (description or "MOSS dev-time script."),
    )

    console.print(f"[green]Script '{name}' created at {script_dir}[/green]")
    console.print(f"\n[dim]Next: moss script run {name}[/dim]")
