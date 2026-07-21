import os
import stat

import typer
from pathlib import Path

project_app = typer.Typer(
    help="MOSS Project tools — inspect workspace location and set up environment.",
    no_args_is_help=True,
)

from .utils import (
    print_simple_table,
    print_simple_panel,
    print_error,
    print_warning,
    print_info,
    print_success,
    echo,
)


_DESIGN_MD = Path(__file__).resolve().parents[1] / "project" / "DESIGN.md"


@project_app.command(
    name="where",
    short_help="Show current workspace location and status.",
)
def where() -> None:
    from ghoshell_moss.core.blueprint.environment import Environment

    try:
        env = Environment.discover()
    except EnvironmentError as e:
        print_error(f"Environment Discovery Failed: {e}")
        fallback = Environment.find_workspace_path()
        print_info(f"MOSS was looking for: {fallback}")
        raise typer.Exit(code=1)

    ws_path = env.workspace_path
    project_path = env.project_path
    exists = ws_path.exists()

    perm_status = "N/A"
    if exists:
        mode = ws_path.stat().st_mode
        is_group_writable = bool(mode & stat.S_IWGRP)
        is_setgid = bool(mode & stat.S_ISGID)
        parts = []
        if is_group_writable:
            parts.append("Group-Writable")
        if is_setgid:
            parts.append("Setgid")
        perm_status = f"OK ({' & '.join(parts)})" if parts else "Restricted"

    meta = env.moss_meta
    moss_md = ws_path / 'MOSS.md'

    print_simple_table(
        data=[
            ["Project Root", str(project_path.absolute())],
            ["Workspace", str(ws_path.absolute())],
            ["Status", "Active" if exists else "Not Found"],
            ["Permissions", perm_status],
            ["Project Name", meta.name],
            ["Default Mode", meta.default_mode],
            ["Default Ghost", meta.default_ghost],
            ["Default Network", meta.default_network],
            ["Network Scope", meta.default_network_scope],
            ["MOSS.md", str(moss_md) if moss_md.exists() else "Missing"],
        ],
        headers=["Property", "Value"],
        title="MOSS Project",
    )


@project_app.command(
    name="env-init",
    short_help="Show the .env.example template and explain how to create .env",
)
def env_init() -> None:
    from ghoshell_moss.core.blueprint.environment import Environment

    try:
        env = Environment.discover()
    except EnvironmentError as e:
        print_error(f"Environment Discovery Failed: {e}")
        raise typer.Exit(code=1)

    ws = env.workspace_path
    example = Environment.env_example_file(ws)
    target = Environment.env_file(ws)

    if not example.exists():
        print_error(f"No .env.example found in workspace")
        raise typer.Exit(code=1)

    content = example.read_text()
    print_simple_panel(content.strip(), title=f"Template: {example.name}")
    echo("")
    echo(f"To create your .env from this template, run:")
    echo(f"  cp {example} {target}")
    echo("")
    echo("Then edit the file and uncomment / set the values you need.")
    echo("The .env file is gitignored — it will not be committed.")
    if target.exists():
        echo("")
        print_warning(f"{target.name} already exists. Manual merge recommended.")


@project_app.command(
    name="design",
    short_help="Print the project-layer declaration ecosystem design.",
)
def design() -> None:
    """Show the design thinking behind MOSS's declaration ecosystem —
    Matrix / Mode manifests, config / provider / topic / signal / resource
    positioning, mode/ghost/network runtime axes. Points to codex commands
    for field-level details.
    """
    if not _DESIGN_MD.exists():
        print_error(f"DESIGN.md missing at {_DESIGN_MD}")
        raise typer.Exit(code=1)
    echo(_DESIGN_MD.read_text())


@project_app.command(
    name="overwrite-stubs",
    short_help="Overwrite .moss/ files with latest stub templates.",
)
def overwrite_stubs(
        yes: bool = typer.Option(
            False, "--yes", "-y",
            help="Skip confirmation prompt.",
        ),
) -> None:
    """Re-copy all stub template files into the current .moss workspace.

    This is useful when the installed ghoshell-moss package has updated stubs
    (new manifest directories, new default nuclei, etc.) and you want to pull
    those updates into an existing workspace.

    Works with git — after running, review and keep what you need:
      git diff .moss/          # see what changed
      git checkout .moss/      # revert unwanted overwrites
      git add .moss/ -p        # stage desired changes interactively
    """
    from ghoshell_moss.core.blueprint.environment import Environment

    try:
        env = Environment.discover()
    except EnvironmentError as e:
        print_error(f"Environment Discovery Failed: {e}")
        raise typer.Exit(code=1)

    ws_dir = env.workspace_path

    if not yes and not typer.confirm(
        f"Overwrite all stub files in '{ws_dir}' with latest templates?",
        default=True,
    ):
        print_warning("Aborted.")
        return

    Environment.init_workspace(ws_dir, force=True)
    print_success("Stub files overwritten.")
    echo("")
    print_info("Review changes with git:")
    print_info("  git diff .moss/")
    print_info("  git checkout .moss/   # revert unwanted overwrites")
    print_info("  git add .moss/ -p     # stage desired changes")
