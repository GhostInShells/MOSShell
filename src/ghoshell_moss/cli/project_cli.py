import os
import stat

import typer
from pathlib import Path

project_app = typer.Typer(
    help="MOSS Project tools — inspect workspace, modes, ghosts, and environment.",
    no_args_is_help=True,
)

from .utils import (
    print_simple_table,
    print_simple_panel,
    print_success,
    print_error,
    print_warning,
    print_info,
    echo,
)


# -- where -- #

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


# -- env-init -- #

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


# -- modes -- #

def _get_project():
    from ghoshell_moss.core.blueprint.project import Project
    return Project.discover()


@project_app.command(
    name="list-modes",
    short_help="List all discovered modes in the workspace.",
)
def list_modes() -> None:
    project = _get_project()
    modes = list(project.list_modes())
    if not modes:
        print_info("No modes found in this workspace.")
        return

    rows = []
    for _, manifest in sorted(modes, key=lambda x: x[1].name()):
        if manifest.is_error():
            rows.append([manifest.name(), "ERROR", str(manifest.error())[:80]])
        else:
            meta = manifest.value()
            rows.append([meta.name, meta.description or "—", meta.ctml_version])
    print_simple_table(
        data=rows,
        headers=["Name", "Description", "CTML Version"],
        title="Modes",
    )


@project_app.command(
    name="show-mode",
    short_help="Show detailed information for a specific mode.",
)
def show_mode(
        name: str = typer.Argument(..., help="Mode name"),
) -> None:
    project = _get_project()
    try:
        mode = project.get_mode(name)
    except LookupError as e:
        print_error(str(e))
        raise typer.Exit(code=1)

    meta = mode.meta
    print_simple_table(
        data=[
            ["Name", meta.name],
            ["Description", meta.description],
            ["CTML Version", meta.ctml_version],
            ["Home", str(mode.workspace_dir)],
            ["Manifest Package", meta.manifest_package],
            ["HOST.md", str(Path(meta.file))],
            ["Cell Paths", "\n".join(str(p) for p in mode.cells_discover_paths()) or "—"],
        ],
        headers=["Property", "Value"],
        title=f"Mode: {meta.name}",
    )
    if meta.system_prompt.strip():
        print_simple_panel(meta.system_prompt.strip(), title="Instruction")


# -- ghosts -- #

@project_app.command(
    name="list-ghosts",
    short_help="List all discovered ghosts in the workspace.",
)
def list_ghosts() -> None:
    project = _get_project()
    ghosts = list(project.ghosts())
    if not ghosts:
        print_info("No ghosts found in this workspace.")
        return

    rows = []
    for _, meta_or_err in ghosts:
        if isinstance(meta_or_err, Exception):
            # ghosts() yields (Path, GhostMeta | Exception), the path is lost here but ghosts() API doesn't expose name for errors cleanly
            rows.append(["<error>", "—", str(meta_or_err)[:80]])
        else:
            rows.append([meta_or_err.name(), meta_or_err.prototype(), meta_or_err.description()])
    print_simple_table(
        data=rows,
        headers=["Name", "Prototype", "Description"],
        title="Ghosts",
    )


@project_app.command(
    name="show-ghost",
    short_help="Show detailed information for a specific ghost.",
)
def show_ghost(
        name: str = typer.Argument(..., help="Ghost name"),
) -> None:
    project = _get_project()
    try:
        meta = project.get_ghost(name)
    except LookupError as e:
        print_error(str(e))
        raise typer.Exit(code=1)

    import inspect
    cls = type(meta)
    import_path = f"{cls.__module__}:{cls.__qualname__}"
    try:
        source_file = str(Path(inspect.getfile(cls)))
    except (TypeError, OSError):
        source_file = "—"

    print_simple_table(
        data=[
            ["Name", meta.name()],
            ["Prototype", meta.prototype()],
            ["Import Path", import_path],
            ["Source File", source_file],
            ["Version", meta.version() or "—"],
            ["Description", meta.description()],
        ],
        headers=["Property", "Value"],
        title=f"Ghost: {meta.name()}",
    )
