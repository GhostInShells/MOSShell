"""
moss cells — discover, launch, and manage runtime cells.

Unified replacement for old moss apps / moss script / moss runtime.
Static discovery via CellRegistry (CELL.md scanning), lifecycle via
CellRegistry.spawn_cell(), runtime governance via runtime files.

Mode-aware: cell discovery paths come from the active mode's HOST.md
when a mode is active, falling back to MOSS.md defaults.
"""

import shlex
from pathlib import Path
from importlib import resources

import typer

from ghoshell_moss.core.blueprint.project import Project, HostMode
from ghoshell_moss.core.blueprint.cell import CellRegistry, CellManifest

from .utils import (
    print_simple_table, print_simple_panel,
    print_error, print_warning, print_info, print_success, echo,
)

cells_app = typer.Typer(
    help="Discover, launch, and manage MOSS cells.",
    no_args_is_help=True,
)

# stub package for moss cells create
_CELL_STUB_PACKAGE = 'ghoshell_moss.stubs.cell'

# ---------------------------------------------------------------------------
# context helpers
# ---------------------------------------------------------------------------

_CellsContext = tuple[
    Project,              # project
    HostMode | None,      # current mode (None if no_mode)
    CellRegistry,         # cell registry (mode-aware)
]


def _get_context() -> _CellsContext:
    """Resolve Project + CellRegistry context.

    CellRegistry is mode-aware: uses mode's cells_discover_paths
    when a mode is active, falls back to env.cell_dirs().
    """
    project = Project.discover()

    try:
        mode = project.current_mode()
    except Exception:
        mode = None

    cells = project.cells
    return project, mode, cells


def _display_no_mode_hint() -> None:
    print_info(
        "No mode is active. Cell discovery uses project defaults (MOSS.md). "
        "Use 'moss project list-modes' to see available modes."
    )


# ---------------------------------------------------------------------------
# specification — cognitive entry for cell development
# ---------------------------------------------------------------------------

_CELLS_SPECIFICATION = """
# MOSS Cells — Runtime Cell Specification

## What is a Cell

A cell is any participant with an address on the Matrix network.
One concept replaces three old abstractions: app / script / runtime node.
All cells are `CellType.worker`. What differs is the launcher configuration.

The CELL.md file is frontmatter + body (instruction).  The full data model:

    moss codex get-source ghoshell_moss.core.blueprint.cell

## Five Categories (launcher configuration, not type)

All cells are `CellType.worker`. The five categories differ only in launcher
and dependency isolation.  Choose one, then set CELL.md fields accordingly.

### standalone — zero deps, sys.executable
Self-contained Python.  `interpreter: python`, `cmd: main.py`.

### project — depends on the project
Imports from `MOSS.manifests`, shared configs, etc.
`interpreter: python`, `cmd: main.py`.  Same launcher as standalone;
the difference is what the code imports.

### isolated — independent venv
Own `pyproject.toml` + `.venv`.  Minimal dep: `ghoshell_moss[matrix]`.
`interpreter: .venv/bin/python`, `cmd: main.py`.

### script — pure shell, no ghoshell
`interpreter: /bin/bash`, `cmd: run.sh`.
OS subprocess; no Python, no Matrix SDK.

### remote — independent .moss workspace
Not discovered via CELL.md scanning.  Registered by network/scope alignment
on another machine.  Connects to the same Matrix network.

## Discovery

Cell discovery paths come from the active mode's HOST.md
(`cell_paths` + `exclude_cell_paths`), falling back to MOSS.md defaults.
One CELL.md per directory; a cell directory is a closed unit (no nesting).

## The Model's View

When a cell is running and provides a channel, the host creates a channel
proxy.  The model sees the cell's instruction body and its channel commands
(code as prompt — the Python function signatures ARE the interface).

The cell's address is `type/name` (singleton) or `type/name/uid` (non-singleton).
This is how the cell is identified on the Matrix network.

## Quick Start

1. moss cells create my-cell
2. Edit CELL.md — set name, description, launcher
3. Write the instruction body — what the model should know
4. moss cells run my-cell

## Further Reading

    moss codex blueprint matrix             — Matrix, CellNetwork, spawn
    moss codex blueprint channel_builder    — how to build a channel inside a cell
    moss codex get-source ghoshell_moss.core.blueprint.states_channel  — stateful channels (complex cells only)
    moss cells --help                       — full CLI surface
""".strip()


@cells_app.command(name="specification")
def show_specification():
    """Cell development guide — CELL.md format, five categories, CLI reference."""
    print_simple_panel(_CELLS_SPECIFICATION, title="Cells Specification")


# ---------------------------------------------------------------------------
# create — generate a cell from stub template
# ---------------------------------------------------------------------------

@cells_app.command(name="create")
def create_cell(
    name: str = typer.Argument(help="Cell name (kebab-case, e.g. 'web-fetch')."),
    group: str = typer.Option(
        "", "--group", "-g",
        help="Cell group (e.g. 'tools', 'sensors'). Creates cells/{group}/{name}/.",
    ),
):
    """Create a new cell from the stub template.

    Generates CELL.md, main.py, README.md, and INSTALL.md under
    {workspace}/cells/{group}/{name}/.
    """
    project, mode, cells = _get_context()

    # target directory
    target_dir = project.workspace_dir / "cells"
    if group:
        target_dir = target_dir / group
    target_dir = target_dir / name

    if target_dir.exists():
        print_error(f"Cell directory already exists: {target_dir}")
        print_info("Remove it first or use a different name.")
        raise typer.Exit(code=1)

    target_dir.mkdir(parents=True, exist_ok=False)

    # copy stub files
    stub_resources = resources.files(_CELL_STUB_PACKAGE)
    _copy_stub(stub_resources, target_dir, name=name, group=group)

    print_success(f"Cell '{name}' created at {target_dir}")
    echo("")
    print_info("Next steps:")
    print_info(f"  1. Edit {target_dir / 'CELL.md'} — fill in name, description, launcher")
    print_info(f"  2. Write the instruction body — what the model should know")
    print_info(f"  3. moss cells run {target_dir}  — test launch")
    print_info(f"  moss cells specification  — full development guide")


def _copy_stub(stub_node, target_dir: Path, *, name: str, group: str) -> None:
    """Copy stub files into target_dir, replacing placeholders."""
    for item in stub_node.iterdir():
        if item.name == "__init__.py":
            continue
        target_item = target_dir / item.name
        if item.is_dir():
            target_item.mkdir(exist_ok=True)
            _copy_stub(item, target_item, name=name, group=group)
        else:
            content = item.read_text()
            content = content.replace("{name}", name)
            content = content.replace("{group}", group or "")
            target_item.write_text(content)


# ---------------------------------------------------------------------------
# register — create a CELL.md shortcut for an external script
# ---------------------------------------------------------------------------

@cells_app.command(name="register")
def register_cell(
    file: Path = typer.Argument(
        help="Path to the script file (absolute or relative to cwd).",
        exists=True, file_okay=True, dir_okay=False,
    ),
    name: str = typer.Option(
        "", "--name", "-n",
        help="Cell name. Defaults to the script filename (without extension).",
    ),
    group: str = typer.Option(
        "", "--group", "-g",
        help="Cell group for discovery path.",
    ),
):
    """Register an external script as a cell (creates a CELL.md shortcut).

    Creates a CELL.md in {workspace}/cells/{group}/{name}/ whose
    launcher.cmd points to the given script file.
    """
    project, mode, cells = _get_context()

    resolved = file.resolve()
    name = name or resolved.stem

    target_dir = project.workspace_dir / "cells"
    if group:
        target_dir = target_dir / group
    target_dir = target_dir / name

    if target_dir.exists():
        print_error(f"Cell directory already exists: {target_dir}")
        raise typer.Exit(code=1)

    target_dir.mkdir(parents=True, exist_ok=False)

    manifest = CellManifest(
        name=name,
        type="worker",
        singleton=False,
        description=f"Shortcut to {resolved}",
        launcher={
            "interpreter": "",
            "cmd": str(resolved),
            "arguments": "",
            "cwd": str(resolved.parent),
        },
        instruction=f"Registered script: {resolved}\n\nmoss cells specification — cell development guide.",
        installed=True,
    )
    manifest.write_file(target_dir)

    print_success(f"Cell '{name}' registered at {target_dir}")
    print_info(f"  Launcher: cmd={resolved}")
    print_info(f"  moss cells run {target_dir}  — test launch")


# ---------------------------------------------------------------------------
# list — scan CELL.md manifests
# ---------------------------------------------------------------------------

@cells_app.command(name="list")
def list_cells(
    installed: bool = typer.Option(
        False, "--installed",
        help="Only show cells marked as installed.",
    ),
    include: list[str] | None = typer.Option(
        None, "--include",
        help="fnmatch pattern to include (repeatable).",
    ),
    exclude: list[str] | None = typer.Option(
        None, "--exclude",
        help="fnmatch pattern to exclude (repeatable).",
    ),
):
    """List discovered cell manifests (CELL.md scanning). Mode-aware."""
    project, mode, cells = _get_context()

    mode_name = mode.name if mode else "none"
    discovery_paths = mode.cells_discover_paths() if mode else project.env.cell_dirs()
    path_strs = [str(p.relative_to(project.root)) for p in discovery_paths]

    echo("")
    print_simple_table(
        data=[
            ["mode", mode_name],
            ["discovery paths", ", ".join(path_strs)],
        ],
        headers=["Context", "Value"],
        title="Cells Context",
    )
    if mode is None:
        _display_no_mode_hint()

    manifests = cells.list_cell_manifests(
        refresh=True,
        installed=installed,
        include=include or None,
        exclude=exclude or None,
    )

    rows: list[list[str]] = []
    for rel_path, m in manifests.items():
        label = m.name
        suffix = ""
        if not m.installed:
            suffix += " [not installed]"
        cell_type = m.type if isinstance(m.type, str) else m.type.value
        if cell_type != "worker":
            suffix += f" (type={cell_type})"
        rows.append([
            label + suffix,
            str(rel_path),
            (m.description or "")[:100],
        ])

    _display_cells_table(
        rows=rows,
        headers=["Name", "Path", "Description"],
        title=f"Cells ({len(rows)} found)",
    )


def _display_cells_table(
    rows: list[list[str]],
    headers: list[str],
    title: str,
) -> None:
    if not rows:
        print_warning("No cells found.")
        return
    echo("")
    print_simple_table(data=rows, headers=headers, title=title)


# ---------------------------------------------------------------------------
# show — view a single cell's manifest
# ---------------------------------------------------------------------------

@cells_app.command(name="show")
def show_cell(
    path: str = typer.Argument(help="Project-relative path to the cell directory."),
):
    """Show a cell's full manifest and instruction."""
    project, mode, cells = _get_context()

    manifest = cells.get_cell_manifest(path)
    if manifest is None:
        print_error(f"No CELL.md found at '{path}'.")
        print_info("Use 'moss cells list' to see discovered cells.")
        raise typer.Exit(code=1)

    echo("")
    print_simple_table(
        data=[
            ["name", manifest.name],
            ["type", str(manifest.type)],
            ["singleton", str(manifest.singleton)],
            ["installed", str(manifest.installed)],
            ["description", manifest.description or "—"],
            ["path", str(path)],
            ["interpreter", manifest.launcher.interpreter or "sys.executable"],
            ["cmd", manifest.launcher.cmd or "—"],
            ["arguments", manifest.launcher.arguments or "—"],
            ["cwd", manifest.launcher.cwd or "(auto)"],
        ],
        headers=["Property", "Value"],
        title=f"Cell: {manifest.name}",
    )

    if manifest.instruction:
        echo("")
        print_simple_panel(manifest.instruction.strip(), title="Instruction")

    echo("")
    print_info(
        f"moss cells specification — cell development guide. "
        f"Edit CELL.md at: {project.root / path / 'CELL.md'}"
    )


# ---------------------------------------------------------------------------
# run — launch a cell (dir / script / name modes)
# ---------------------------------------------------------------------------

@cells_app.command(name="run")
def run_cell(
    target: str = typer.Argument(help="Cell target: directory path, script file, or bare name."),
    args: list[str] | None = typer.Argument(
        None, help="Arguments passed through to the cell process.",
    ),
):
    """Launch a cell process. Three modes:

    Directory mode:  moss cells run ./cells/tools/web-fetch/ --flag
    Script mode:     moss cells run ./cells/tools/web-fetch/scripts/debug.py
    Name mode:       moss cells run web-fetch
    """
    project, mode, cells = _get_context()
    target_path = Path(target)

    # --- resolve: directory / script file / bare name ---
    if target_path.is_dir():
        _run_by_directory(project, cells, target_path, args or [])
    elif target_path.is_file():
        _run_by_script(project, cells, target_path, args or [])
    else:
        _run_by_name(project, cells, target, args or [])


def _run_by_directory(
    project: Project,
    cells: CellRegistry,
    target_dir: Path,
    args: list[str],
) -> None:
    """Directory mode: find CELL.md, read launcher, spawn."""
    manifest = CellManifest.read_from_directory(target_dir)
    if manifest is None:
        print_error(f"No CELL.md found in '{target_dir}'.")
        print_info("Use 'moss cells create <name>' to create one, or check the path.")
        raise typer.Exit(code=1)

    _launch_cell(project, cells, manifest, extra_args=args)


def _run_by_script(
    project: Project,
    cells: CellRegistry,
    script_file: Path,
    args: list[str],
) -> None:
    """Script mode: determine interpreter, upward-find CELL.md, spawn."""
    resolved = script_file.resolve()

    # determine interpreter
    suffix = resolved.suffix
    if suffix == ".py":
        import sys
        interpreter = sys.executable
    elif suffix == ".sh":
        interpreter = "/bin/bash"
    else:
        interpreter = ""

    # upward search for CELL.md (max 3 levels) for name/group context
    search_dir = resolved.parent
    manifest = None
    for _ in range(3):
        manifest = CellManifest.read_from_directory(search_dir)
        if manifest is not None:
            break
        search_dir = search_dir.parent

    if manifest is not None:
        # use CELL.md context but override launcher
        manifest.launcher.interpreter = interpreter
        manifest.launcher.cmd = str(resolved)
        manifest.launcher.cwd = str(resolved.parent)
    else:
        # no CELL.md found — create a minimal manifest
        manifest = CellManifest(
            name=resolved.stem,
            type="worker",
            singleton=False,
            description=f"Script: {resolved.name}",
            launcher={
                "interpreter": interpreter,
                "cmd": str(resolved),
                "arguments": "",
                "cwd": str(resolved.parent),
            },
            instruction="",
            installed=True,
        )

    _launch_cell(project, cells, manifest, extra_args=args)


def _run_by_name(
    project: Project,
    cells: CellRegistry,
    name: str,
    args: list[str],
) -> None:
    """Name mode: scan all manifests, match by CELL.md name field."""
    manifests = cells.list_cell_manifests(refresh=True, installed=False)

    matches = [m for _, m in manifests.items() if m.name == name]
    if not matches:
        print_error(f"No cell found with name '{name}'.")
        print_info("Use 'moss cells list' to see available cells.")
        raise typer.Exit(code=1)

    if len(matches) > 1:
        print_warning(f"Multiple cells match name '{name}':")
        for m in matches:
            print_info(f"  {m.name}  ({m.description or '—'})")
        print_info("Use directory path to disambiguate.")
        raise typer.Exit(code=1)

    _launch_cell(project, cells, matches[0], extra_args=args)


def _launch_cell(
    project: Project,
    cells: CellRegistry,
    manifest: CellManifest,
    *,
    extra_args: list[str] | None = None,
) -> None:
    """Resolve manifest, show pre-launch info, spawn."""
    import asyncio

    # show pre-launch info
    echo("")
    print_simple_table(
        data=[
            ["name", manifest.name],
            ["type", str(manifest.type)],
            ["interpreter", manifest.launcher.interpreter or "sys.executable"],
            ["cmd", manifest.launcher.cmd or "—"],
            ["cwd", manifest.launcher.cwd or "(cwd)"],
            ["installed", str(manifest.installed)],
        ],
        headers=["Property", "Value"],
        title="Launching Cell",
    )

    if not manifest.installed:
        print_warning(
            "Cell is not installed. If INSTALL.md exists, read it and "
            "run the install steps before launching."
        )

    # merge extra_args into launcher arguments
    if extra_args:
        manifest = manifest.model_copy()
        existing = shlex.split(manifest.launcher.arguments) if manifest.launcher.arguments else []
        manifest.launcher.arguments = shlex.join(existing + extra_args)

    async def _spawn():
        proc = await cells.spawn_cell(
            manifest,
            stdout=None,   # inherit terminal
            stderr=None,
        )
        print_success(f"Cell '{manifest.name}' started (pid={proc.pid}).")
        return proc

    try:
        asyncio.run(_spawn())
    except Exception as e:
        print_error(f"Failed to launch cell: {e}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# install — mark a cell as installed
# ---------------------------------------------------------------------------

@cells_app.command(name="install")
def install_cell(
    name: str = typer.Argument(help="Cell name (matches CELL.md name field)."),
):
    """Mark a cell as installed (creates the .installed marker file).

    This does NOT run any install script.  If the cell has an INSTALL.md,
    read it and run the steps yourself, then call this command.
    """
    project, mode, cells = _get_context()

    manifests = cells.list_cell_manifests(refresh=True, installed=False)
    matches = [(rp, m) for rp, m in manifests.items() if m.name == name]

    if not matches:
        print_error(f"No cell found with name '{name}'.")
        raise typer.Exit(code=1)

    rel_path, manifest = matches[0]
    cell_dir = project.root / rel_path

    install_md = cell_dir / CellManifest.INSTALL_FILENAME
    if not install_md.exists():
        print_warning(f"No INSTALL.md found in '{cell_dir}'.")
        print_info("This cell requires no installation. Nothing to do.")
        return

    installed_file = cell_dir / CellManifest.INSTALLED_FILE
    installed_file.touch()
    print_success(f"Cell '{name}' marked as installed ({installed_file}).")
    echo("")
    print_info("The cell will now appear in 'moss cells list --installed'.")


# ---------------------------------------------------------------------------
# status — runtime status of cells
# ---------------------------------------------------------------------------

@cells_app.command(name="status")
def status_cells(
    address: str = typer.Argument(
        "", help="Cell address to inspect (type/name/uid). Omit to list all.",
    ),
):
    """Show runtime status of cells. Without argument, lists all running cells."""
    project, mode, cells = _get_context()

    if address:
        _show_cell_runtime_detail(cells, address)
    else:
        _list_running_cells(cells)


def _list_running_cells(cells: CellRegistry) -> None:
    """List all cells with runtime files, showing alive/stopped."""
    runtime_cells = cells.local_runtime_cells()
    if not runtime_cells:
        print_info("No cell runtime files found.")
        return

    rows: list[list[str]] = []
    for cell in runtime_cells:
        alive = "alive" if cell.is_alive() else "stopped"
        rows.append([
            cell.address,
            cell.meta.name,
            str(cell.status.pid) if cell.status.pid else "—",
            alive,
            cell.status.failure[:60] if cell.status.failure else "—",
        ])

    _display_cells_table(
        rows=rows,
        headers=["Address", "Name", "PID", "State", "Failure"],
        title=f"Runtime Cells ({len(rows)} found)",
    )


def _show_cell_runtime_detail(cells: CellRegistry, address: str) -> None:
    """Show detailed runtime info for a specific cell by address."""
    runtime_cells = cells.local_runtime_cells()
    matched = None
    for cell in runtime_cells:
        if cell.address == address or cell.address.startswith(address):
            matched = cell
            break

    if matched is None:
        print_error(f"No runtime cell found for address '{address}'.")
        return

    alive = "alive" if matched.is_alive() else "stopped"
    echo("")
    print_simple_table(
        data=[
            ["address", matched.address],
            ["name", matched.meta.name],
            ["type", str(matched.meta.type)],
            ["state", alive],
            ["pid", str(matched.status.pid)],
            ["uid", matched.status.uid],
            ["project_id", matched.status.project_id],
            ["failure", matched.status.failure or "—"],
            ["stdout_log", matched.status.stdout_log or "—"],
            ["stderr_log", matched.status.stderr_log or "—"],
        ],
        headers=["Property", "Value"],
        title=f"Runtime Cell: {matched.meta.name}",
    )


# ---------------------------------------------------------------------------
# kill / kill-all — force-kill running cells
# ---------------------------------------------------------------------------

@cells_app.command(name="kill")
def kill_cell(
    address: str = typer.Argument(help="Cell address to kill (e.g. 'worker/my-cell/uid')."),
):
    """Force-kill a running cell and remove its runtime file."""
    project, mode, cells = _get_context()

    runtime_cells = cells.local_runtime_cells()
    matched = None
    for cell in runtime_cells:
        if cell.address == address or cell.address.startswith(address):
            matched = cell
            break

    if matched is None:
        print_error(f"No runtime cell found for address '{address}'.")
        raise typer.Exit(code=1)

    if matched.status.pid > 0:
        cells.recursively_kill_process(matched.status.pid)

    runtime_dir = cells.cell_runtimes_dir
    file = matched.runtime_filepath(runtime_dir)
    if file.exists():
        file.unlink()

    print_success(f"Cell '{matched.address}' killed and removed from runtime.")


@cells_app.command(name="kill-all")
def kill_all_cells():
    """Force-kill all running cells and clear runtime files."""
    project, mode, cells = _get_context()

    runtime_cells = cells.local_runtime_cells()
    if not runtime_cells:
        print_info("No cell runtime files to clean up.")
        return

    count = len(runtime_cells)
    cells.kill_all_runtime_cells()
    print_success(f"Killed {count} cell(s) and cleared runtime files.")
