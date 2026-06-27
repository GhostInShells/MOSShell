import typer
from pathlib import Path

modes_app = typer.Typer(
    help="MOSS mode management — list and inspect available runtime modes.",
    no_args_is_help=True,
)

from .utils import (
    print_simple_table,
    print_simple_panel,
    print_error,
    print_info,
)


def _get_project():
    from ghoshell_moss.core.blueprint.project import Project
    return Project.discover()


@modes_app.command(
    name="list",
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


@modes_app.command(
    name="show",
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
