import typer
from pathlib import Path

ghosts_app = typer.Typer(
    help="MOSS ghost management — list and inspect available ghosts.",
    no_args_is_help=True,
)

from .utils import (
    print_simple_table,
    print_error,
    print_info,
)


def _get_project():
    from ghoshell_moss.core.blueprint.project import Project
    return Project.discover()


@ghosts_app.command(
    name="list",
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
            rows.append(["<error>", "—", str(meta_or_err)[:80]])
        else:
            rows.append([meta_or_err.name(), meta_or_err.prototype(), meta_or_err.description()])
    print_simple_table(
        data=rows,
        headers=["Name", "Prototype", "Description"],
        title="Ghosts",
    )


@ghosts_app.command(
    name="show",
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
