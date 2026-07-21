import typer

from .utils import (
    print_simple_table, print_info, print_error, echo,
)
from ghoshell_moss.core.blueprint.project import Project
from ghoshell_moss.core.ctml.versions import CTML_VERSION, default_moss_ctml_meta_instruction_directory

ctml_app = typer.Typer(
    help="CTML version management — list and read available CTML prompts.",
    no_args_is_help=True,
)


@ctml_app.command(name="list")
def list_ctml_versions():
    """List all available CTML versions discovered in this project."""
    try:
        project = Project.discover()
    except Exception as e:
        print_error(f"Project discovery failed: {e}")
        raise typer.Exit(code=1)

    versions = project.ctml_versions()
    if not versions:
        print_info("No CTML versions found in this project.")
        return

    # 当前 mode 的默认版本
    try:
        mode = project.current_mode()
        default_version = mode.meta.ctml_version if mode else CTML_VERSION
    except Exception:
        default_version = CTML_VERSION

    builtin_dir = default_moss_ctml_meta_instruction_directory()
    rows = []
    for version, path in sorted(versions.items()):
        is_builtin = str(path).startswith(str(builtin_dir))
        source_label = "Built-in" if is_builtin else "Workspace"
        rows.append([version, str(path.absolute()), source_label])

    print_simple_table(
        data=rows,
        headers=["Version", "Location", "Source"],
        title=f"CTML Versions  (default: {default_version})",
    )


@ctml_app.command(name="read")
def read_ctml_version(
    version: str = typer.Argument(
        CTML_VERSION,
        help=f"CTML version name (default: {CTML_VERSION})",
    ),
    raw: bool = typer.Option(False, "--raw", help="Output raw content without formatting."),
):
    """Read the content of a specific CTML version."""
    try:
        project = Project.discover()
    except Exception as e:
        print_error(f"Project discovery failed: {e}")
        raise typer.Exit(code=1)

    versions = project.ctml_versions()
    if version not in versions:
        print_error(f"CTML version '{version}' not found.")
        available = ", ".join(sorted(versions.keys()))
        if available:
            print_info(f"Available: {available}")
        raise typer.Exit(code=1)

    file_path = versions[version]
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print_error(f"Error reading file: {e}")
        raise typer.Exit(code=1)

    if raw:
        echo(content)
    else:
        print_info(f"Source: {file_path.absolute()}")
        echo("")
        echo(content)
