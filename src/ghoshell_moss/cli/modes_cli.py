import re
from importlib import resources
from pathlib import Path

import typer

modes_app = typer.Typer(
    help="MOSS mode management — list and inspect available runtime modes.",
    no_args_is_help=True,
)

from .utils import (
    print_simple_table,
    print_simple_panel,
    print_error,
    print_info,
    print_success,
    echo,
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
            ["Node Paths", "\n".join(str(p) for p in mode.nodes_discover_paths()) or "—"],
        ],
        headers=["Property", "Value"],
        title=f"Mode: {meta.name}",
    )
    if meta.system_prompt.strip():
        print_simple_panel(meta.system_prompt.strip(), title="Instruction")


@modes_app.command(
    name="create",
    short_help="Create a new mode from the default stub template.",
)
def create_mode(
        name: str = typer.Argument(..., help="Mode name (identifier)."),
) -> None:
    """Create a new mode by copying the default stub to .moss/modes/<name>/.

    After creation, edit HOST.md to configure node_paths, description, and other
    mode-level settings. The mode stub provides HOST.md, src/HOST/ scaffold, and
    a runtime/.gitignore.
    """
    from ghoshell_moss.core.blueprint.environment import (
        MOSS_NAME_PATTERN, MODE_STUB_PACKAGE,
    )

    if not re.match(MOSS_NAME_PATTERN, name):
        print_error(f"Invalid mode name '{name}'. Must match {MOSS_NAME_PATTERN}")
        raise typer.Exit(code=1)

    project = _get_project()
    target_dir = project.get_mode_home(name)

    if target_dir.exists():
        print_error(f"Mode directory already exists: {target_dir}")
        raise typer.Exit(code=1)

    target_dir.mkdir(parents=True, exist_ok=False)
    stub_resources = resources.files(MODE_STUB_PACKAGE)
    _copy_stub(stub_resources, target_dir, name=name)

    host_md = target_dir / "HOST.md"
    print_success(f"Mode '{name}' created at {target_dir}")
    echo("")
    print_info(f"  Edit {host_md}  — description, node_paths, ctml_version.")
    print_info(f"  Show: moss modes show {name}")


def _copy_stub(stub_node, target_dir: Path, *, name: str) -> None:
    """Copy stub files into target_dir, replacing {name} placeholders."""
    for item in stub_node.iterdir():
        if item.name == "__init__.py":
            continue
        target_item = target_dir / item.name
        if item.is_dir():
            target_item.mkdir(exist_ok=True)
            _copy_stub(item, target_item, name=name)
        else:
            content = item.read_bytes()
            try:
                text = content.decode('utf-8')
                text = text.replace("{name}", name)
                target_item.write_text(text)
            except UnicodeDecodeError:
                target_item.write_bytes(content)
