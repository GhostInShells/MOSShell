import json
import typer

networks_app = typer.Typer(
    help="Network configuration management — list and inspect available network drivers.",
    no_args_is_help=True,
)

from .utils import (
    print_simple_table, print_simple_panel,
    print_error, print_info,
)


def _get_project():
    from ghoshell_moss.core.blueprint.project import Project
    return Project.discover()


@networks_app.command(name="list")
def list_networks():
    """List all discovered network configurations."""
    project = _get_project()
    metas = project.network_metas()

    if not metas:
        print_info("No network configurations found.")
        print_info(f"Network configs are stored in: {project.network_configs_dir}")
        return

    rows = []
    for name, meta in sorted(metas.items()):
        rows.append([
            name,
            meta.driver,
            meta.scope,
            meta.description or "—",
        ])

    print_simple_table(
        data=rows,
        headers=["Name", "Driver", "Scope", "Description"],
        title="Networks",
    )


@networks_app.command(name="show")
def show_network(
    name: str = typer.Argument(..., help="Network name"),
):
    """Show detailed information for a specific network."""
    project = _get_project()
    metas = project.network_metas()

    if name not in metas:
        print_error(f"Network '{name}' not found.")
        available = ", ".join(sorted(metas.keys()))
        if available:
            print_info(f"Available: {available}")
        raise typer.Exit(code=1)

    meta = metas[name]
    print_simple_table(
        data=[
            ["Name", meta.name],
            ["Description", meta.description or "—"],
            ["Driver", meta.driver],
            ["Scope", meta.scope],
        ],
        headers=["Property", "Value"],
        title=f"Network: {meta.name}",
    )

    # driver-specific config
    if meta.config:
        config_json = json.dumps(meta.config, indent=2, ensure_ascii=False)
        print_simple_panel(config_json, title="Driver Config")
