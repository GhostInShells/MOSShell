"""
moss manifests — workspace static declarations.

Queries the manifest system via Project (not Host): MatrixManifest for global
baseline (MOSS.manifests) and ModeManifests for the active mode's effective
view (HOST, which extends MOSS.manifests via Python import).

Each command shows two tables when a mode is active; a single Matrix table
plus a hint when no mode is active.
"""

import json
import inspect as _inspect
from pathlib import Path
from typing import Iterable

import typer

from ghoshell_container import Provider
from ghoshell_moss.core.blueprint.project import Manifest, Project, MatrixManifest, ModeManifests, HostMode
from ghoshell_moss.project.manifests.impl import ScannedMatrixManifest

from .utils import (
    print_simple_table, print_simple_panel,
    print_error, print_warning, print_info, echo,
)

manifest_app = typer.Typer(
    help="Inspect capability declarations: providers, configs, topics, signals, parameters, resources, channel, nuclei.",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# context helpers
# ---------------------------------------------------------------------------

_Context = tuple[
    Project,          # project
    HostMode | None,  # current mode (None if no_mode)
    MatrixManifest,   # global (MOSS.manifests)
    ModeManifests | None,  # mode effective view (None if no mode)
]


def _get_context() -> _Context:
    """Resolve the current Project and manifests context.

    Returns (project, mode, matrix_manifests, mode_manifests).
    mode / mode_manifests are None when no mode is active.
    """
    project = Project.discover()

    try:
        mode = project.current_mode()
    except Exception:
        mode = None

    matrix_mf = ScannedMatrixManifest()

    mode_mf = None
    if mode is not None:
        try:
            mode.bootstrap()
            mode_mf = mode.manifests()
        except Exception:
            mode_mf = None

    return project, mode, matrix_mf, mode_mf


def _display_context_header(project: Project, mode: HostMode | None) -> None:
    """Show the active context: mode, network, ghost, scope."""
    env = project.env
    mode_name = mode.name if mode else "none"
    mode_source = env.moss_meta.default_mode if env.moss_meta.default_mode else "—"
    ghost_name = env.ghost_name or env.moss_meta.default_ghost or "—"
    network_name = env.network or "default"
    scope = env.network_scope

    rows = [
        ["Mode", mode_name + (f"  (from MOSS.md default: {mode_source})" if mode_name != "none" else "")],
        ["Network", f"{network_name}  /  scope: {scope}"],
        ["Ghost", ghost_name],
    ]
    print_simple_table(
        data=rows,
        headers=["Context", "Value"],
        title="Active Context",
    )


def _display_no_mode_hint() -> None:
    print_warning("No mode is active. Mode-level declarations are unavailable.")
    print_info("Use 'moss modes list' to see available modes.")


def _display_no_matrix_items(type_label: str, matrix_pkg: str = "MOSS.manifests") -> None:
    print_warning(f"No {type_label} found in Matrix ({matrix_pkg}.{type_label}).")


def _display_no_mode_items(type_label: str, mode_name: str, mode_pkg: str = "HOST") -> None:
    print_warning(f"No {type_label} found in mode '{mode_name}' ({mode_pkg}.{type_label}).")


# ---------------------------------------------------------------------------
# table helpers
# ---------------------------------------------------------------------------

def _collect_table_rows(
    manifests: Iterable[Manifest],
    columns: list[str],
) -> list[list[str]]:
    """Convert Manifest[T] items to table rows based on column spec.

    columns: list of attribute keys. Supported:
      'name', 'description', 'found_at', 'import_path',
      'type' (provider singleton/factory),
      'scheme' (resource scheme),
      'host' (resource host),
      'signals' (nucleus signal names),
    """
    rows = []
    for m in manifests:
        if m.is_error():
            rows.append([m.name(), "ERROR", str(m.error())[:80], str(m.found_at())])
            continue
        row = []
        for col in columns:
            if col == 'name':
                row.append(m.name())
            elif col == 'description':
                row.append((m.description() or '')[:120])
            elif col == 'found_at':
                row.append(str(m.found_at()))
            elif col == 'import_path':
                row.append(m.import_path() or '')
            elif col == 'type':
                # provider specific
                v = m.value()
                row.append("Singleton" if v.singleton() else "Factory")
            elif col == 'topic_type':
                v = m.value()
                row.append(getattr(v, 'topic_type', ''))
            elif col == 'scheme':
                v = m.value()
                row.append(v.scheme())
            elif col == 'host':
                v = m.value()
                row.append(v.host)
            elif col == 'signals':
                v = m.value()
                names = [s.signal_name() for s in v.signals()]
                row.append(", ".join(names))
            else:
                row.append('')
        rows.append(row)
    return rows


def _display_manifest_list(
    manifests: Iterable[Manifest],
    headers: list[str],
    columns: list[str],
    title: str,
) -> int:
    """Display a manifest list as a table. Returns count of displayed items."""
    items = list(manifests)
    if not items:
        return 0
    rows = _collect_table_rows(items, columns)
    print_simple_table(data=rows, headers=headers, title=title)
    return len(items)


def _display_two_layer(
    matrix_items: Iterable[Manifest],
    matrix_mf: MatrixManifest,
    mode_items: Iterable[Manifest] | None,
    mode_mf: ModeManifests | None,
    headers: list[str],
    columns: list[str],
    type_label: str,
    mode: HostMode | None,
) -> None:
    """Display Matrix + Mode tables, handling edge cases."""
    matrix_pkg = matrix_mf.root_package()
    matrix_title = f"Matrix ({matrix_pkg}.{type_label})"

    matrix_count = _display_manifest_list(
        matrix_items, headers, columns,
        title=matrix_title,
    )
    if matrix_count == 0:
        _display_no_matrix_items(type_label, matrix_pkg)

    if mode is not None and mode_items is not None and mode_mf is not None:
        echo("")
        mode_pkg = mode_mf.root_package()
        mode_title = f"Mode: {mode.name} ({mode_pkg}.{type_label})"
        mode_count = _display_manifest_list(
            mode_items, headers, columns,
            title=mode_title,
        )
        if mode_count == 0:
            _display_no_mode_items(type_label, mode.name, mode_pkg)
        else:
            echo("")
            print_info(
                f"{matrix_count} from {matrix_pkg}, {mode_count} effective in mode "
                f"({mode_pkg} extends {matrix_pkg} via import)"
            )
    elif mode is None:
        _display_no_mode_hint()


def _filter_manifests(
    manifests: Iterable[Manifest],
    search: str,
) -> list[Manifest]:
    """Filter manifests by name or description containing search string."""
    s = search.lower()
    return [m for m in manifests if s in m.name().lower() or s in m.description().lower()]


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------

@manifest_app.command(name="providers")
def list_providers(
    search: str = typer.Argument(
        "", help="Search pattern for contract import path or provider type.",
    ),
):
    """List IoC providers discovered from manifests."""
    project, mode, matrix_mf, mode_mf = _get_context()
    _display_context_header(project, mode)

    matrix_raw = matrix_mf.providers()
    mode_raw = mode_mf.providers() if mode_mf else None

    if search:
        matrix_raw = _filter_manifests(matrix_raw, search)
        mode_raw = _filter_manifests(mode_raw, search) if mode_raw else None
        # single match in mode → detail
        mode_list = list(mode_raw) if mode_raw else []
        matrix_list = list(matrix_raw)
        if len(mode_list) == 1 and len(matrix_list) <= 1:
            _display_provider_detail(mode_list[0])
            return
        if len(matrix_list) == 1 and not mode_list:
            _display_provider_detail(matrix_list[0])
            return
        if not matrix_list and not mode_list:
            print_warning(f"No providers matching '{search}'.")
            return
        matrix_raw = matrix_list
        mode_raw = mode_list

    _display_two_layer(
        matrix_raw, matrix_mf,
        mode_raw, mode_mf,
        headers=["Contract", "Type", "Found At"],
        columns=["name", "type", "found_at"],
        type_label="providers",
        mode=mode,
    )


def _display_provider_detail(manifest: Manifest[Provider]) -> None:
    """Show a single provider in detail: contract source code."""
    if manifest.is_error():
        print_error(f"Provider scan error: {manifest.error()}")
        return
    provider = manifest.value()
    contract_type = provider.contract()

    echo("")
    print_simple_table(
        data=[
            ["Contract", manifest.name()],
            ["Type", "Singleton" if provider.singleton() else "Factory"],
            ["Found At", str(manifest.found_at())],
            ["Import Path", manifest.import_path() or "—"],
            ["Docstring", (manifest.description() or "—")[:200]],
        ],
        headers=["Property", "Value"],
        title="Provider Detail",
    )

    # contract source
    try:
        source = _inspect.getsource(contract_type)
        echo("")
        print_simple_panel(source, title="Contract Source")
    except (TypeError, OSError):
        print_info("Source unavailable (compiled or built-in contract).")


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------

@manifest_app.command(name="configs")
def list_configs(
    search: str = typer.Argument("", help="Search pattern for config name."),
    detail: bool = typer.Option(False, "--detail", "-d", help="Show full schema and defaults."),
):
    """List configuration models discovered from manifests."""
    project, mode, matrix_mf, mode_mf = _get_context()
    _display_context_header(project, mode)

    matrix_raw = matrix_mf.configs()
    mode_raw = mode_mf.configs() if mode_mf else None

    if search:
        matrix_raw = _filter_manifests(matrix_raw, search)
        mode_raw = _filter_manifests(mode_raw, search) if mode_raw else None
        mode_list = list(mode_raw) if mode_raw else []
        matrix_list = list(matrix_raw)
        if (len(mode_list) == 1 and len(matrix_list) <= 1) or (len(matrix_list) == 1 and not mode_list):
            target = mode_list[0] if mode_list else matrix_list[0]
            if detail or search:
                _display_config_detail(target)
                return
        if not matrix_list and not mode_list:
            print_warning(f"No configs matching '{search}'.")
            return
        matrix_raw = matrix_list
        mode_raw = mode_list

    if detail and mode_raw:
        mode_list = list(mode_raw)
        if len(mode_list) == 1:
            _display_config_detail(mode_list[0])
            return

    _display_two_layer(
        matrix_raw, matrix_mf,
        mode_raw, mode_mf,
        headers=["Name", "Module Path", "Description"],
        columns=["name", "import_path", "description"],
        type_label="configs",
        mode=mode,
    )


def _display_config_detail(manifest: Manifest) -> None:
    """Show a single config: YAML defaults, JSON Schema, source."""
    if manifest.is_error():
        print_error(f"Config scan error: {manifest.error()}")
        return
    cfg = manifest.value()
    echo("")
    print_simple_table(
        data=[
            ["Name", manifest.name()],
            ["Module", manifest.import_path() or "—"],
            ["Found At", str(manifest.found_at())],
            ["Description", (manifest.description() or "—")[:200]],
        ],
        headers=["Property", "Value"],
        title="Config Detail",
    )

    # YAML defaults
    try:
        yaml_str = cfg.to_yaml()
    except AttributeError:
        try:
            yaml_str = cfg.model_dump_json(indent=2)
        except Exception:
            yaml_str = str(cfg)
    echo("")
    print_simple_panel(yaml_str, title="Default Values (YAML)")

    # JSON Schema
    try:
        schema = cfg.to_config_schema().json_schema
        schema_json = json.dumps(schema, indent=2, ensure_ascii=False)
        echo("")
        print_simple_panel(schema_json, title="JSON Schema")
    except Exception:
        pass

    # source
    try:
        source = _inspect.getsource(type(cfg))
        echo("")
        print_simple_panel(source, title="Config Source")
    except (TypeError, OSError):
        pass


# ---------------------------------------------------------------------------
# topics
# ---------------------------------------------------------------------------

@manifest_app.command(name="topics")
def list_topics(
    search: str = typer.Argument("", help="Search pattern for topic name or type."),
):
    """List event topic schemas discovered from manifests."""
    project, mode, matrix_mf, mode_mf = _get_context()
    _display_context_header(project, mode)

    matrix_raw = matrix_mf.topics()
    mode_raw = mode_mf.topics() if mode_mf else None

    if search:
        matrix_raw = _filter_manifests(matrix_raw, search)
        mode_raw = _filter_manifests(mode_raw, search) if mode_raw else None
        mode_list = list(mode_raw) if mode_raw else []
        matrix_list = list(matrix_raw)
        if len(mode_list) == 1 and len(matrix_list) <= 1:
            _display_topic_detail(mode_list[0])
            return
        if len(matrix_list) == 1 and not mode_list:
            _display_topic_detail(matrix_list[0])
            return
        if not matrix_list and not mode_list:
            print_warning(f"No topics matching '{search}'.")
            return
        matrix_raw = matrix_list
        mode_raw = mode_list

    _display_two_layer(
        matrix_raw, matrix_mf,
        mode_raw, mode_mf,
        headers=["Name", "Type", "Description"],
        columns=["name", "topic_type", "description"],
        type_label="topics",
        mode=mode,
    )


def _display_topic_detail(manifest: Manifest) -> None:
    """Show a single topic: JSON Schema + model source."""
    if manifest.is_error():
        print_error(f"Topic scan error: {manifest.error()}")
        return
    schema = manifest.value()
    echo("")
    print_simple_table(
        data=[
            ["Name", schema.topic_name],
            ["Type", schema.topic_type],
            ["Description", schema.description or "—"],
            ["Found At", str(manifest.found_at())],
        ],
        headers=["Property", "Value"],
        title="Topic Detail",
    )

    # JSON Schema
    schema_json = json.dumps(schema.json_schema, indent=2, ensure_ascii=False)
    echo("")
    print_simple_panel(schema_json, title="Payload JSON Schema")


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------

@manifest_app.command(name="signals")
def list_signals(
    search: str = typer.Argument("", help="Search pattern for signal name or description."),
):
    """List signal schemas discovered from manifests."""
    project, mode, matrix_mf, mode_mf = _get_context()
    _display_context_header(project, mode)

    matrix_raw = matrix_mf.signals()
    mode_raw = mode_mf.signals() if mode_mf else None

    if search:
        matrix_raw = _filter_manifests(matrix_raw, search)
        mode_raw = _filter_manifests(mode_raw, search) if mode_raw else None
        mode_list = list(mode_raw) if mode_raw else []
        matrix_list = list(matrix_raw)
        if not matrix_list and not mode_list:
            print_warning(f"No signals matching '{search}'.")
            return
        matrix_raw = matrix_list
        mode_raw = mode_list

    _display_two_layer(
        matrix_raw, matrix_mf,
        mode_raw, mode_mf,
        headers=["Name", "Description", "Found At"],
        columns=["name", "description", "found_at"],
        type_label="signals",
        mode=mode,
    )


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

@manifest_app.command(name="parameters")
def show_parameters():
    """Show the parameter schema (single-value, mode-overrides-matrix)."""
    project, mode, matrix_mf, mode_mf = _get_context()
    _display_context_header(project, mode)

    matrix_param = matrix_mf.parameters()
    echo("")
    _display_single_manifest_detail(
        matrix_param,
        layer=f"Matrix ({matrix_mf.root_package()}.parameters)",
    )

    if mode is not None and mode_mf is not None:
        mode_param = mode_mf.parameters()
        echo("")
        _display_single_manifest_detail(
            mode_param,
            layer=f"Mode: {mode.name} ({mode_mf.root_package()}.parameters)",
        )
    elif mode is None:
        _display_no_mode_hint()


def _display_single_manifest_detail(manifest: Manifest, layer: str) -> None:
    """Display a single-value manifest as a detail block."""
    if manifest.is_error():
        print_warning(f"{layer}: {manifest.error()}")
        return
    echo(layer)
    print_simple_table(
        data=[
            ["Name", manifest.name()],
            ["Description", manifest.description() or "—"],
            ["Found At", str(manifest.found_at())],
            ["Import Path", manifest.import_path() or "—"],
        ],
        headers=["Property", "Value"],
        title="Parameter Schema",
    )


# ---------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------

@manifest_app.command(name="resources")
def list_resources(
    search: str = typer.Argument("", help="Search pattern for scheme, host, or description."),
):
    """List resource storage declarations discovered from manifests."""
    project, mode, matrix_mf, mode_mf = _get_context()
    _display_context_header(project, mode)

    matrix_raw = matrix_mf.resources()
    mode_raw = mode_mf.resources() if mode_mf else None

    if search:
        matrix_raw = _filter_manifests(matrix_raw, search)
        mode_raw = _filter_manifests(mode_raw, search) if mode_raw else None
        mode_list = list(mode_raw) if mode_raw else []
        matrix_list = list(matrix_raw)
        if not matrix_list and not mode_list:
            print_warning(f"No resources matching '{search}'.")
            return
        matrix_raw = matrix_list
        mode_raw = mode_list

    _display_two_layer(
        matrix_raw, matrix_mf,
        mode_raw, mode_mf,
        headers=["Scheme", "Host", "Description", "Found At"],
        columns=["scheme", "host", "description", "found_at"],
        type_label="resources",
        mode=mode,
    )


# ---------------------------------------------------------------------------
# channel
# ---------------------------------------------------------------------------

@manifest_app.command(name="channel")
def show_channel():
    """Show the __main__ channel (mode only)."""
    project, mode, matrix_mf, mode_mf = _get_context()
    _display_context_header(project, mode)

    if mode is None or mode_mf is None:
        print_error("Channel is mode-scoped. No active mode.")
        print_info("Use 'moss modes list' to see available modes, then activate one.")
        raise typer.Exit(code=1)

    manifest = mode_mf.channel()
    if manifest.is_error():
        print_error(f"Channel not found: {manifest.error()}")
        return

    channel_obj = manifest.value()
    echo("")
    print_simple_table(
        data=[
            ["Name", channel_obj.name()],
            ["Type", type(channel_obj).__name__],
            ["Description", channel_obj.description() or "—"],
            ["Discovered At", str(manifest.found_at())],
            ["Import Path", manifest.import_path() or "—"],
        ],
        headers=["Property", "Value"],
        title=f"Channel: {channel_obj.name()}",
    )


# ---------------------------------------------------------------------------
# nuclei
# ---------------------------------------------------------------------------

@manifest_app.command(name="nuclei")
def list_nuclei(
    search: str = typer.Argument("", help="Search pattern for nucleus name, description, or signal."),
):
    """List nucleus factories (mode only)."""
    project, mode, matrix_mf, mode_mf = _get_context()
    _display_context_header(project, mode)

    if mode is None or mode_mf is None:
        print_error("Nuclei are mode-scoped. No active mode.")
        print_info("Use 'moss modes list' to see available modes, then activate one.")
        raise typer.Exit(code=1)

    mode_raw = mode_mf.nuclei()
    if search:
        mode_raw = _filter_manifests(mode_raw, search)
        items = list(mode_raw)
        if not items:
            print_warning(f"No nuclei matching '{search}'.")
            return
        mode_raw = items

    count = _display_manifest_list(
        mode_raw,
        headers=["Name", "Description", "Signals", "Found At"],
        columns=["name", "description", "signals", "found_at"],
        title=f"Mode: {mode.name} ({mode_mf.root_package()}.nuclei)",
    )
    if count == 0:
        print_warning(f"No nuclei found in mode '{mode.name}'.")


# ---------------------------------------------------------------------------
# ctml-versions
# ---------------------------------------------------------------------------

@manifest_app.command(name="ctml-versions")
def list_ctml_versions():
    """List CTML versions available in this project."""
    project, mode, matrix_mf, mode_mf = _get_context()
    _display_context_header(project, mode)

    versions = project.ctml_versions()
    if not versions:
        print_warning("No CTML versions found in this project.")
        return

    rows = []
    for ver, filepath in sorted(versions.items()):
        rows.append([ver, str(filepath)])
    print_simple_table(
        data=rows,
        headers=["Version", "File"],
        title="CTML Versions",
    )


# ---------------------------------------------------------------------------
# explain
# ---------------------------------------------------------------------------

@manifest_app.command(name="explain")
def explain_manifests():
    """Self-describe the manifest system — the single source of truth."""
    project, mode, matrix_mf, mode_mf = _get_context()
    _display_context_header(project, mode)

    echo(matrix_mf.explain())

    if mode_mf is not None:
        echo("")
        echo(mode_mf.explain())
        echo("")
        print_info(
            f"当前模式 '{mode.name}' 的有效视图 = "
            f"{matrix_mf.root_package()} (全局) + {mode_mf.root_package()} (模式追加)"
        )
    else:
        echo("")
        _display_no_mode_hint()
