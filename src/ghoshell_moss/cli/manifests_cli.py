"""
moss manifests — workspace static declarations.

Queries the manifest system via the Project abstraction, across three layers:
  - project (MOSS.manifests)   — 全模式装线 (workspace 级基线)
  - matrix  (MATRIX.manifests) — 跨 cell 环境能力 (mode 激活时)
  - host    (HOST)             — host 节点专属

Each command shows three tables when a mode is active; a single project table
when no mode is active.  Context header (mode/network/ghost) only appears in
``explain`` — list commands show just their data.
"""

import inspect as _inspect
import json
from typing import Iterable

import typer
from ghoshell_container import Provider

from ghoshell_moss.core.blueprint.environment import (
    NONE_MOSS_MODE,
    NONE_GHOST_NAME,
    DEFAULT_NETWORK_NAME,
    DEFAULT_NETWORK_SCOPE,
)
from ghoshell_moss.core.blueprint.project import Manifest, Project, ProjectManifest, HostModeManifests, HostMode

from .utils import (
    print_simple_table, print_simple_panel,
    print_error, print_warning, print_info, echo,
)


def _print_hint() -> None:
    echo("")
    print_info(
        "Design: 'moss project design'. "
        "Files to edit are the 'Found At' paths in the tables above."
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
    ProjectManifest,   # project (MOSS.manifests) — 全模式装线
    ProjectManifest,   # matrix (MATRIX.manifests) — 跨 cell 环境能力, None if no_mode
    HostModeManifests | None,  # host (HOST) — host 节点专属, None if no_mode
]


def _get_context() -> _Context:
    """Resolve the current Project and three-layer manifests context.

    Returns (project, mode, project_manifests, matrix_manifests, host_manifests).
    matrix / host manifests are None when no mode is active.
    """
    project = Project.discover()

    try:
        mode = project.current_mode()
    except Exception:
        mode = None

    project_mf = project.project_manifests()

    matrix_mf = None
    host_mf = None
    if mode is not None:
        try:
            mode.bootstrap()
            matrix_mf = mode.matrix_manifests()
            host_mf = mode.manifests()
        except Exception:
            matrix_mf = None
            host_mf = None

    return project, mode, project_mf, matrix_mf, host_mf


def _display_context_header(project: Project, mode: HostMode | None) -> None:
    """Show the active context: mode, network, ghost, scope."""
    env = project.env
    mode_name = mode.name if mode else NONE_MOSS_MODE
    mode_source = env.moss_meta.default_mode if env.moss_meta.default_mode else "—"
    ghost_name = env.ghost_name or env.moss_meta.default_ghost or NONE_GHOST_NAME
    network_name = env.network or DEFAULT_NETWORK_NAME
    scope = env.network_scope or DEFAULT_NETWORK_SCOPE

    rows = [
        ["Mode", mode_name + (f"  (from MOSS.md default: {mode_source})" if mode_name != NONE_MOSS_MODE else "")],
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


# ---------------------------------------------------------------------------
# two-layer display (Matrix + optional Mode)
# ---------------------------------------------------------------------------

def _display_three_layer(
    project_rows: list[list[str]],
    project_count: int,
    project_pkg: str,
    matrix_rows: list[list[str]] | None,
    matrix_count: int | None,
    matrix_pkg: str | None,
    host_rows: list[list[str]] | None,
    host_count: int | None,
    host_pkg: str | None,
    host_name: str | None,
    headers: list[str],
    title_label: str,
) -> None:
    """Display project + matrix + optional host tables from pre-built rows."""
    if project_count == 0:
        print_warning(f"No {title_label} found in project ({project_pkg}.{title_label}).")
    else:
        print_simple_table(
            data=project_rows, headers=headers,
            title=f"Project ({project_pkg}.{title_label})",
        )

    if matrix_rows is not None and matrix_pkg is not None:
        if matrix_count == 0:
            echo("")
            print_warning(f"No {title_label} found in matrix ({matrix_pkg}.{title_label}).")
        else:
            echo("")
            print_simple_table(
                data=matrix_rows, headers=headers,
                title=f"Matrix ({matrix_pkg}.{title_label})",
            )

    if host_rows is not None and host_pkg is not None and host_name is not None:
        if host_count == 0:
            echo("")
            print_warning(f"No {title_label} found in host '{host_name}' ({host_pkg}.{title_label}).")
        else:
            echo("")
            print_simple_table(
                data=host_rows, headers=headers,
                title=f"Host: {host_name} ({host_pkg}.{title_label})",
            )
    _print_hint()


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

def _import_path(cls: type) -> str:
    """Convert a class to its Python import path."""
    module = getattr(cls, '__module__', '')
    qualname = getattr(cls, '__qualname__', cls.__name__)
    if module:
        return f'{module}.{qualname}'
    return qualname


def _provider_rows(manifests: Iterable[Manifest]) -> tuple[list[list[str]], int]:
    """Build table rows for provider manifests.

    Columns: Contract | Aliases | Type | Docstring | Found At
    """
    rows = []
    for m in manifests:
        if m.is_error():
            rows.append([m.name(), "ERROR", "—", str(m.error())[:80], str(m.found_at())])
            continue
        p = m.value()
        alias_strs = [_import_path(a) for a in p.aliases()] if p.aliases() else []
        aliases = ", ".join(alias_strs) if alias_strs else "—"
        ptype = "Singleton" if p.singleton() else "Factory"
        desc = (m.description() or "—")[:100]
        rows.append([
            m.name(),
            aliases,
            ptype,
            desc,
            str(m.found_at()),
        ])
    return rows, len(rows)


def _display_provider_detail(manifest: Manifest[Provider]) -> None:
    """Show a single provider in detail: contract source code."""
    if manifest.is_error():
        print_error(f"Provider scan error: {manifest.error()}")
        return
    provider = manifest.value()
    contract_type = provider.contract()

    alias_strs = [_import_path(a) for a in provider.aliases()] if provider.aliases() else []
    aliases = ", ".join(alias_strs) if alias_strs else "—"
    echo("")
    print_simple_table(
        data=[
            ["Contract", manifest.name()],
            ["Aliases", aliases],
            ["Type", "Singleton" if provider.singleton() else "Factory"],
            ["Found At", str(manifest.found_at())],
            ["Import Path", manifest.import_path() or "—"],
            ["Docstring", (manifest.description() or "—")[:200]],
        ],
        headers=["Property", "Value"],
        title="Provider Detail",
    )

    if manifest.source():
        echo("")
        print_simple_panel(manifest.source(), title="Contract Source")
    else:
        try:
            source = _inspect.getsource(contract_type)
            echo("")
            print_simple_panel(source, title="Contract Source")
        except (TypeError, OSError):
            print_info("Source unavailable (compiled or built-in contract).")


@manifest_app.command(name="providers")
def list_providers(
    search: str = typer.Argument(
        "", help="Search pattern for contract import path or provider type.",
    ),
):
    """List IoC providers discovered from manifests."""
    project, mode, project_mf, matrix_mf, host_mf = _get_context()

    project_raw = list(project_mf.providers())
    matrix_raw = list(matrix_mf.providers()) if matrix_mf else []
    host_raw = list(host_mf.providers()) if host_mf else []

    if search:
        project_raw = _filter_manifests(project_raw, search)
        matrix_raw = _filter_manifests(matrix_raw, search) if matrix_raw else []
        host_raw = _filter_manifests(host_raw, search) if host_raw else []
        all_matches = project_raw + matrix_raw + host_raw
        if len(all_matches) == 1:
            _display_provider_detail(all_matches[0])
            return
        if not all_matches:
            print_warning(f"No providers matching '{search}'.")
            return

    p_rows, p_count = _provider_rows(project_raw)
    matrix_rows, matrix_count = _provider_rows(matrix_raw) if matrix_raw else (None, None)
    host_rows, host_count = _provider_rows(host_raw) if host_raw else (None, None)

    _display_three_layer(
        project_rows=p_rows, project_count=p_count,
        project_pkg=project_mf.root_package(),
        matrix_rows=matrix_rows, matrix_count=matrix_count,
        matrix_pkg=matrix_mf.root_package() if matrix_mf else None,
        host_rows=host_rows, host_count=host_count,
        host_pkg=host_mf.root_package() if host_mf else None,
        host_name=mode.name if mode else None,
        headers=["Contract", "Aliases", "Type", "Docstring", "Found At"],
        title_label="providers",
    )


# ---------------------------------------------------------------------------
# configs
# ---------------------------------------------------------------------------

def _config_rows(manifests: Iterable[Manifest]) -> tuple[list[list[str]], int]:
    """Build table rows for config manifests.

    Columns: Name | Fields | Description | Found At
    """
    rows = []
    for m in manifests:
        if m.is_error():
            rows.append([m.name(), "ERROR", str(m.error())[:80], str(m.found_at())])
            continue
        cfg = m.value()
        # extract field names:type from json_schema
        schema = None
        try:
            schema = cfg.to_config_schema()
            props = schema.json_schema.get("properties", {})
            fields = ", ".join(
                f"{k}:{v.get('type', '?')}" for k, v in props.items()
            ) if props else "—"
        except Exception:
            fields = "—"
        desc = schema.description if schema and schema.description else (m.description() or "—")
        rows.append([
            m.name(),
            fields[:120],
            desc[:100],
            str(m.found_at()),
        ])
    return rows, len(rows)


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

    try:
        yaml_str = cfg.to_yaml()
    except AttributeError:
        try:
            yaml_str = cfg.model_dump_json(indent=2)
        except Exception:
            yaml_str = str(cfg)
    echo("")
    print_simple_panel(yaml_str, title="Default Values (YAML)")

    try:
        schema = cfg.to_config_schema().json_schema
        schema_json = json.dumps(schema, indent=2, ensure_ascii=False)
        echo("")
        print_simple_panel(schema_json, title="JSON Schema")
    except Exception:
        pass

    try:
        source = _inspect.getsource(type(cfg))
        echo("")
        print_simple_panel(source, title="Config Source")
    except (TypeError, OSError):
        pass


@manifest_app.command(name="configs")
def list_configs(
    search: str = typer.Argument("", help="Search pattern for config name."),
    detail: bool = typer.Option(False, "--detail", "-d", help="Show full schema and defaults."),
):
    """List configuration models discovered from manifests."""
    project, mode, project_mf, matrix_mf, host_mf = _get_context()

    project_raw = list(project_mf.configs())
    matrix_raw = list(matrix_mf.configs()) if matrix_mf else []
    host_raw = list(host_mf.configs()) if host_mf else []

    if search:
        project_raw = _filter_manifests(project_raw, search)
        matrix_raw = _filter_manifests(matrix_raw, search) if matrix_raw else []
        host_raw = _filter_manifests(host_raw, search) if host_raw else []
        all_matches = project_raw + matrix_raw + host_raw
        if len(all_matches) == 1:
            _display_config_detail(all_matches[0])
            return
        if not all_matches:
            print_warning(f"No configs matching '{search}'.")
            return

    if detail and len(host_raw) == 1:
        _display_config_detail(host_raw[0])
        return

    p_rows, p_count = _config_rows(project_raw)
    matrix_rows, matrix_count = _config_rows(matrix_raw) if matrix_raw else (None, None)
    host_rows, host_count = _config_rows(host_raw) if host_raw else (None, None)

    _display_three_layer(
        project_rows=p_rows, project_count=p_count,
        project_pkg=project_mf.root_package(),
        matrix_rows=matrix_rows, matrix_count=matrix_count,
        matrix_pkg=matrix_mf.root_package() if matrix_mf else None,
        host_rows=host_rows, host_count=host_count,
        host_pkg=host_mf.root_package() if host_mf else None,
        host_name=mode.name if mode else None,
        headers=["Name", "Fields", "Description", "Found At"],
        title_label="configs",
    )


# ---------------------------------------------------------------------------
# topics
# ---------------------------------------------------------------------------

def _topic_rows(manifests: Iterable[Manifest]) -> tuple[list[list[str]], int]:
    """Build table rows for topic manifests.

    Columns: Name | Type | Model Path | Description | Found At
    """
    rows = []
    for m in manifests:
        if m.is_error():
            rows.append([m.name(), "ERROR", "—", str(m.error())[:80], str(m.found_at())])
            continue
        schema = m.value()
        model_path = m.import_path() or "—"
        desc = schema.description or m.description() or "—"
        rows.append([
            m.name(),
            schema.topic_type,
            model_path,
            desc[:100],
            str(m.found_at()),
        ])
    return rows, len(rows)


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
            ["Import Path", manifest.import_path() or "—"],
        ],
        headers=["Property", "Value"],
        title="Topic Detail",
    )

    schema_json = json.dumps(schema.json_schema, indent=2, ensure_ascii=False)
    echo("")
    print_simple_panel(schema_json, title="Payload JSON Schema")

    if manifest.source():
        echo("")
        print_simple_panel(manifest.source(), title="Topic Model Source")


@manifest_app.command(name="topics")
def list_topics(
    search: str = typer.Argument("", help="Search pattern for topic name or type."),
):
    """List event topic schemas discovered from manifests."""
    project, mode, project_mf, matrix_mf, host_mf = _get_context()

    project_raw = list(project_mf.topics())
    matrix_raw = list(matrix_mf.topics()) if matrix_mf else []
    host_raw = list(host_mf.topics()) if host_mf else []

    if search:
        project_raw = _filter_manifests(project_raw, search)
        matrix_raw = _filter_manifests(matrix_raw, search) if matrix_raw else []
        host_raw = _filter_manifests(host_raw, search) if host_raw else []
        all_matches = project_raw + matrix_raw + host_raw
        if len(all_matches) == 1:
            _display_topic_detail(all_matches[0])
            return
        if not all_matches:
            print_warning(f"No topics matching '{search}'.")
            return

    p_rows, p_count = _topic_rows(project_raw)
    matrix_rows, matrix_count = _topic_rows(matrix_raw) if matrix_raw else (None, None)
    host_rows, host_count = _topic_rows(host_raw) if host_raw else (None, None)

    _display_three_layer(
        project_rows=p_rows, project_count=p_count,
        project_pkg=project_mf.root_package(),
        matrix_rows=matrix_rows, matrix_count=matrix_count,
        matrix_pkg=matrix_mf.root_package() if matrix_mf else None,
        host_rows=host_rows, host_count=host_count,
        host_pkg=host_mf.root_package() if host_mf else None,
        host_name=mode.name if mode else None,
        headers=["Name", "Type", "Model Path", "Description", "Found At"],
        title_label="topics",
    )


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------

def _signal_rows(manifests: Iterable[Manifest]) -> tuple[list[list[str]], int]:
    """Build table rows for signal manifests.

    Columns: Name | Description | Found At
    """
    rows = []
    for m in manifests:
        if m.is_error():
            rows.append([m.name(), str(m.error())[:80], str(m.found_at())])
            continue
        desc = m.description() or "—"
        rows.append([m.name(), desc[:120], str(m.found_at())])
    return rows, len(rows)


def _display_signal_detail(manifest: Manifest) -> None:
    """Show a single signal with full description and source if available."""
    if manifest.is_error():
        print_error(f"Signal scan error: {manifest.error()}")
        return
    schema = manifest.value()
    echo("")
    print_simple_table(
        data=[
            ["Name", manifest.name()],
            ["Description", (manifest.description() or "—")[:300]],
            ["Found At", str(manifest.found_at())],
            ["Import Path", manifest.import_path() or "—"],
        ],
        headers=["Property", "Value"],
        title="Signal Detail",
    )
    if manifest.source():
        echo("")
        print_simple_panel(manifest.source(), title="Signal Source")


@manifest_app.command(name="signals")
def list_signals(
    search: str = typer.Argument("", help="Search pattern for signal name or description."),
):
    """List signal schemas discovered from manifests."""
    project, mode, project_mf, matrix_mf, host_mf = _get_context()

    project_raw = list(project_mf.signals())
    matrix_raw = list(matrix_mf.signals()) if matrix_mf else []
    host_raw = list(host_mf.signals()) if host_mf else []

    if search:
        project_raw = _filter_manifests(project_raw, search)
        matrix_raw = _filter_manifests(matrix_raw, search) if matrix_raw else []
        host_raw = _filter_manifests(host_raw, search) if host_raw else []
        all_matches = project_raw + matrix_raw + host_raw
        if len(all_matches) == 1:
            _display_signal_detail(all_matches[0])
            return
        if not all_matches:
            print_warning(f"No signals matching '{search}'.")
            return

    p_rows, p_count = _signal_rows(project_raw)
    matrix_rows, matrix_count = _signal_rows(matrix_raw) if matrix_raw else (None, None)
    host_rows, host_count = _signal_rows(host_raw) if host_raw else (None, None)

    _display_three_layer(
        project_rows=p_rows, project_count=p_count,
        project_pkg=project_mf.root_package(),
        matrix_rows=matrix_rows, matrix_count=matrix_count,
        matrix_pkg=matrix_mf.root_package() if matrix_mf else None,
        host_rows=host_rows, host_count=host_count,
        host_pkg=host_mf.root_package() if host_mf else None,
        host_name=mode.name if mode else None,
        headers=["Name", "Description", "Found At"],
        title_label="signals",
    )


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------

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


@manifest_app.command(name="parameters")
def show_parameters():
    """Show the parameter schema (project/matrix/host three layers)."""
    project, mode, project_mf, matrix_mf, host_mf = _get_context()

    echo("")
    _display_single_manifest_detail(
        project_mf.parameters(),
        layer=f"Project ({project_mf.root_package()}.parameters)",
    )

    if matrix_mf is not None:
        echo("")
        _display_single_manifest_detail(
            matrix_mf.parameters(),
            layer=f"Matrix ({matrix_mf.root_package()}.parameters)",
        )

    if mode is not None and host_mf is not None:
        echo("")
        _display_single_manifest_detail(
            host_mf.parameters(),
            layer=f"Host: {mode.name} ({host_mf.root_package()}.parameters)",
        )
    _print_hint()


# ---------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------

def _resource_rows(manifests: Iterable[Manifest]) -> tuple[list[list[str]], int]:
    """Build table rows for resource manifests.

    Columns: Scheme | Storage Host | Description | Found At
    """
    rows = []
    for m in manifests:
        if m.is_error():
            rows.append([m.name(), "ERROR", str(m.error())[:80], str(m.found_at())])
            continue
        meta = m.value()
        scheme = meta.scheme() if hasattr(meta, 'scheme') else m.name()
        host = meta.host if hasattr(meta, 'host') else "—"
        desc = (meta.description() if hasattr(meta, 'description') else m.description() or "—")
        rows.append([
            scheme,
            host,
            desc[:100],
            str(m.found_at()),
        ])
    return rows, len(rows)


@manifest_app.command(name="resources")
def list_resources(
    search: str = typer.Argument("", help="Search pattern for scheme, host, or description."),
):
    """List resource storage declarations discovered from manifests."""
    project, mode, project_mf, matrix_mf, host_mf = _get_context()

    project_raw = list(project_mf.resources())
    matrix_raw = list(matrix_mf.resources()) if matrix_mf else []
    host_raw = list(host_mf.resources()) if host_mf else []

    if search:
        project_raw = _filter_manifests(project_raw, search)
        matrix_raw = _filter_manifests(matrix_raw, search) if matrix_raw else []
        host_raw = _filter_manifests(host_raw, search) if host_raw else []
        if not project_raw and not matrix_raw and not host_raw:
            print_warning(f"No resources matching '{search}'.")
            return

    p_rows, p_count = _resource_rows(project_raw)
    matrix_rows, matrix_count = _resource_rows(matrix_raw) if matrix_raw else (None, None)
    host_rows, host_count = _resource_rows(host_raw) if host_raw else (None, None)

    _display_three_layer(
        project_rows=p_rows, project_count=p_count,
        project_pkg=project_mf.root_package(),
        matrix_rows=matrix_rows, matrix_count=matrix_count,
        matrix_pkg=matrix_mf.root_package() if matrix_mf else None,
        host_rows=host_rows, host_count=host_count,
        host_pkg=host_mf.root_package() if host_mf else None,
        host_name=mode.name if mode else None,
        headers=["Scheme", "Storage Host", "Description", "Found At"],
        title_label="resources",
    )


# ---------------------------------------------------------------------------
# channel
# ---------------------------------------------------------------------------

@manifest_app.command(name="channel")
def show_channel():
    """Show the __main__ channel (host only)."""
    project, mode, project_mf, matrix_mf, host_mf = _get_context()

    if mode is None or host_mf is None:
        print_error("Channel is host-scoped. No active mode.")
        print_info("Use 'moss modes list' to see available modes, then activate one.")
        raise typer.Exit(code=1)

    manifest = host_mf.channel()
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
    _print_hint()


# ---------------------------------------------------------------------------
# nuclei
# ---------------------------------------------------------------------------

def _nucleus_rows(manifests: Iterable[Manifest]) -> tuple[list[list[str]], int]:
    """Build table rows for nucleus manifests.

    Columns: Name | Description | Signals | Found At
    """
    rows = []
    for m in manifests:
        if m.is_error():
            rows.append([m.name(), str(m.error())[:80], "—", str(m.found_at())])
            continue
        v = m.value()
        signals = ", ".join(s.signal_name() for s in v.signals()) if hasattr(v, 'signals') else "—"
        desc = (v.description() if hasattr(v, 'description') else m.description() or "—")
        rows.append([
            m.name(),
            desc[:100],
            signals,
            str(m.found_at()),
        ])
    return rows, len(rows)


@manifest_app.command(name="nuclei")
def list_nuclei(
    search: str = typer.Argument("", help="Search pattern for nucleus name, description, or signal."),
):
    """List nucleus factories (project + matrix + host three layers)."""
    project, mode, project_mf, matrix_mf, host_mf = _get_context()

    project_raw = list(project_mf.nuclei())
    matrix_raw = list(matrix_mf.nuclei()) if matrix_mf else []
    host_raw = list(host_mf.nuclei()) if host_mf else []

    if search:
        project_raw = _filter_manifests(project_raw, search)
        matrix_raw = _filter_manifests(matrix_raw, search) if matrix_raw else []
        host_raw = _filter_manifests(host_raw, search) if host_raw else []
        if not project_raw and not matrix_raw and not host_raw:
            print_warning(f"No nuclei matching '{search}'.")
            return

    p_rows, p_count = _nucleus_rows(project_raw)
    matrix_rows, matrix_count = _nucleus_rows(matrix_raw) if matrix_raw else (None, None)
    host_rows, host_count = _nucleus_rows(host_raw) if host_raw else (None, None)

    _display_three_layer(
        project_rows=p_rows, project_count=p_count,
        project_pkg=project_mf.root_package(),
        matrix_rows=matrix_rows, matrix_count=matrix_count,
        matrix_pkg=matrix_mf.root_package() if matrix_mf else None,
        host_rows=host_rows, host_count=host_count,
        host_pkg=host_mf.root_package() if host_mf else None,
        host_name=mode.name if mode else None,
        headers=["Name", "Description", "Signals", "Found At"],
        title_label="nuclei",
    )


# ---------------------------------------------------------------------------
# ctml-versions
# ---------------------------------------------------------------------------

@manifest_app.command(name="ctml-versions")
def list_ctml_versions():
    """List CTML versions available in this project."""
    project, mode, _, _, _ = _get_context()

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
    _print_hint()


# ---------------------------------------------------------------------------
# explain — single source of truth for context + manifest system
# ---------------------------------------------------------------------------

@manifest_app.command(name="explain")
def explain_manifests():
    """Self-describe the manifest system — the single source of truth."""
    project, mode, project_mf, matrix_mf, host_mf = _get_context()
    _display_context_header(project, mode)

    echo("")
    echo(project_mf.explain())

    if matrix_mf is not None:
        echo("")
        echo(matrix_mf.explain())
    if host_mf is not None:
        echo("")
        echo(host_mf.explain())

    if mode is not None:
        echo("")
        parts = [f"{project_mf.root_package()} (project 全模式)"]
        if matrix_mf is not None:
            parts.append(f"{matrix_mf.root_package()} (matrix 跨cell)")
        if host_mf is not None:
            parts.append(f"{host_mf.root_package()} (host 仅host)")
        print_info(f"当前模式 '{mode.name}' 的有效视图 = " + " + ".join(parts))
    else:
        _display_no_mode_hint()
    _print_hint()


# ---------------------------------------------------------------------------
# contracts — live IoC container view (not manifest scan)
# ---------------------------------------------------------------------------

@manifest_app.command(name="contracts")
def show_contracts(
    search: str = typer.Argument(
        "",
        help="Contract name or import path pattern. Exact match shows source.",
    ),
    json_mode: bool = typer.Option(
        False, "--json",
        help="Output as JSON array.",
    ),
):
    """
    List all bound IoC contracts from the live Matrix container.

    Unlike other manifest commands, this queries the live Matrix instance
    (Matrix.discover -> container.contracts), not the static manifest scans.
    Pass a contract name to read its source code.
    """
    from ghoshell_moss.core.blueprint.matrix import Matrix

    try:
        matrix = Matrix.discover()
    except Exception as e:
        print_error(f"Failed to discover Matrix: {e}")
        raise typer.Exit(code=1)

    contracts = sorted(
        matrix.container.contracts(),
        key=lambda c: (c.__module__, c.__name__),
    )

    if json_mode:
        _show_contracts_json(contracts)
    elif search:
        _show_contract_detail(contracts, search)
    else:
        _show_contracts_table(contracts)


def _show_contracts_table(contracts: list[type]) -> None:
    from ghoshell_common.helpers import generate_import_path

    rows = []
    for ct in contracts:
        import_path = generate_import_path(ct)
        try:
            source_file = _inspect.getfile(ct)
        except (TypeError, OSError):
            source_file = "—"  # em dash — builtin / C extension
        rows.append([ct.__name__, import_path, source_file])

    print_simple_table(
        data=rows,
        headers=["Contract", "Import Path", "Source File"],
        title="IoC Contracts (live container)",
    )


def _show_contract_detail(contracts: list[type], search: str) -> None:
    from ghoshell_common.helpers import generate_import_path

    matches = [
        ct for ct in contracts
        if search.lower() in ct.__name__.lower()
        or search.lower() in generate_import_path(ct).lower()
    ]

    if not matches:
        available = ", ".join(sorted({c.__name__ for c in contracts}))
        print_error(
            f"Contract matching '{search}' not found. Available: {available}"
        )
        raise typer.Exit(code=1)

    # single exact name match -> show source
    exact_names = [ct for ct in matches if ct.__name__ == search]
    if len(exact_names) == 1:
        ct = exact_names[0]
        import_path = generate_import_path(ct)
        try:
            source_file = _inspect.getfile(ct)
        except (TypeError, OSError):
            source_file = str(None)

        print_simple_panel(
            f"Name: {ct.__name__}\n"
            f"Import: {import_path}\n"
            f"Source: {source_file}",
            title=f"Contract: {ct.__name__}",
        )
        echo("")

        try:
            source = _inspect.getsource(ct)
            from .utils import print_code
            print_code(source)
        except (TypeError, OSError) as e:
            print_warning(f"Cannot read source (builtin / C extension): {e}")
        return

    # multiple or fuzzy match -> filtered table
    rows = []
    for ct in matches:
        import_path = generate_import_path(ct)
        try:
            source_file = _inspect.getfile(ct)
        except (TypeError, OSError):
            source_file = "—"
        rows.append([ct.__name__, import_path, source_file])
    print_simple_table(
        data=rows,
        headers=["Contract", "Import Path", "Source File"],
        title=f"IoC Contracts matching '{search}' ({len(matches)} results)",
    )


def _show_contracts_json(contracts: list[type]) -> None:
    from ghoshell_common.helpers import generate_import_path

    result = []
    for ct in contracts:
        try:
            source_file = _inspect.getfile(ct)
        except (TypeError, OSError):
            source_file = None
        result.append({
            "name": ct.__name__,
            "import_path": generate_import_path(ct),
            "source_file": source_file,
        })
    echo(json.dumps(result, indent=2))
