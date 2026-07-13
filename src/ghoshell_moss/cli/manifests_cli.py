"""
moss manifests — workspace static declarations.

Queries the manifest system via Project (not Host): MatrixManifest for global
baseline (MOSS.manifests) and ModeManifests for the active mode's effective
view (HOST, which extends MOSS.manifests via Python import).

Each command shows two tables when a mode is active; a single Matrix table
when no mode is active.  Context header (mode/network/ghost) only appears in
``explain`` — list commands show just their data.
"""

import inspect as _inspect
import json
from pathlib import Path
from typing import Iterable

import typer
from ghoshell_container import Provider

from ghoshell_moss.core.blueprint.environment import (
    NONE_MOSS_MODE,
    NONE_GHOST_NAME,
    DEFAULT_NETWORK_NAME,
    DEFAULT_NETWORK_SCOPE,
)
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

def _display_two_layer(
    matrix_rows: list[list[str]],
    matrix_count: int,
    matrix_pkg: str,
    mode_rows: list[list[str]] | None,
    mode_count: int | None,
    mode_pkg: str | None,
    mode_name: str | None,
    headers: list[str],
    title_label: str,
) -> None:
    """Display Matrix + optional Mode tables from pre-built rows."""
    if matrix_count == 0:
        print_warning(f"No {title_label} found in Matrix ({matrix_pkg}.{title_label}).")
    else:
        print_simple_table(
            data=matrix_rows, headers=headers,
            title=f"Matrix ({matrix_pkg}.{title_label})",
        )

    if mode_rows is not None and mode_pkg is not None and mode_name is not None:
        if mode_count == 0:
            echo("")
            print_warning(f"No {title_label} found in mode '{mode_name}' ({mode_pkg}.{title_label}).")
        else:
            echo("")
            print_simple_table(
                data=mode_rows, headers=headers,
                title=f"Mode: {mode_name} ({mode_pkg}.{title_label})",
            )
            echo("")
            print_info(
                f"{matrix_count} from {matrix_pkg}, {mode_count} effective in mode "
                f"({mode_pkg} extends {matrix_pkg} via import)"
            )


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
    project, mode, matrix_mf, mode_mf = _get_context()

    matrix_raw = list(matrix_mf.providers())
    mode_raw = list(mode_mf.providers()) if mode_mf else []

    if search:
        matrix_raw = _filter_manifests(matrix_raw, search)
        mode_raw = _filter_manifests(mode_raw, search) if mode_raw else []
        # single match → detail
        if len(mode_raw) == 1 and len(matrix_raw) <= 1:
            _display_provider_detail(mode_raw[0])
            return
        if len(matrix_raw) == 1 and not mode_raw:
            _display_provider_detail(matrix_raw[0])
            return
        if not matrix_raw and not mode_raw:
            print_warning(f"No providers matching '{search}'.")
            return

    m_rows, m_count = _provider_rows(matrix_raw)
    mode_rows, mode_count = _provider_rows(mode_raw) if mode_raw else (None, None)

    _display_two_layer(
        matrix_rows=m_rows, matrix_count=m_count,
        matrix_pkg=matrix_mf.root_package(),
        mode_rows=mode_rows, mode_count=mode_count,
        mode_pkg=mode_mf.root_package() if mode_mf else None,
        mode_name=mode.name if mode else None,
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
        try:
            schema = cfg.to_config_schema()
            props = schema.json_schema.get("properties", {})
            fields = ", ".join(
                f"{k}:{v.get('type', '?')}" for k, v in props.items()
            ) if props else "—"
        except Exception:
            fields = "—"
        desc = schema.description if schema.description else (m.description() or "—")
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
    project, mode, matrix_mf, mode_mf = _get_context()

    matrix_raw = list(matrix_mf.configs())
    mode_raw = list(mode_mf.configs()) if mode_mf else []

    if search:
        matrix_raw = _filter_manifests(matrix_raw, search)
        mode_raw = _filter_manifests(mode_raw, search) if mode_raw else []
        if len(mode_raw) == 1 and len(matrix_raw) <= 1:
            _display_config_detail(mode_raw[0])
            return
        if len(matrix_raw) == 1 and not mode_raw:
            _display_config_detail(matrix_raw[0])
            return
        if not matrix_raw and not mode_raw:
            print_warning(f"No configs matching '{search}'.")
            return

    if detail and mode_raw:
        if len(mode_raw) == 1:
            _display_config_detail(mode_raw[0])
            return

    m_rows, m_count = _config_rows(matrix_raw)
    mode_rows, mode_count = _config_rows(mode_raw) if mode_raw else (None, None)

    _display_two_layer(
        matrix_rows=m_rows, matrix_count=m_count,
        matrix_pkg=matrix_mf.root_package(),
        mode_rows=mode_rows, mode_count=mode_count,
        mode_pkg=mode_mf.root_package() if mode_mf else None,
        mode_name=mode.name if mode else None,
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
    project, mode, matrix_mf, mode_mf = _get_context()

    matrix_raw = list(matrix_mf.topics())
    mode_raw = list(mode_mf.topics()) if mode_mf else []

    if search:
        matrix_raw = _filter_manifests(matrix_raw, search)
        mode_raw = _filter_manifests(mode_raw, search) if mode_raw else []
        if len(mode_raw) == 1 and len(matrix_raw) <= 1:
            _display_topic_detail(mode_raw[0])
            return
        if len(matrix_raw) == 1 and not mode_raw:
            _display_topic_detail(matrix_raw[0])
            return
        if not matrix_raw and not mode_raw:
            print_warning(f"No topics matching '{search}'.")
            return

    m_rows, m_count = _topic_rows(matrix_raw)
    mode_rows, mode_count = _topic_rows(mode_raw) if mode_raw else (None, None)

    _display_two_layer(
        matrix_rows=m_rows, matrix_count=m_count,
        matrix_pkg=matrix_mf.root_package(),
        mode_rows=mode_rows, mode_count=mode_count,
        mode_pkg=mode_mf.root_package() if mode_mf else None,
        mode_name=mode.name if mode else None,
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
    project, mode, matrix_mf, mode_mf = _get_context()

    matrix_raw = list(matrix_mf.signals())
    mode_raw = list(mode_mf.signals()) if mode_mf else []

    if search:
        matrix_raw = _filter_manifests(matrix_raw, search)
        mode_raw = _filter_manifests(mode_raw, search) if mode_raw else []
        if len(mode_raw) == 1 and len(matrix_raw) <= 1:
            _display_signal_detail(mode_raw[0])
            return
        if len(matrix_raw) == 1 and not mode_raw:
            _display_signal_detail(matrix_raw[0])
            return
        if not matrix_raw and not mode_raw:
            print_warning(f"No signals matching '{search}'.")
            return

    m_rows, m_count = _signal_rows(matrix_raw)
    mode_rows, mode_count = _signal_rows(mode_raw) if mode_raw else (None, None)

    _display_two_layer(
        matrix_rows=m_rows, matrix_count=m_count,
        matrix_pkg=matrix_mf.root_package(),
        mode_rows=mode_rows, mode_count=mode_count,
        mode_pkg=mode_mf.root_package() if mode_mf else None,
        mode_name=mode.name if mode else None,
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
    """Show the parameter schema (single-value, mode-overrides-matrix)."""
    project, mode, matrix_mf, mode_mf = _get_context()

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
    project, mode, matrix_mf, mode_mf = _get_context()

    matrix_raw = list(matrix_mf.resources())
    mode_raw = list(mode_mf.resources()) if mode_mf else []

    if search:
        matrix_raw = _filter_manifests(matrix_raw, search)
        mode_raw = _filter_manifests(mode_raw, search) if mode_raw else []
        if not matrix_raw and not mode_raw:
            print_warning(f"No resources matching '{search}'.")
            return

    m_rows, m_count = _resource_rows(matrix_raw)
    mode_rows, mode_count = _resource_rows(mode_raw) if mode_raw else (None, None)

    _display_two_layer(
        matrix_rows=m_rows, matrix_count=m_count,
        matrix_pkg=matrix_mf.root_package(),
        mode_rows=mode_rows, mode_count=mode_count,
        mode_pkg=mode_mf.root_package() if mode_mf else None,
        mode_name=mode.name if mode else None,
        headers=["Scheme", "Storage Host", "Description", "Found At"],
        title_label="resources",
    )


# ---------------------------------------------------------------------------
# channel
# ---------------------------------------------------------------------------

@manifest_app.command(name="channel")
def show_channel():
    """Show the __main__ channel (mode only)."""
    project, mode, matrix_mf, mode_mf = _get_context()

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
    """List nucleus factories (mode only)."""
    project, mode, matrix_mf, mode_mf = _get_context()

    if mode is None or mode_mf is None:
        print_error("Nuclei are mode-scoped. No active mode.")
        print_info("Use 'moss modes list' to see available modes, then activate one.")
        raise typer.Exit(code=1)

    raw = list(mode_mf.nuclei())
    if search:
        raw = _filter_manifests(raw, search)
        if not raw:
            print_warning(f"No nuclei matching '{search}'.")
            return

    rows, count = _nucleus_rows(raw)
    if count == 0:
        print_warning(f"No nuclei found in mode '{mode.name}'.")
        return

    print_simple_table(
        data=rows,
        headers=["Name", "Description", "Signals", "Found At"],
        title=f"Mode: {mode.name} ({mode_mf.root_package()}.nuclei)",
    )


# ---------------------------------------------------------------------------
# ctml-versions
# ---------------------------------------------------------------------------

@manifest_app.command(name="ctml-versions")
def list_ctml_versions():
    """List CTML versions available in this project."""
    project, mode, matrix_mf, mode_mf = _get_context()

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
# explain — single source of truth for context + manifest system
# ---------------------------------------------------------------------------

@manifest_app.command(name="explain")
def explain_manifests():
    """Self-describe the manifest system — the single source of truth."""
    project, mode, matrix_mf, mode_mf = _get_context()
    _display_context_header(project, mode)

    echo("")
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
        _display_no_mode_hint()
