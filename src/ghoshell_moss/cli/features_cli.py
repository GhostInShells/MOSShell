"""
Features command group — AI-native feature tracking via file system convention.
"""
from pathlib import Path
from datetime import date
from typing import Optional

import typer

from ghoshell_moss.core.codex._features import (
    list_features,
    get_feature,
    create_feature,
    update_feature_status,
    archive_feature,
    init_features,
    VALID_STATUSES,
)
from ghoshell_moss.cli.utils import (
    print_success, print_error, print_info, print_warning,
    print_simple_table, print_simple_panel, console, echo,
)

features_app = typer.Typer(
    short_help="AI-native feature tracking via file system convention.",
    help="AI-native feature tracking via file system convention.",
    no_args_is_help=True,
)

# Default features directory for the MOSShell project itself
_DEFAULT_FEATURES_DIR = Path.cwd() / ".ai_partners" / "features"


def _resolve_dir(features_dir: Optional[Path]) -> Path:
    if features_dir is not None:
        return features_dir
    return _DEFAULT_FEATURES_DIR


# ---------------------------------------------------------------------------
# specification
# ---------------------------------------------------------------------------

@features_app.command("specification")
def specification(
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to .ai_partners/features/ directory. Defaults to current project.",
    ),
):
    """
    Display the AI-Native Feature Tracking convention specification (README.md).
    """
    fd = _resolve_dir(features_dir)
    readme = fd / "README.md"
    if not readme.is_file():
        print_error(f"Specification not found: {readme}")
        print_info("Run 'moss features init' to create the features skeleton first.")
        raise typer.Exit(code=1)
    echo(readme.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@features_app.command("list")
def list_cmd(
    status: Optional[str] = typer.Option(
        None, "--status", "-s",
        help="Filter by status: draft, in-progress, completed, abandoned, blocked",
    ),
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to .ai_partners/features/ directory. Defaults to current project.",
    ),
):
    """
    List all active features with their status and priority.
    """
    fd = _resolve_dir(features_dir)
    if status and status not in VALID_STATUSES:
        print_error(f"Invalid status '{status}'. Valid: {', '.join(sorted(VALID_STATUSES))}")
        raise typer.Exit(code=1)

    features = list_features(str(fd), status_filter=status)

    if not features:
        print_info("No features found.")
        return

    table_data = []
    for fm in features:
        sid = fm.get("id", "?")
        stat = fm.get("status", "?")
        pri = fm.get("priority", "?")
        title = fm.get("title", sid)
        desc = fm.get("description", "")
        updated = fm.get("updated", "")
        feat_dir = fm.get("_feature_dir", sid)
        fm_path = str(fd / "active" / feat_dir / "FEATURE.md")

        status_display = stat
        if stat == "in-progress":
            status_display = f"[bold green]{stat}[/bold green]"
        elif stat == "blocked":
            status_display = f"[bold red]{stat}[/bold red]"
        elif stat == "draft":
            status_display = f"[dim]{stat}[/dim]"

        table_data.append([sid, title, status_display, pri, updated, desc, fm_path])

    title = "Features"
    if status:
        title += f" [status={status}]"

    print_simple_table(
        data=table_data,
        headers=["ID", "Title", "Status", "Priority", "Updated", "Description", "Path"],
        title=title,
        column_ratios=[1, 1.2, 0.6, 0.3, 0.5, 2, 3],
    )
    console.print(f"\n[dim]{len(features)} feature(s)[/dim]")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@features_app.command("status")
def status_cmd(
    feature_id: Optional[str] = typer.Argument(None, help="Feature ID to show. Omit to show all."),
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to .ai_partners/features/ directory. Defaults to current project.",
    ),
):
    """
    Show detailed status of one or all active features.
    """
    fd = _resolve_dir(features_dir)

    if feature_id:
        meta = get_feature(str(fd), feature_id)
        if meta is None:
            print_error(f"Feature '{feature_id}' not found.")
            raise typer.Exit(code=1)

        lines = [f"ID:          {meta.get('id', feature_id)}",
                 f"Title:       {meta.get('title', '')}",
                 f"Status:      {meta.get('status', '')}",
                 f"Priority:    {meta.get('priority', '')}",
                 f"Created:     {meta.get('created', '')}",
                 f"Updated:     {meta.get('updated', '')}",
                 f"Depends:     {', '.join(meta.get('depends', [])) or 'none'}",
                 f"Milestone:   {meta.get('milestone', '') or 'none'}",
                 f"Description: {meta.get('description', '')}",
                 f"Path:        {fd / 'active' / feature_id / 'FEATURE.md'}"]
        print_simple_panel("\n".join(lines), title=f"Feature: {feature_id}")
    else:
        # Show all — delegate to list display
        features = list_features(str(fd))
        if not features:
            print_info("No features found.")
            return

        for fm in features:
            sid = fm.get("id", "?")
            fm_path = fd / "active" / sid / "FEATURE.md"
            lines = [
                f"Status:      {fm.get('status', '?')}",
                f"Priority:    {fm.get('priority', '?')}",
                f"Updated:     {fm.get('updated', '?')}",
                f"Description: {fm.get('description', '')}",
                f"Path:        {fm_path}",
            ]
            print_simple_panel("\n".join(lines), title=f"{sid}: {fm.get('title', '')}")


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

@features_app.command("create")
def create_cmd(
    name: str = typer.Argument(..., help="Feature name in kebab-case."),
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to .ai_partners/features/ directory. Defaults to current project.",
    ),
):
    """
    Create a new feature from the TEMPLATE.md.
    """
    fd = _resolve_dir(features_dir)
    template = fd / "TEMPLATE.md"

    try:
        fm_path = create_feature(str(fd), name, template_path=template if template.is_file() else None)
        print_success(f"Feature '{name}' created: {fm_path}")
    except FileExistsError:
        print_error(f"Feature '{name}' already exists.")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# archive
# ---------------------------------------------------------------------------

@features_app.command("archive")
def archive_cmd(
    feature_id: str = typer.Argument(..., help="Feature ID to archive."),
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to .ai_partners/features/ directory. Defaults to current project.",
    ),
):
    """
    Archive a completed or abandoned feature.

    Moves the feature directory to archived/<year>/<month>/<name>/
    and appends to the archive index.
    """
    fd = _resolve_dir(features_dir)
    meta = get_feature(str(fd), feature_id)

    if meta is None:
        print_error(f"Feature '{feature_id}' not found.")
        raise typer.Exit(code=1)

    status = meta.get("status", "")
    if status not in ("completed", "abandoned"):
        print_error(
            f"Feature '{feature_id}' has status '{status}'. "
            "Must be 'completed' or 'abandoned' to archive."
        )
        raise typer.Exit(code=1)

    ok = archive_feature(str(fd), feature_id)
    if ok:
        print_success(f"Feature '{feature_id}' archived.")
    else:
        print_error(f"Failed to archive feature '{feature_id}'.")


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@features_app.command("init")
def init_cmd(
    project_root: Optional[Path] = typer.Option(
        None, "--project", "-p",
        help="Project root directory. Defaults to current working directory.",
    ),
):
    """
    Initialize the .ai_partners/features/ skeleton in a project.

    Creates the directory structure with README.md and TEMPLATE.md.
    """
    root = project_root or Path.cwd()
    fd = init_features(str(root))
    print_success(f"Features skeleton created: {fd}")
    print_info("Next steps:")
    print_info(f"  1. Edit {fd / 'README.md'} to customize the convention")
    print_info(f"  2. Run 'moss features create <name>' to create your first feature")
