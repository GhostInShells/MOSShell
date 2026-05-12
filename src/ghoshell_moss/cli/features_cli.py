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
    init_features,
    update_feature_status,
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
    echo(f"\nSpecification path: {readme.resolve()}")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@features_app.command("list")
def list_cmd(
    status: Optional[str] = typer.Option(
        None, "--status", "-s",
        help="Filter by status: draft, in-progress, completed, abandoned, blocked",
    ),
    all_months: bool = typer.Option(
        False, "--all",
        help="List features from all time (default: last 2 months only).",
    ),
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to .ai_partners/features/ directory. Defaults to current project.",
    ),
):
    """
    List active features with their status and priority.

    Defaults to features from the last 2 months. Use --all to see everything.
    """
    fd = _resolve_dir(features_dir)
    if status and status not in VALID_STATUSES:
        print_error(f"Invalid status '{status}'. Valid: {', '.join(sorted(VALID_STATUSES))}")
        raise typer.Exit(code=1)

    features = list_features(str(fd), status_filter=status, all_months=all_months)
    title = "Features"
    if status:
        title += f" [status={status}]"
    if not all_months:
        title += " (last 2 months)"

    if not features:
        print_info("No features found.")
        return

    table_data = []
    for fm in features:
        name = fm.get("_feature_dir", "?")
        stat = fm.get("status", "?")
        pri = fm.get("priority", "?")
        title_str = fm.get("title", name)
        desc = fm.get("description", "")
        updated = fm.get("updated", "")
        feat_path = fm.get("_feature_path", name)
        fm_path = str(fd / "active" / feat_path / "FEATURE.md")

        status_display = stat
        if stat == "in-progress":
            status_display = f"[bold green]{stat}[/bold green]"
        elif stat == "blocked":
            status_display = f"[bold red]{stat}[/bold red]"
        elif stat == "draft":
            status_display = f"[dim]{stat}[/dim]"
        elif stat == "completed":
            status_display = f"[bold cyan]{stat}[/bold cyan]"
        elif stat == "abandoned":
            status_display = f"[dim red]{stat}[/dim red]"

        table_data.append([name, title_str, status_display, pri, updated, desc, fm_path])

    print_simple_table(
        data=table_data,
        headers=["Name", "Title", "Status", "Priority", "Updated", "Description", "Path"],
        title=title,
        column_ratios=[1, 1.2, 0.6, 0.3, 0.5, 2, 3],
    )
    console.print(f"\n[dim]{len(features)} feature(s)[/dim]")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@features_app.command("status")
def status_cmd(
    feature_name: Optional[str] = typer.Argument(None, help="Feature name to show. Omit to show all."),
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to .ai_partners/features/ directory. Defaults to current project.",
    ),
):
    """
    Show detailed status of one or all active features.
    """
    fd = _resolve_dir(features_dir)

    if feature_name:
        meta = get_feature(str(fd), feature_name)
        if meta is None:
            print_error(f"Feature '{feature_name}' not found.")
            raise typer.Exit(code=1)

        feat_path = meta.get("_feature_path", feature_name)
        lines = [f"Name:        {feature_name}",
                 f"Title:       {meta.get('title', '')}",
                 f"Status:      {meta.get('status', '')}",
                 f"Priority:    {meta.get('priority', '')}",
                 f"Created:     {meta.get('created', '')}",
                 f"Updated:     {meta.get('updated', '')}",
                 f"Depends:     {', '.join(meta.get('depends', [])) or 'none'}",
                 f"Milestone:   {meta.get('milestone', '') or 'none'}",
                 f"Description: {meta.get('description', '')}",
                 f"Status Note: {meta.get('status_note', '') or 'none'}",
                 f"Path:        {fd / 'active' / feat_path / 'FEATURE.md'}"]
        print_simple_panel("\n".join(lines), title=f"Feature: {feature_name}")
    else:
        # Show all — delegate to list display
        features = list_features(str(fd))
        if not features:
            print_info("No features found.")
            return

        for fm in features:
            name = fm.get("_feature_dir", "?")
            feat_path = fm.get("_feature_path", name)
            fm_path = fd / "active" / feat_path / "FEATURE.md"
            lines = [
                f"Status:      {fm.get('status', '?')}",
                f"Priority:    {fm.get('priority', '?')}",
                f"Updated:     {fm.get('updated', '?')}",
                f"Description: {fm.get('description', '')}",
            ]
            note = fm.get("status_note", "")
            if note:
                lines.append(f"Status Note: {note}")
            lines.append(f"Path:        {fm_path}")
            print_simple_panel("\n".join(lines), title=f"{name}: {fm.get('title', '')}")


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
        print_info("If you haven't read the features convention, run: moss features specification")
    except FileExistsError:
        print_error(f"Feature '{name}' already exists.")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# set-status
# ---------------------------------------------------------------------------

@features_app.command("set-status")
def set_status_cmd(
    feature_name: str = typer.Argument(..., help="Feature name to update."),
    status: str = typer.Argument(..., help=f"New status: {', '.join(sorted(VALID_STATUSES))}"),
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to .ai_partners/features/ directory. Defaults to current project.",
    ),
    message: Optional[str] = typer.Option(
        None, "--message", "-m",
        help="One-line context note explaining the current status (e.g. why blocked, what's next).",
    ),
):
    """
    Quick-set the status of a feature without opening the file.

    Updates the 'status' and 'updated' fields in the YAML frontmatter.
    Use -m to attach a one-line status_note for context (e.g. why blocked, what's next).

    Faster than manually editing FEATURE.md — one shell call vs Read+Edit.
    """
    fd = _resolve_dir(features_dir)

    if status not in VALID_STATUSES:
        print_error(f"Invalid status '{status}'. Valid: {', '.join(sorted(VALID_STATUSES))}")
        raise typer.Exit(code=1)

    meta = get_feature(str(fd), feature_name)
    if meta is None:
        print_error(f"Feature '{feature_name}' not found.")
        raise typer.Exit(code=1)

    old_status = meta.get("status", "?")
    old_note = meta.get("status_note", "")
    if old_status == status and (message is None or message == old_note):
        print_info(f"Feature '{feature_name}' status is already '{status}'.")
        return

    ok = update_feature_status(str(fd), feature_name, status, status_note=message)
    if ok:
        msg = f"Feature '{feature_name}': {old_status} -> {status}"
        if message:
            msg += f"  ({message})"
        print_success(msg)
    else:
        print_error(f"Failed to update status for '{feature_name}'.")


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
