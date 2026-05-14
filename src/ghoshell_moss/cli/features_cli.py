"""
Features command group — AI-native development tracking via file system convention.

Tracks active development workstreams, their decision history, and completion state.
This is NOT a project capability catalog — it tracks what's being built right now.
"""
from pathlib import Path

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
    print_simple_table, print_simple_panel, console, echo, is_ai_mode,
)

features_app = typer.Typer(
    short_help="AI-native development tracking via file system convention.",
    help="AI-native development tracking via file system convention. Tracks active workstreams, not a project capability catalog.",
    no_args_is_help=True,
)

# Next-step hints per status transition
_STATUS_HINTS = {
    ("draft", "in-progress"): "Record key decisions in FEATURE.md as you implement.",
    ("in-progress", "completed"): "Verify all key decisions are recorded. The workstream stays at its creation path.",
    ("in-progress", "blocked"): "Update depends: in frontmatter if a specific workstream is blocking this one.",
    ("in-progress", "draft"): "Update the Motivation section if context has changed.",
}
_ABANDONED_HINT = "Record why in -m 'reason' for future reference. The workstream stays in place."

# Default features directory for the MOSShell project itself
_DEFAULT_FEATURES_DIR = Path.cwd() / ".ai_partners" / "features"


def _resolve_dir(features_dir: Optional[Path]) -> Path:
    if features_dir is not None:
        return features_dir
    return _DEFAULT_FEATURES_DIR


# ---------------------------------------------------------------------------
# specification
# ---------------------------------------------------------------------------

@features_app.command("specification", short_help="Display the convention specification (README.md).")
def specification(
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to .ai_partners/features/ directory. Defaults to current project.",
    ),
):
    """
    Display the AI-Native Development Tracking convention specification (README.md).
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

@features_app.command("list", short_help="List workstreams with status and priority.")
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
    List active development workstreams with status and priority.

    Defaults to workstreams from the last 2 months. Use --all to see everything.
    """
    fd = _resolve_dir(features_dir)
    if status and status not in VALID_STATUSES:
        print_error(f"Invalid status '{status}'. Valid: {', '.join(sorted(VALID_STATUSES))}")
        raise typer.Exit(code=1)

    features = list_features(str(fd), status_filter=status, all_months=all_months)
    title = "Workstreams"
    if status:
        title += f" [status={status}]"
    if not all_months:
        title += " (last 2 months)"

    if not features:
        print_info("No workstreams found.")
        return

    table_data = []
    for fm in features:
        name = fm.get("_feature_dir", "?")
        stat = fm.get("status", "?")
        pri = fm.get("priority", "?")
        title_str = fm.get("title", name)
        updated = fm.get("updated", "")
        feat_path = fm.get("_feature_path", name)

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

        table_data.append([name, status_display, pri, title_str, updated, feat_path])

    print_simple_table(
        data=table_data,
        headers=["Name", "Status", "Pri", "Title", "Updated", "Path"],
        title=title,
        column_ratios=[1, 0.7, 0.3, 1.5, 0.6, 1.5],
    )
    console.print(f"\n[dim]{len(features)} workstream(s)[/dim]")
    console.print(
        "[dim]New to this? Read the convention: [/dim]"
        "[bold]moss features specification[/bold]"
    )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@features_app.command("status", short_help="Show detailed status of workstreams.")
def status_cmd(
    feature_name: Optional[str] = typer.Argument(None, help="Feature name to show. Omit to show all."),
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to .ai_partners/features/ directory. Defaults to current project.",
    ),
):
    """
    Show detailed status of one or all active workstreams.

    With a name: shows full detail panel for that workstream.
    Without a name: shows a compact summary of all workstreams.
    """
    fd = _resolve_dir(features_dir)

    if feature_name:
        meta = get_feature(str(fd), feature_name)
        if meta is None:
            print_error(f"Workstream '{feature_name}' not found.")
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
                 f"Path:        workstreams/{feat_path}/FEATURE.md"]
        print_simple_panel("\n".join(lines), title=f"Workstream: {feature_name}")
    else:
        # Compact all-workstreams view
        features = list_features(str(fd))
        if not features:
            print_info("No workstreams found.")
            return

        if is_ai_mode():
            echo(f"## Workstreams\n")
            for fm in features:
                name = fm.get("_feature_dir", "?")
                stat = fm.get("status", "?")
                pri = fm.get("priority", "?")
                title_str = fm.get("title", name)
                desc = fm.get("description", "")
                updated = fm.get("updated", "")
                feat_path = fm.get("_feature_path", name)
                note = fm.get("status_note", "")
                echo(f"### {name} [{stat} {pri}] — {title_str}")
                echo(f"  Updated:     {updated}")
                echo(f"  Description: {desc}")
                if note:
                    echo(f"  Status Note: {note}")
                echo(f"  Path:        workstreams/{feat_path}/")
                echo("")
        else:
            console.print()
            for fm in features:
                name = fm.get("_feature_dir", "?")
                stat = fm.get("status", "?")
                pri = fm.get("priority", "?")
                title_str = fm.get("title", name)
                desc = fm.get("description", "")
                updated = fm.get("updated", "")
                feat_path = fm.get("_feature_path", name)
                note = fm.get("status_note", "")

                # Color the status
                if stat == "in-progress":
                    stat_disp = f"[bold green]{stat}[/bold green]"
                elif stat == "blocked":
                    stat_disp = f"[bold red]{stat}[/bold red]"
                elif stat == "draft":
                    stat_disp = f"[dim]{stat}[/dim]"
                elif stat == "completed":
                    stat_disp = f"[bold cyan]{stat}[/bold cyan]"
                elif stat == "abandoned":
                    stat_disp = f"[dim red]{stat}[/dim red]"
                else:
                    stat_disp = stat

                console.print(f"[bold]{name}[/bold] [{stat_disp} {pri}] — {title_str}")
                console.print(f"  Updated:     {updated}")
                console.print(f"  Description: {desc}")
                if note:
                    console.print(f"  Status Note: {note}")
                console.print(f"  Path:        workstreams/{feat_path}/")
                console.print()

        console.print(f"[dim]{len(features)} workstream(s)[/dim]")


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

@features_app.command("create", short_help="Create a new workstream from the template.")
def create_cmd(
    name: str = typer.Argument(..., help="Feature name in kebab-case."),
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to .ai_partners/features/ directory. Defaults to current project.",
    ),
):
    """
    Create a new development workstream from the TEMPLATE.md.
    """
    fd = _resolve_dir(features_dir)
    template = fd / "TEMPLATE.md"

    try:
        fm_path = create_feature(str(fd), name, template_path=template if template.is_file() else None)
        print_success(f"Workstream '{name}' created: {fm_path}")
        # NOTE: 不在此输出模板内容。Claude Code 不允许编辑未读取的文件，
        # 所以"省一次 Read"的设计意图在 Claude Code 约束下无效。
        # 将来整体完善 create 流程时再重新设计。
        # AI 消费者应随后 Read FEATURE.md 获取可编辑内容（含行号）。
    except FileExistsError:
        print_error(f"Workstream '{name}' already exists.")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# set-status
# ---------------------------------------------------------------------------

@features_app.command("set-status", short_help="Set workstream status without opening the file.")
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
    Quick-set the status of a workstream without opening the file.

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
        print_error(f"Workstream '{feature_name}' not found.")
        raise typer.Exit(code=1)

    old_status = meta.get("status", "?")
    old_note = meta.get("status_note", "")
    if old_status == status and (message is None or message == old_note):
        print_info(f"Workstream '{feature_name}' status is already '{status}'.")
        return

    ok = update_feature_status(str(fd), feature_name, status, status_note=message)
    if ok:
        msg = f"Workstream '{feature_name}': {old_status} -> {status}"
        if message:
            msg += f"  ({message})"
        print_success(msg)

        # Print a next-step hint for the transition
        hint = None
        if status == "abandoned":
            hint = _ABANDONED_HINT
        else:
            hint = _STATUS_HINTS.get((old_status, status))
        if hint:
            print_info(hint)
    else:
        print_error(f"Failed to update status for '{feature_name}'.")


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@features_app.command("init", short_help="Initialize the features skeleton in a project.")
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
    print_info(f"  2. Run 'moss features create <name>' to create your first workstream")
