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
    RESERVED_STATUSES,
    _find_templates_dir,
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
    ("in-progress", "completed"): "Now commit this FEATURE.md with your code in the same commit — status change must land together with the code, not after.",
    ("in-progress", "draft"): "Update the Motivation section if context has changed.",
}
_DROPPED_HINT = "Record why in -m 'reason' for future reference. The workstream stays in place."

# Canonical display order for status stats; free-form statuses aggregate under 'others'.
_STATUS_ORDER = ["draft", "in-progress", "completed", "dropped"]


def _status_label(stat: str) -> str:
    """Rich-markup status label, matching the list table coloring."""
    if stat == "in-progress":
        return f"[bold green]{stat}[/bold green]"
    if stat == "draft":
        return f"[dim]{stat}[/dim]"
    if stat == "completed":
        return f"[bold cyan]{stat}[/bold cyan]"
    if stat == "dropped":
        return f"[dim red]{stat}[/dim red]"
    return stat


def _status_counts(features: list[dict]) -> dict[str, int]:
    """Status distribution — reserved statuses keyed individually, free-form aggregated under 'others'."""
    counts: dict[str, int] = {}
    others = 0
    for fm in features:
        st = fm.get("status", "?")
        if st in RESERVED_STATUSES:
            counts[st] = counts.get(st, 0) + 1
        else:
            others += 1
    if others:
        counts["others"] = others
    return counts


def _ordered_statuses(*count_sets: dict[str, int]) -> list[str]:
    """Union of statuses across count sets, reserved first then free-form alphabetically."""
    seen: set[str] = set()
    for counts in count_sets:
        seen.update(counts)
    return [s for s in _STATUS_ORDER if s in seen] + sorted(seen - set(_STATUS_ORDER))


def _print_parse_errors(parse_errors: list[dict]) -> None:
    """Print FEATURE.md frontmatter parse errors — called at end of list/status output."""
    echo("")
    print_warning(f"Parse errors — {len(parse_errors)} FEATURE.md file(s) have broken YAML frontmatter "
                  f"and were excluded from the list:")
    echo("")
    for err in parse_errors:
        echo(f"  [{err['feature_dir']}]")
        echo(f"    path:  {err['path']}")
        echo(f"    error: {err['error']}")
        echo(f"    hint:  {err['hint']}")
        echo("")
    echo("  Fix: open the file and correct the YAML frontmatter between the --- markers.")
    echo("")


# Default features directory for the MOSShell project itself
_DEFAULT_FEATURES_DIR = Path.cwd() / ".ai_partners" / "features"


def _resolve_dir(features_dir: Optional[Path]) -> Path:
    if features_dir is not None:
        return features_dir
    return _DEFAULT_FEATURES_DIR


# ---------------------------------------------------------------------------
# specification
# ---------------------------------------------------------------------------

@features_app.command("specification", short_help="Display the features convention — read this first.")
def specification(
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to .ai_partners/features/ directory. Defaults to current project.",
    ),
):
    """
    Display the AI-Native Development Tracking convention specification.

    Reads from the local .ai_partners/features/ copy first.
    Falls back to the bundled canonical copy shipped with the package.
    """
    fd = _resolve_dir(features_dir)
    readme = fd / "README.md"

    if not readme.is_file():
        templates = _find_templates_dir()
        if templates and (bundled := templates / "README.md").is_file():
            echo(bundled.read_text(encoding="utf-8"))
            echo(f"\nSpecification path: {bundled.resolve()}")
            print_info("Shown from bundled copy. Run 'moss features init' to create a local one.")
            return
        print_error(f"Specification not found: {readme}")
        print_info("Run 'moss features init' to create the features skeleton first.")
        raise typer.Exit(code=1)

    echo(readme.read_text(encoding="utf-8"))
    echo(f"\nSpecification path: {readme.resolve()}")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@features_app.command("list", short_help="List active workstreams with status and priority.")
def list_cmd(
    status: Optional[str] = typer.Option(
        None, "--status", "-s",
        help="Filter by status. Reserved: draft, in-progress, completed, dropped; free-form values match exactly.",
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
    features, parse_errors = list_features(str(fd), status_filter=status, all_months=all_months)
    # All-time (status-filtered) view — feeds the stats table and the empty-window hint.
    total_features, _ = list_features(str(fd), status_filter=status, all_months=True)
    title = "Workstreams"
    if status:
        title += f" [status={status}]"
    if not all_months:
        title += " (last 2 months)"

    if not features and not parse_errors:
        if not all_months and total_features:
            print_info(f"No workstreams in the last 2 months — {len(total_features)} all-time (use --all).")
        else:
            print_info("No workstreams found.")
        return

    table_data = []
    for fm in features:
        name = fm.get("_feature_dir", "?")
        stat = fm.get("status", "?")
        pri = fm.get("priority", "?")
        title_str = fm.get("title", name)
        updated = fm.get("updated", "")
        feat_path = f"workstreams/{fm.get('_feature_path', name)}"

        status_display = _status_label(stat)

        table_data.append([name, status_display, pri, title_str, updated, feat_path])

    print_simple_table(
        data=table_data,
        headers=["Name", "Status", "Pri", "Title", "Updated", "Path"],
        title=title,
        column_ratios=[1, 0.7, 0.3, 1.5, 0.6, 1.5],
    )

    console.print(
        "[dim]Pri = importance within the current stage (iteration cycle), not development urgency.[/dim]"
    )

    if features or total_features:
        listed_counts = _status_counts(features)
        total_counts = _status_counts(total_features)
        statuses = _ordered_statuses(listed_counts, total_counts)
        stats_headers = [""] + statuses + ["Total"]
        if listed_counts == total_counts:
            stats_rows = [["total"] + [str(total_counts.get(s, 0)) for s in statuses] + [str(len(total_features))]]
        else:
            stats_rows = [
                ["listed"] + [str(listed_counts.get(s, 0)) for s in statuses] + [str(len(features))],
                ["total"] + [str(total_counts.get(s, 0)) for s in statuses] + [str(len(total_features))],
            ]
        print_simple_table(
            data=stats_rows,
            headers=stats_headers,
            title="Workstream counts",
            column_styles=["bold"] + [None] * len(statuses) + ["bold"],
        )
        console.print("")

    console.print(f"\n[dim]Features root: {fd.resolve()}/[/dim]")
    console.print(
        "[dim]Workstreams are one part of [/dim]"
        "[bold].ai_partners/features[/bold]"
        "[dim]; read the convention: [/dim]"
        "[bold]moss features specification[/bold]"
    )

    if parse_errors:
        _print_parse_errors(parse_errors)


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
        meta, parse_err = get_feature(str(fd), feature_name)
        if parse_err is not None:
            print_error(f"Workstream '{feature_name}' exists but FEATURE.md has a YAML parse error:")
            _print_parse_errors([parse_err])
            raise typer.Exit(code=1)
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
                 f"Path:        .ai_partners/features/workstreams/{feat_path}/FEATURE.md"]
        print_simple_panel("\n".join(lines), title=f"Workstream: {feature_name}")
        console.print(
            "[dim]Read the convention: [/dim]"
            "[bold]moss features specification[/bold]"
        )
    else:
        # Compact all-workstreams view
        features, parse_errors = list_features(str(fd))
        if not features and not parse_errors:
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
                echo(f"  Path:        .ai_partners/features/workstreams/{feat_path}/")
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
                elif stat == "draft":
                    stat_disp = f"[dim]{stat}[/dim]"
                elif stat == "completed":
                    stat_disp = f"[bold cyan]{stat}[/bold cyan]"
                elif stat == "dropped":
                    stat_disp = f"[dim red]{stat}[/dim red]"
                else:
                    stat_disp = stat

                console.print(f"[bold]{name}[/bold] [{stat_disp} {pri}] — {title_str}")
                console.print(f"  Updated:     {updated}")
                console.print(f"  Description: {desc}")
                if note:
                    console.print(f"  Status Note: {note}")
                console.print(f"  Path:        .ai_partners/features/workstreams/{feat_path}/")
                console.print()

        console.print(
            "[dim]Read the convention: [/dim]"
            "[bold]moss features specification[/bold]"
        )

        if parse_errors:
            _print_parse_errors(parse_errors)


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

@features_app.command("create", short_help="Create a new workstream from the template.")
def create_cmd(
    name: str = typer.Argument(..., help="Feature name in kebab-case."),
    features_dir: Optional[Path] = typer.Option(
        None, "--dir",
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
        print_info("Read the convention: moss features specification")
        print_info(f"Next: edit {fm_path} to record motivation and key decisions.")
    except FileExistsError:
        print_error(f"Workstream '{name}' already exists.")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# set-status
# ---------------------------------------------------------------------------

@features_app.command("set-status", short_help="Set workstream status without opening the file.")
def set_status_cmd(
    feature_name: str = typer.Argument(..., help="Feature name to update."),
    status: str = typer.Argument(..., help=f"New status. Reserved: {', '.join(sorted(RESERVED_STATUSES))}; free-form allowed."),
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to .ai_partners/features/ directory. Defaults to current project.",
    ),
    message: Optional[str] = typer.Option(
        None, "--message", "-m",
        help="One-line context note explaining the current status (e.g. why dropped, what's next).",
    ),
):
    """
    Quick-set the status of a workstream without opening the file.

    Updates the 'status' and 'updated' fields in the YAML frontmatter.
    Use -m to attach a one-line status_note for context (e.g. why dropped, what's next).

    Faster than manually editing FEATURE.md — one shell call vs Read+Edit.
    """
    fd = _resolve_dir(features_dir)

    if status not in RESERVED_STATUSES:
        print_warning(f"'{status}' is not a reserved status ({', '.join(sorted(RESERVED_STATUSES))}). "
                      f"Free-form statuses are allowed; reserved values drive list coloring and check.")

    meta, parse_err = get_feature(str(fd), feature_name)
    if parse_err is not None:
        print_error(f"Workstream '{feature_name}' exists but FEATURE.md has a YAML parse error:")
        _print_parse_errors([parse_err])
        raise typer.Exit(code=1)
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
        if status == "dropped":
            hint = _DROPPED_HINT
        else:
            hint = _STATUS_HINTS.get((old_status, status))
        if hint:
            print_info(hint)
        print_info("This only updates status. For key decisions, design, or motivation, edit FEATURE.md directly.")
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
    print_success(f"Features templates synced to: {fd}")
    print_info("Template files overwritten; existing workstreams left untouched.")


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

_TERMINAL_STATUSES = {"completed", "dropped"}


@features_app.command("check", short_help="List unfinished workstreams — pre-commit reminder.")
def check_cmd(
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to .ai_partners/features/ directory. Defaults to current project.",
    ),
):
    """
    List workstreams that are NOT in a terminal state (completed/dropped).

    Intended as a non-blocking pre-commit hook — always exits 0.
    If you're committing code for any listed feature, run:

        moss features set-status <name> completed

    before committing.
    """
    fd = _resolve_dir(features_dir)
    features, parse_errors = list_features(str(fd))

    if parse_errors:
        _print_parse_errors(parse_errors)

    unfinished = [f for f in features if f.get("status") not in _TERMINAL_STATUSES]
    if not unfinished:
        return

    # Group by status
    grouped: dict[str, list[dict]] = {}
    for f in unfinished:
        stat = f.get("status", "?")
        grouped.setdefault(stat, []).append(f)

    echo("")
    print_warning("Unfinished workstreams — if committing code for any, set status first:")
    echo("")

    for stat in sorted(grouped.keys()):
        items = grouped[stat]
        label = f"{stat} ({len(items)})"
        echo(f"  {label}:")
        for f in items:
            name = f.get("_feature_dir", "?")
            pri = f.get("priority", "?")
            title = f.get("title", name)
            echo(f"    {name:<36s} {pri}  {title}")
        echo("")

    echo("  moss features set-status <name> completed")
    echo("")
