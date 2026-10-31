"""
Features command group — model-native development via file system convention.

Tracks workstreams (L1), stages (L2), regressions, and the project surface (L3).
"""
from pathlib import Path

from typing import Optional

import typer

from ghoshell_moss.core.codex.features import (
    list_features,
    get_feature,
    locate_feature,
    count_statuses,
    is_terminal,
    discover_perspectives,
    resolve_features_dir,
    create_feature,
    init_features,
    update_feature_status,
    list_stages,
    list_regressions,
    RESERVED_STATUSES,
    RESERVED_STATUS_ORDER,
    WorkstreamMeta,
    ParseError,
    ISSUES_CONVENTION,
    README_FILENAME,
    SURFACE_FILENAME,
    find_stubs_dir,
)
from ghoshell_moss.cli.utils import (
    print_success, print_error, print_info, print_warning,
    print_simple_table, print_simple_panel, console, echo, is_ai_mode,
)

features_app = typer.Typer(
    short_help="Model-native development system via file system convention.",
    help="Model-native development system via file system convention — workstreams, stages, regressions, and a project surface.",
    no_args_is_help=True,
)

# Next-step hints per status transition
_STATUS_HINTS = {
    ("draft", "in-progress"): "Record key decisions in FEATURE.md as you implement.",
    ("in-progress", "completed"): "Now commit this FEATURE.md with your code in the same commit — status change must land together with the code, not after.",
    ("in-progress", "draft"): "Update the Motivation section if context has changed.",
}
_DROPPED_HINT = "Record why in -m 'reason' for future reference. The workstream stays in place."
_PARKED_HINT = ("Record why in -m 'reason' and what would reopen it. Parked workstreams are hidden "
                "from 'list' and 'check'; retrieve with: moss features list --status parked")


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
    if stat == "parked":
        return f"[dim yellow]{stat}[/dim yellow]"
    return stat


def _ordered_statuses(*count_sets: dict[str, int]) -> list[str]:
    """Union of statuses across count sets, reserved first then free-form alphabetically."""
    seen: set[str] = set()
    for counts in count_sets:
        seen.update(counts)
    return [s for s in RESERVED_STATUS_ORDER if s in seen] + sorted(seen - set(RESERVED_STATUS_ORDER))


def _print_parse_errors(parse_errors: list[ParseError]) -> None:
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


# ---------------------------------------------------------------------------
# specification
# ---------------------------------------------------------------------------

@features_app.command("specification", short_help="Display the features convention — read this first.")
def specification(
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to the features/ directory. Defaults to $CWD/features (or $CWD/.ai_partners/features).",
    ),
):
    """
    Display the features system convention specification.

    Reads from the local features/ copy first.
    Falls back to the bundled canonical copy shipped with the package.
    """
    fd = resolve_features_dir(features_dir)
    readme = fd / README_FILENAME

    if not readme.is_file():
        stubs = find_stubs_dir()
        if stubs and (bundled := stubs / README_FILENAME).is_file():
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
# issues
# ---------------------------------------------------------------------------

@features_app.command("issues", short_help="Read this project's issue convention (issues/SKILL.md).")
def issues_cmd(
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to the features/ directory. Defaults to $CWD/features (or $CWD/.ai_partners/features).",
    ),
):
    """
    Print the project's issue convention.

    The features system governs how work evolves; issues are the outside world
    knocking. This command is only an entry point — it reads and prints
    `<features>/issues/SKILL.md` and does nothing else. Where issues actually
    live (GitHub, GitLab, elsewhere), which tool reads them, and who may open or
    close one are the project's business, stated in that file.

    A project without the file is not an error: it simply has not stated a
    convention yet, and the message says how to write one.
    """
    fd = resolve_features_dir(features_dir)
    path = fd / ISSUES_CONVENTION

    if not path.is_file():
        print_info(f"No issue convention in this project — {path} does not exist.")
        echo("")
        echo("  Issues are not part of the features data model: features track how work")
        echo("  evolves, issues arrive from outside. This command only surfaces the")
        echo("  convention. A project that wants one states it in:")
        echo(f"    {path}")
        echo("")
        echo("  That file should answer, for a reader with no context:")
        echo("    - where issues live and which tool reads them (gh, glab, anything)")
        echo("    - the state machine, and what each state means")
        echo("    - label rules, and which of them are firm vs provisional")
        echo("    - who may open, comment on, and close one")
        echo("    - what a closing receipt must contain")
        return

    echo(path.read_text(encoding="utf-8"))
    echo(f"\nConvention path: {path.resolve()}")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@features_app.command("list", short_help="List active workstreams with status and priority.")
def list_cmd(
    status: Optional[str] = typer.Option(
        None, "--status", "-s",
        help="Filter by status. Reserved: draft, in-progress, completed, dropped, parked; free-form values match exactly.",
    ),
    all_months: bool = typer.Option(
        False, "--all",
        help="Widen to all time (default: last 60 days only). Parked stay hidden — use --status parked.",
    ),
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to the features/ directory. Defaults to $CWD/features (or $CWD/.ai_partners/features).",
    ),
):
    """
    List active development workstreams with status and priority.

    Defaults to workstreams touched in the last 60 days. Use --all to widen the
    time window. Parked workstreams stay hidden either way — ask for them
    explicitly with --status parked.
    """
    fd = resolve_features_dir(features_dir)
    features, parse_errors = list_features(str(fd), status_filter=status, all_months=all_months)
    # All-time (status-filtered) view — feeds the stats table and the empty-window hint.
    total_features, _ = list_features(str(fd), status_filter=status, all_months=True)
    title = "Workstreams"
    if status:
        title += f" [status={status}]"
    if not all_months:
        title += " (last 60 days)"

    if not features and not parse_errors:
        if not all_months and total_features:
            print_info(f"No workstreams in the last 60 days — {len(total_features)} all-time (use --all).")
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

    console.print(
        "[dim]Pri = delivery weight within the current development stage, not a time-based priority.[/dim]"
    )

    print_simple_table(
        data=table_data,
        headers=["Name", "Status", "Pri", "Title", "Updated", "Path"],
        title=title,
        column_ratios=[1, 0.7, 0.3, 1.5, 0.6, 1.5],
    )

    if features or total_features:
        listed_counts = count_statuses(features)
        total_counts = count_statuses(total_features)
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
        "[dim]Workstreams are one axis of the features system "
        "(workstreams / stages / regressions / surface); read the convention: [/dim]"
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
        help="Path to the features/ directory. Defaults to $CWD/features (or $CWD/.ai_partners/features).",
    ),
):
    """
    Show detailed status of one or all active workstreams.

    With a name: shows full detail panel for that workstream.
    Without a name: shows a compact summary of all workstreams.
    """
    fd = resolve_features_dir(features_dir)

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
                 f"Path:        workstreams/{feat_path}/FEATURE.md"]
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
                console.print(f"  Path:        workstreams/{feat_path}/")
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
        help="Path to the features/ directory. Defaults to $CWD/features (or $CWD/.ai_partners/features).",
    ),
):
    """
    Create a new development workstream from the TEMPLATE.md.
    """
    fd = resolve_features_dir(features_dir)

    try:
        fm_path = create_feature(str(fd), name)
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
        help="Path to the features/ directory. Defaults to $CWD/features (or $CWD/.ai_partners/features).",
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
    fd = resolve_features_dir(features_dir)

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
        elif status == "parked":
            hint = _PARKED_HINT
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
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Target features directory. Defaults to $CWD/features.",
    ),
):
    """
    Initialize the features/ skeleton: copy the bundled stubs (full overwrite)
    and create the workstreams/ directory. Defaults to $CWD/features.
    """
    fd = init_features(features_dir)
    print_success(f"Features skeleton synced to: {fd}")
    print_info("Stub files overwritten; existing workstreams left untouched.")


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------

@features_app.command("check", short_help="List unfinished workstreams — pre-commit reminder.")
def check_cmd(
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to the features/ directory. Defaults to $CWD/features (or $CWD/.ai_partners/features).",
    ),
):
    """
    List workstreams that are NOT in a terminal state (completed/dropped).

    Parked workstreams never appear here — the query already drops quiet
    statuses, so a parked proposal raises no pre-commit reminder.

    Intended as a non-blocking pre-commit hook — always exits 0.
    If you're committing code for any listed feature, run:

        moss features set-status <name> completed

    before committing.
    """
    fd = resolve_features_dir(features_dir)
    features, parse_errors = list_features(str(fd))

    if parse_errors:
        _print_parse_errors(parse_errors)

    unfinished = [f for f in features if not is_terminal(f.get("status"))]
    if not unfinished:
        return

    # Group by status
    grouped: dict[str, list[WorkstreamMeta]] = {}
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


# ---------------------------------------------------------------------------
# review
# ---------------------------------------------------------------------------

def _feature_basic_info(meta: WorkstreamMeta, fm_path: Path) -> list[str]:
    """Feature basic info — frontmatter only, not the FEATURE.md body (zero-context)."""
    return [
        f"- Name:       {meta.get('_feature_dir', '?')}",
        f"- Title:      {meta.get('title', '')}",
        f"- Status:     {meta.get('status', '?')} ({meta.get('priority', '?')})",
        f"- Milestone:  {meta.get('milestone', '') or 'none'}",
        f"- Depends:    {', '.join(meta.get('depends', [])) or 'none'}",
        f"- Description: {meta.get('description', '') or 'none'}",
        f"- FEATURE.md: {fm_path}",
    ]


@features_app.command("review", short_help="Generate a zero-context feature review prompt.")
def review_cmd(
    feature: str = typer.Argument(..., help="Feature name or FEATURE.md path. Optional '<name>@<perspective>'."),
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to the features/ directory. Defaults to $CWD/features (or $CWD/.ai_partners/features).",
    ),
):
    """
    Generate a zero-context feature review prompt (forgetting test).

    Without '@': the meta prompt — top-level review instruction + file discovery
    (matched FEATURE.md + the review perspectives found) + how to dispatch a review.

    With '<name>@<perspective>': the perspective prompt — feature basic info, the
    perspective doc (path + content), and a report-back note.

    Emit a command for a fresh sub-agent to run (it has the CLI), not a finished
    brief: the sub-agent executes `moss features review '<feature>@<perspective>'`,
    reads FEATURE.md and code from a zero-context standpoint, and reports back.
    """
    if "@" in feature:
        feature_part, perspective = feature.rsplit("@", 1)
        perspective = perspective.strip()
    else:
        feature_part, perspective = feature, None

    source, err = locate_feature(feature_part, features_dir)
    if err is not None:
        print_error(f"Workstream '{feature_part}' exists but FEATURE.md has a YAML parse error:")
        _print_parse_errors([err])
        raise typer.Exit(code=1)
    if source is None:
        print_error(f"Workstream '{feature_part}' not found.")
        print_info("Expected a workstream name, a feature directory, or a FEATURE.md path.")
        raise typer.Exit(code=1)

    meta, fm_path = source["meta"], source["fm_path"]
    perspectives, broken = discover_perspectives(source["features_dir"], meta["_feature_path"])

    if perspective is None:
        lines = [
            "# Zero-Context Feature Review — meta prompt",
            "",
            "You are about to review a feature from a zero-context standpoint "
            "(the forgetting test). A fresh reviewer does NOT know the intent the",
            "developer model held — so it can see gaps that intent blinded it to.",
            "",
            "## Feature to review",
            *_feature_basic_info(meta, fm_path),
            "",
            "## Review principle",
            "Read the declaration (FEATURE.md) and the delivery (code) from scratch.",
            "Stand is reconstruct, not audit: 'I can't start here' IS a failure; a",
            "declaration promising X while the code does Y — or never got to X — is drift.",
            "",
            "## Perspectives available",
        ]
        if perspectives:
            for p in perspectives:
                lines.append(f"- {p['name']} (when={p['when']}) — {p['description']}  [{p['path']}]")
            lines += [
                "",
                "To do a perspective review, spawn a fresh sub-agent and tell it to run:",
                f"    moss features review '{feature_part}@<perspective>'",
                "That command emits the sub-agent's brief; it reads the FEATURE.md and code itself.",
            ]
        else:
            lines.append("none — no review doc found under features/review/ or <feature>/review/.")
        if broken:
            lines += ["", "Broken review doc(s) skipped (unparseable or missing when/description):"]
            for p in broken:
                lines.append(f"  - {p}")
        lines += [
            "",
            "Zero-context discipline: if a sub-agent is available, do NOT read the",
            "FEATURE.md or code yourself first — stay out of the material, hand the",
            "review to a fresh sub-agent, and only read it after the sub-agent reports back.",
            "",
            "You are not required to use the CLI. You may hand a sub-agent your own",
            "review instruction instead — the point is a fresh, zero-context reviewer.",
        ]
        echo("\n".join(lines))
        return

    persp = next((p for p in perspectives if p["name"] == perspective), None)
    if persp is None:
        available = ", ".join(p["name"] for p in perspectives) or "none"
        print_error(f"Perspective '{perspective}' not found for '{feature_part}'. Available: {available}")
        raise typer.Exit(code=1)
    when = f" ({persp['when']})" if persp["when"] else ""
    lines = [
        f"# Zero-Context Feature Review — {perspective}{when}",
        "",
        "## Feature (basic info)",
        *_feature_basic_info(meta, fm_path),
        f"- Perspective doc: {persp['path']}",
        "",
        "## Review brief",
        persp["content"],
        "",
        "Read the FEATURE.md and its code yourself (you have the tools).",
        "Report back when done — report what you found, not a verdict.",
    ]
    if broken:
        lines += ["", "Note: broken review doc(s) skipped (unparseable or missing when/description):", *(f"  - {p}" for p in broken)]
    echo("\n".join(lines))


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

@features_app.command("stages", short_help="List stages (L2 development periods).")
def stages_cmd(
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to the features/ directory. Defaults to $CWD/features (or $CWD/.ai_partners/features).",
    ),
):
    """List stages from stages/*/STAGE.md with status, period, and delivery."""
    fd = resolve_features_dir(features_dir)
    stages = list_stages(str(fd))
    if not stages:
        print_info("No stages found under stages/.")
        return
    table_data = [
        [st.get("_stage_dir", "?"), st.get("status", "?"), st.get("delivery", "—"), st.get("period", "—")]
        for st in stages
    ]
    print_simple_table(
        data=table_data,
        headers=["Stage", "Status", "Delivery", "Period"],
        title="Stages",
        column_ratios=[1.6, 0.6, 0.8, 1],
    )
    console.print("\n[dim]See stages/README.md for the convention; ROADMAP.md is the cross-stage index.[/dim]")


# ---------------------------------------------------------------------------
# regressions
# ---------------------------------------------------------------------------

@features_app.command("regressions", short_help="List regression sets.")
def regressions_cmd(
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to the features/ directory. Defaults to $CWD/features (or $CWD/.ai_partners/features).",
    ),
):
    """List regression sets from regressions/*/REGRESSION.md."""
    fd = resolve_features_dir(features_dir)
    regs = list_regressions(str(fd))
    if not regs:
        print_info("No regression sets found under regressions/.")
        return
    table_data = [
        [r.get("_regression_dir", "?"), r.get("title", r.get("_regression_dir", "?")),
         r.get("version", "?"), r.get("status", "?")]
        for r in regs
    ]
    print_simple_table(
        data=table_data,
        headers=["Set", "Title", "Ver", "Status"],
        title="Regressions",
        column_ratios=[1, 1.5, 0.3, 0.6],
    )
    console.print("\n[dim]See regressions/README.md for the convention.[/dim]")


# ---------------------------------------------------------------------------
# surface
# ---------------------------------------------------------------------------

@features_app.command("surface", short_help="Render the project surface (L3).")
def surface_cmd(
    features_dir: Optional[Path] = typer.Option(
        None, "--dir", "-d",
        help="Path to the features/ directory. Defaults to $CWD/features (or $CWD/.ai_partners/features).",
    ),
):
    """Render SURFACE.md — the project's L3 deliverables checklist."""
    fd = resolve_features_dir(features_dir)
    surface = fd / SURFACE_FILENAME
    if not surface.is_file():
        print_error(f"SURFACE.md not found at {surface}")
        print_info("Run 'moss features init' to scaffold it, or create it manually.")
        raise typer.Exit(code=1)
    echo(surface.read_text(encoding="utf-8"))
    echo(f"\nSurface path: {surface.resolve()}")
