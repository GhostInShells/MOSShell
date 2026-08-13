"""
Memento CLI — CLI for the memento cognitive-trajectory system (FORMAT v3).

Each command maps 1:1 to the Memento / Line interface.
Storage root defaults to ``.memento/`` under the current directory.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from ghoshell_moss.cli.utils import (
    console,
    echo,
    is_ai_mode,
    print_error,
    print_info,
    print_simple_panel,
    print_simple_table,
    print_success,
    print_warning,
)
from ghoshell_moss.contracts.workspace import FileLocker
from ghoshell_moss.memento.abc import (
    BranchRef,
    CommitDetail,
    CommitNote,
    CommitNotFoundError,
    CommitRef,
    CommitView,
    EmptyStagingError,
    LineNotFoundError,
    MementoError,
    MomentFrozenError,
    MomentRecord,
    new_commit_id,
)
from ghoshell_moss.memento.fs_memento import new_filesystem_memento

memento_app = typer.Typer(
    short_help="Memento — cognitive-trajectory system (commit anchors, fork, annotate).",
    help="Memento cognitive-trajectory CLI. Manage owners, lines (branches), and commits.",
    no_args_is_help=True,
)

_DEFAULT_ROOT = Path.cwd() / ".memento"


def _resolve_root(root: Optional[Path]) -> Path:
    return root or _DEFAULT_ROOT


def _parse_owner_name(arg: str) -> tuple[str, str]:
    if "/" not in arg:
        raise typer.BadParameter(f"expected <owner/name>, got {arg!r}")
    owner, name = arg.split("/", 1)
    return owner, name


def _parse_owner_commit(arg: str) -> tuple[str, str]:
    owner, cid = _parse_owner_name(arg)
    if not cid.startswith("cmt_"):
        raise typer.BadParameter(f"expected <owner/cmt_...>, got {arg!r}")
    return owner, cid


def _parse_ref(raw: str) -> BranchRef:
    """Parse 'owner/cmt_xxx' or 'owner/cmt_xxx/mmt_yyy' into BranchRef."""
    parts = raw.split("/")
    if len(parts) == 2:
        return BranchRef(origin=parts[0], commit_id=parts[1])
    elif len(parts) == 3:
        return BranchRef(origin=parts[0], commit_id=parts[1], moment_id=parts[2])
    raise typer.BadParameter(f"expected <owner/cmt_...> or <owner/cmt_.../moment_id>, got {raw!r}")


def _note_kind(v: CommitView) -> str:
    """Extract Kind trailer value (pydantic v2 swallows the method)."""
    from ghoshell_moss.memento.abc import TRAILER_KIND, trailer_values
    vals = trailer_values(v.note.trailers(), TRAILER_KIND)
    return vals[-1] if vals else ""


def _format_view(v: CommitView) -> str:
    kind = _note_kind(v)
    kind_part = f"[{kind}]  " if kind else ""
    return f"{v.id}  {kind_part}{v.summary()[:80]}"


def _format_ref(r: CommitRef) -> str:
    uid_short = r.branch[:16] if r.branch else ""
    return f"{r.commit_id}  [{r.kind}]  {uid_short}"


# ═══════════════════════════════════════════════════════════════════════════════
# specification
# ═══════════════════════════════════════════════════════════════════════════════

_SPEC_PATH = Path(__file__).parent / "docs" / "memento_spec.md"


@memento_app.command("specification", short_help="Read the memento concept map — start here.")
def specification():
    """Display the memento cognitive-trajectory system specification."""
    if not _SPEC_PATH.is_file():
        print_error(f"Specification not found: {_SPEC_PATH}")
        raise typer.Exit(code=1)
    echo(_SPEC_PATH.read_text(encoding="utf-8"))
    echo(f"\nSpecification path: {_SPEC_PATH.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
# init
# ═══════════════════════════════════════════════════════════════════════════════


@memento_app.command("init", short_help="Initialize the memento storage directory.")
def init_cmd(
    root: Optional[Path] = typer.Option(
        None, "--root", "-r",
        help="Memento root directory. Defaults to .memento/ in current dir.",
    ),
):
    """Create the memento storage root. Safe to run multiple times."""
    r = _resolve_root(root)
    r.mkdir(parents=True, exist_ok=True)
    print_success(f"Memento root initialized: {r.resolve()}")


# ═══════════════════════════════════════════════════════════════════════════════
# owner
# ═══════════════════════════════════════════════════════════════════════════════

owner_app = typer.Typer(no_args_is_help=True, short_help="Owner-level operations.")
memento_app.add_typer(owner_app, name="owner")


@owner_app.command("list", short_help="List all owners.")
def owner_list(
    root: Optional[Path] = typer.Option(
        None, "--root", "-r",
        help="Memento root directory.",
    ),
):
    r = _resolve_root(root)
    if not r.exists():
        print_info("No memento root found. Run 'moss memento init' first.")
        return
    owners = sorted(p.name for p in r.iterdir() if p.is_dir() and not p.name.startswith("."))
    if not owners:
        print_info("No owners found.")
        return
    for o in owners:
        echo(o)


@owner_app.command("status", short_help="Show owner status.")
def owner_status(
    owner: str = typer.Argument(..., help="Owner name."),
    root: Optional[Path] = typer.Option(None, "--root", "-r"),
):
    r = _resolve_root(root)
    m = new_filesystem_memento(r, owner)
    lines = m.list_lines()
    all_branches = m.list_all_branches()
    log_entries = len(m.log())
    print_simple_panel(
        f"owner:          {owner}\n"
        f"active lines:   {', '.join(lines) or '(none)'} ({len(lines)})\n"
        f"all branches:   {len(all_branches)}\n"
        f"commits:        {log_entries}",
        title=f"Owner: {owner}",
    )


@owner_app.command("log", short_help="Show owner-level commit log (commits.jsonl).")
def owner_log(
    owner: str = typer.Argument(..., help="Owner name."),
    limit: int = typer.Option(0, "--limit", "-n", help="Show last N entries. 0 = all."),
    root: Optional[Path] = typer.Option(None, "--root", "-r"),
):
    r = _resolve_root(root)
    m = new_filesystem_memento(r, owner)
    entries = m.log()
    if limit > 0:
        entries = entries[-limit:]
    if not entries:
        print_info("No commits.")
        return
    for e in entries:
        echo(_format_ref(e))


# ═══════════════════════════════════════════════════════════════════════════════
# branch
# ═══════════════════════════════════════════════════════════════════════════════

branch_app = typer.Typer(no_args_is_help=True, short_help="Line (branch) operations.")
memento_app.add_typer(branch_app, name="branch")


def _get_line(root: Path, owner: str, name: str):
    m = new_filesystem_memento(root, owner)
    return m, m.get_line(name)


@branch_app.command("create", short_help="Create a new line (branch).")
def branch_create(
    owner_name: str = typer.Argument(..., help="<owner/name> for the new line."),
    from_ref: Optional[str] = typer.Option(
        None, "--from-ref",
        help="Fork from <owner/cmt_...>. Cross-owner: set origin != owner.",
    ),
    overlay: Optional[str] = typer.Option(None, "--overlay", help="JSON overlay for fork."),
    root: Optional[Path] = typer.Option(None, "--root", "-r"),
):
    owner, name = _parse_owner_name(owner_name)
    r = _resolve_root(root)
    m = new_filesystem_memento(r, owner)
    fr = _parse_ref(from_ref) if from_ref else None
    ov = json.loads(overlay) if overlay else None
    line = m.create_line(name, from_ref=fr, overlay=ov)
    print_success(f"Line created: {owner}/{name} (uid={line.branch_identifier})")


@branch_app.command("list", short_help="List lines for an owner.")
def branch_list(
    owner: str = typer.Argument(..., help="Owner name."),
    root: Optional[Path] = typer.Option(None, "--root", "-r"),
):
    r = _resolve_root(root)
    m = new_filesystem_memento(r, owner)
    for name in m.list_lines():
        line = m.get_line(name)
        ref_str = ""
        if line.ref and line.ref.commit_id:
            origin = line.ref.origin or owner
            ref_str = f" -> {origin}/{line.ref.commit_id}"
        echo(f"{owner}/{name}  uid={line.branch_identifier}{ref_str}")


@branch_app.command("record", short_help="Record a moment to a line's staging.")
def branch_record(
    owner_name: str = typer.Argument(..., help="<owner/name>."),
    data: Optional[str] = typer.Argument(None, help="JSON payload. Reads stdin if omitted."),
    type_: str = typer.Option("raw/v1", "--type", "-t", help="Payload type identifier."),
    moment_id: Optional[str] = typer.Option(None, "--id", help="Moment id. Auto-generated if omitted."),
    root: Optional[Path] = typer.Option(None, "--root", "-r"),
):
    owner, name = _parse_owner_name(owner_name)
    r = _resolve_root(root)
    m, line = _get_line(r, owner, name)

    payload_str = data
    if payload_str is None:
        payload_str = sys.stdin.read().strip()
    if not payload_str:
        raise typer.BadParameter("data required (argument or stdin)")

    try:
        payload = json.loads(payload_str)
    except json.JSONDecodeError:
        payload = {"text": payload_str}

    mid = moment_id or new_commit_id().replace("cmt_", "mmt_")
    record = MomentRecord(id=mid, type=type_, payload=payload)
    line.record(record)
    echo(mid)


@branch_app.command("commit", short_help="Freeze staging into a commit.")
def branch_commit(
    owner_name: str = typer.Argument(..., help="<owner/name>."),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Commit message."),
    kind: str = typer.Option("semantic", "--kind", help="semantic | mechanical."),
    to: Optional[str] = typer.Option(None, "--to", help="Boundary moment id (inclusive prefix)."),
    root: Optional[Path] = typer.Option(None, "--root", "-r"),
):
    owner, name = _parse_owner_name(owner_name)
    r = _resolve_root(root)
    locker = _acquire_owner_lock(r, owner)
    try:
        m, line = _get_line(r, owner, name)
        view = line.commit(text=message or "", kind=kind, boundary_moment_id=to)
        echo(view.id)
    finally:
        locker.release()


@branch_app.command("staging", short_help="Show staging (unfrozen moments).")
def branch_staging(
    owner_name: str = typer.Argument(..., help="<owner/name>."),
    root: Optional[Path] = typer.Option(None, "--root", "-r"),
):
    owner, name = _parse_owner_name(owner_name)
    r = _resolve_root(root)
    m, line = _get_line(r, owner, name)
    records = line.staging()
    if not records:
        print_info("Staging is empty.")
        return
    for rec in records:
        content_preview = rec.content[:80] if rec.content else ""
        echo(f"{rec.id}  [{rec.type}]  {content_preview}")


@branch_app.command("log", short_help="Show line history (parent chain).")
def branch_log(
    owner_name: str = typer.Argument(..., help="<owner/name>."),
    limit: int = typer.Option(0, "--limit", "-n", help="Show last N commits."),
    root: Optional[Path] = typer.Option(None, "--root", "-r"),
):
    owner, name = _parse_owner_name(owner_name)
    r = _resolve_root(root)
    m, line = _get_line(r, owner, name)
    history = line.log()
    if limit > 0:
        history = history[-limit:]
    for v in history:
        echo(_format_view(v))


@branch_app.command("window", short_help="Show sliding window (summaries + details).")
def branch_window(
    owner_name: str = typer.Argument(..., help="<owner/name>."),
    detail_n: int = typer.Option(10, "--detail-n", "-d", help="Number of detail frames."),
    summary_m: int = typer.Option(-1, "--summary-m", "-s", help="Number of summaries (-1 = all)."),
    root: Optional[Path] = typer.Option(None, "--root", "-r"),
):
    owner, name = _parse_owner_name(owner_name)
    r = _resolve_root(root)
    m, line = _get_line(r, owner, name)
    win = line.window(detail_n=detail_n, summary_m=summary_m)
    if win.summaries:
        echo("── summaries ──")
        for v in win.summaries:
            echo(_format_view(v))
    if win.details:
        echo("── details ──")
        for rec in win.details:
            content_preview = f"  {rec.content[:60]}" if rec.content else ""
            echo(f"{rec.id}  [{rec.type}]{content_preview}")
    if not win.summaries and not win.details:
        print_info("Empty window.")


@branch_app.command("list-all", short_help="List all branches including abandoned (full index).")
def branch_list_all(
    owner: str = typer.Argument(..., help="Owner name."),
    root: Optional[Path] = typer.Option(None, "--root", "-r"),
):
    r = _resolve_root(root)
    m = new_filesystem_memento(r, owner)
    branches = m.list_all_branches()
    if not branches:
        print_info("No branches.")
        return
    for br in branches:
        active_mark = " " if br.status == "active" else "*"
        echo(f"{active_mark} uid={br.uid}  name={br.name}  [{br.status}]")


@branch_app.command("delete", short_help="Delete a line (head only, workspace and commits survive).")
def branch_delete(
    owner_name: str = typer.Argument(..., help="<owner/name>."),
    root: Optional[Path] = typer.Option(None, "--root", "-r"),
):
    owner, name = _parse_owner_name(owner_name)
    r = _resolve_root(root)
    m = new_filesystem_memento(r, owner)
    m.delete_line(name)
    print_success(f"Line {owner}/{name} deleted (head only — workspace and commits preserved).")


# ═══════════════════════════════════════════════════════════════════════════════
# commit
# ═══════════════════════════════════════════════════════════════════════════════

commit_app = typer.Typer(no_args_is_help=True, short_help="Commit read and annotate.")
memento_app.add_typer(commit_app, name="commit")


def _get_memento_for_owner(root: Path, owner: str):
    return new_filesystem_memento(root, owner)


@commit_app.command("show", short_help="Show a commit's full content.")
def commit_show(
    owner_commit: str = typer.Argument(..., help="<owner/cmt_...>."),
    notes: bool = typer.Option(False, "--notes", help="Show all annotation versions."),
    root: Optional[Path] = typer.Option(None, "--root", "-r"),
):
    owner, cid = _parse_owner_commit(owner_commit)
    r = _resolve_root(root)
    m = _get_memento_for_owner(r, owner)
    detail = m.show(cid)
    echo(f"commit: {detail.commit.id}")
    echo(f"created: {detail.commit.created}")
    if detail.moments:
        echo(f"moments ({len(detail.moments)}):")
        for mr in detail.moments:
            content_part = f"  content={mr.content[:60]!r}" if mr.content else ""
            echo(f"  {mr.id}  [{mr.type}]  threads={mr.threads}{content_part}")
    if notes:
        echo(f"notes ({len(detail.notes)}):")
        for n in detail.notes:
            echo(f"  [{n.ts}] {n.title}: {n.body[:120]}")


@commit_app.command("annotate", short_help="Annotate a commit (aperture 2).")
def commit_annotate(
    owner_commit: str = typer.Argument(..., help="<owner/cmt_...>."),
    message: str = typer.Option(..., "--message", "-m", help="Annotation body."),
    title: str = typer.Option("", "--title", "-t", help="One-line summary title."),
    root: Optional[Path] = typer.Option(None, "--root", "-r"),
):
    owner, cid = _parse_owner_commit(owner_commit)
    r = _resolve_root(root)
    m = _get_memento_for_owner(r, owner)
    view = m.annotate(cid, title=title, body=message)
    print_success(f"Annotated: {_format_view(view)}")


@commit_app.command("space", short_help="Show a commit's directory path.")
def commit_space(
    owner_commit: str = typer.Argument(..., help="<owner/cmt_...>."),
    root: Optional[Path] = typer.Option(None, "--root", "-r"),
):
    owner, cid = _parse_owner_commit(owner_commit)
    r = _resolve_root(root)
    m = _get_memento_for_owner(r, owner)
    echo(m.commit_space(cid))


# ═══════════════════════════════════════════════════════════════════════════════
# witness (placeholder — this iteration doesn't need git)
# ═══════════════════════════════════════════════════════════════════════════════

witness_app = typer.Typer(no_args_is_help=True, short_help="Witness layer (git sidecar).")
memento_app.add_typer(witness_app, name="witness")


@witness_app.command("status", short_help="Show witness status (not yet implemented).")
def witness_status():
    print_info("Witness layer not implemented in this iteration.")


@witness_app.command("snapshot", short_help="Trigger a witness snapshot (not yet implemented).")
def witness_snapshot():
    print_info("Witness layer not implemented in this iteration.")


# ═══════════════════════════════════════════════════════════════════════════════
# agent
# ═══════════════════════════════════════════════════════════════════════════════

agent_app = typer.Typer(no_args_is_help=True, short_help="Agent operations.")
memento_app.add_typer(agent_app, name="agent")

_AGENT_IMPORT_ERROR = (
    "agent support requires pydantic-ai + anthropic extras. "
    "Install with: pip install ghoshell-moss[ghost]"
)


def _owner_from_path(agent_path: Path) -> str:
    """Derive owner name from agent file stem. translator.agent.py → translator."""
    stem = agent_path.stem
    return stem.removesuffix(".agent") if stem.endswith(".agent") else stem


def _build_agent(agent_path: str, cwd: Path | None = None):
    """Build a MementoAgent from a .py file, with a friendly error if deps missing.

    cwd defaults to the agent .py parent directory and grounds filesystem
    capabilities (e.g. look_at). CLI --cwd overrides.
    """
    try:
        from ghoshell_moss.agents.memento_pydantic_agent import factory
    except ImportError:
        print_error(_AGENT_IMPORT_ERROR)
        raise typer.Exit(code=1)
    path = Path(agent_path).resolve()
    if not path.is_file():
        print_error(f"agent .py not found: {path}")
        raise typer.Exit(code=1)
    if not path.suffix == ".py":
        print_warning(f"expected .py file, got {path.suffix!r}; attempting anyway")
    return factory(path, cwd=cwd)


def _resolve_agent_cwd(agent_path: Path, cwd: Optional[Path]) -> Path:
    """Resolve cwd: explicit flag wins, else agent .py parent directory."""
    if cwd is not None:
        return cwd.resolve()
    return agent_path.parent.resolve()


def _resolve_memento(root: Optional[Path], owner: str):
    """Resolve memento if the root exists, else return None."""
    r = _resolve_root(root)
    if r.exists():
        return new_filesystem_memento(r, owner)
    return None


def _acquire_owner_lock(root: Path, owner: str) -> FileLocker:
    """Acquire an exclusive lock on the owner directory in the memento root.

    Uses the same fcntl.flock mechanism as cell.py run-cycle protection
    (§3 nail 7: owner-level lock, timeout=0 fast-fail).  When the lock is
    already held by another process the command exits 1 with a diagnostic
    on stderr — the caller never competes.
    """
    lock_path = root / owner / ".lock"
    locker = FileLocker(lock_path)
    if not locker.acquire(timeout=0):
        print_error(
            f"owner {owner!r} is locked — another process is using this "
            f"memento (lock file: {lock_path})."
        )
        raise typer.Exit(code=1)
    return locker


def _staging_count(memento, line_name: str) -> Optional[int]:
    """Moments currently in a line's staging, or None when unresolvable.

    Used for the invoke-before/after staging diff (§9.2 — the runner only
    observes, it never writes). None propagates as "unknown", not as zero.
    """
    if memento is None:
        return None
    try:
        return len(memento.get_line(line_name).staging())
    except Exception:
        return None


_LOG_TARGET = "moss.memento_agent"


def _configure_log_file(log_file: Optional[Path]) -> None:
    """Attach a file handler to the memento-agent logger for opt-in tracing.

    Without --log-file the logger stays inert: INFO/DEBUG are dropped by the
    default level, WARNING+ surface on stderr via root's last-resort handler
    (raw format — not user-friendly, but never silent).  With --log-file the
    file captures everything at DEBUG; the logger's propagation is cut so
    the file is the sole sink — no raw last-resort stderr mixed with the
    CLI's own rich print_error diagnostics.
    """
    if log_file is None:
        return
    target = Path(log_file).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(_LOG_TARGET)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(str(target), encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    logger.info("log file opened: %s", target)


@agent_app.command("parse", short_help="Show the composed instruction (what the model sees).")
def agent_parse(
    agent_path: str = typer.Argument(..., help="Path to *.agent.py file."),
):
    """Display the full system instruction that will be sent to the model.

    This is the parse-vs-run parity guarantee: what you see here is exactly
    what the model receives as its system text on `invoke`.
    """
    agent = _build_agent(agent_path)
    instruction = agent.compose_instruction()
    echo(instruction)
    echo(f"\nsha: {agent.instruction_sha()}")


@agent_app.command("invoke", short_help="Run the agent with a user prompt.")
def agent_invoke(
    agent_path: str = typer.Argument(..., help="Path to *.agent.py file."),
    prompt: str = typer.Argument(..., help="User prompt for the agent."),
    owner: Optional[str] = typer.Option(
        None, "--owner",
        help="Owner name. Default: derived from file stem (translator.agent.py → translator).",
    ),
    branch: str = typer.Option(
        "main", "--branch", "-b",
        help="Line (branch) name. Default: main.",
    ),
    cwd: Optional[Path] = typer.Option(
        None, "--cwd",
        help="Working directory. Defaults to agent .py parent.",
    ),
    root: Optional[Path] = typer.Option(None, "--root", "-r", help="Memento root directory."),
    as_json: bool = typer.Option(
        False, "--json",
        help="Emit structured output: {ok, output, moment_count, commit_id, branch}.",
    ),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file",
        help="Write runtime lifecycle and error detail to this file (opt-in tracing).",
    ),
):
    """Run one agent invocation. Returns the model's final answer on stdout.

    Owner defaults to the agent file stem (translator.agent.py → translator).
    Branch defaults to "main". When the memento root exists, moments are
    recorded to that line's staging.

    Output contract (loop-callable): exit 0 ⟺ a non-empty final answer was
    produced; stdout carries the answer (or JSON with --json). Any failure
    exits 1 with a diagnostic on stderr; stdout is not trusted.
    """
    _configure_log_file(log_file)
    agent_py_path = Path(agent_path).resolve()
    resolved_cwd = _resolve_agent_cwd(agent_py_path, cwd)
    agent = _build_agent(agent_path, cwd=resolved_cwd)
    resolved_owner = owner or _owner_from_path(agent_py_path)

    # Resolve memento root and lock at the owner scope (fast-fail).
    # Built separately from _resolve_memento so the resolved root is available
    # for lock construction at the owner-directory level.
    resolved_root = _resolve_root(root)
    locker: Optional[FileLocker] = None
    memento = None
    if resolved_root.exists():
        locker = _acquire_owner_lock(resolved_root, resolved_owner)
        memento = new_filesystem_memento(resolved_root, resolved_owner)

    staging_before = _staging_count(memento, branch)

    try:
        result = asyncio.run(agent.invoke(
            user_prompt=prompt,
            memento=memento,
            line_name=branch,
            cwd=resolved_cwd,
        ))
    except Exception as exc:
        logging.getLogger(_LOG_TARGET).error(
            "invoke failed: agent=%s owner=%s branch=%s: %s",
            agent_path, resolved_owner, branch, exc, exc_info=True,
        )
        print_error(f"invoke failed: {exc}")
        raise typer.Exit(code=1)
    finally:
        if locker is not None:
            locker.release()

    if not result.strip():
        logging.getLogger(_LOG_TARGET).error(
            "invoke produced empty output: agent=%s owner=%s branch=%s",
            agent_path, resolved_owner, branch,
        )
        print_error(
            f"invoke produced an empty final answer on {resolved_owner}/{branch} — "
            f"treating as failure (a clean completion always ends in plain text)."
        )
        raise typer.Exit(code=1)

    if as_json:
        staging_after = _staging_count(memento, branch)
        moment_count = None
        if staging_before is not None and staging_after is not None:
            moment_count = staging_after - staging_before
        # v1 has no auto-commit (staging accumulates; humans commit explicitly),
        # so commit_id is always null here — the field exists for contract shape.
        echo(json.dumps({
            "ok": True,
            "output": result,
            "moment_count": moment_count,
            "commit_id": None,
            "branch": branch,
        }, ensure_ascii=False))
        return

    echo(result)


@agent_app.command("dry-run", short_help="Probe the agent: real generation, tools not executed.")
def agent_dry_run(
    agent_path: str = typer.Argument(..., help="Path to *.agent.py file."),
    prompt: str = typer.Argument(..., help="User prompt for the agent."),
    owner: Optional[str] = typer.Option(
        None, "--owner",
        help="Owner name. Default: derived from file stem.",
    ),
    branch: str = typer.Option(
        "main", "--branch", "-b",
        help="Line (branch) name. Default: main.",
    ),
    cwd: Optional[Path] = typer.Option(
        None, "--cwd",
        help="Working directory. Defaults to agent .py parent.",
    ),
    root: Optional[Path] = typer.Option(None, "--root", "-r", help="Memento root directory."),
    as_json: bool = typer.Option(
        False, "--json", "-j",
        help="Emit the full InvocationRecord (tool_calls + messages + usage) as JSON.",
    ),
    log_file: Optional[Path] = typer.Option(
        None, "--log-file",
        help="Write runtime lifecycle and error detail to this file (opt-in tracing).",
    ),
):
    """Pure probe — one real generation, paused at the tool-call position.

    The model's tool calls are NOT executed (zero side effects) and nothing is
    written to memento. Prints the unexecuted tool calls; with --json, emits
    the full record (tool_calls + messages + usage).
    """
    _configure_log_file(log_file)
    agent_py_path = Path(agent_path).resolve()
    resolved_cwd = _resolve_agent_cwd(agent_py_path, cwd)
    agent = _build_agent(agent_path, cwd=resolved_cwd)
    resolved_owner = owner or _owner_from_path(agent_py_path)

    resolved_root = _resolve_root(root)
    memento = None
    if resolved_root.exists():
        memento = new_filesystem_memento(resolved_root, resolved_owner)

    try:
        record = asyncio.run(agent.dry_run(
            user_prompt=prompt,
            memento=memento,
            line_name=branch,
            cwd=resolved_cwd,
        ))
    except Exception as exc:
        logging.getLogger(_LOG_TARGET).error(
            "dry-run failed: agent=%s owner=%s branch=%s: %s",
            agent_path, resolved_owner, branch, exc, exc_info=True,
        )
        print_error(f"dry-run failed: {exc}")
        raise typer.Exit(code=1)

    if as_json:
        echo(json.dumps(record.model_dump(), ensure_ascii=False, indent=2))
        return

    parts: list[str] = []
    if record.content:
        parts.append(record.content)
    for call in record.tool_calls:
        name = call.get("tool_name", "")
        code = call.get("args", {}).get("code", "")
        parts.append(f"{name}:\n{code}" if code else name)
    echo("\n\n".join(parts) if parts else "(no content, no tool use)")


@agent_app.command("dump-anchor", short_help="Dump the agent's cognitive conditions as an anchor file.")
def agent_dump_anchor(
    agent_path: str = typer.Argument(..., help="Path to *.agent.py file."),
    owner: Optional[str] = typer.Option(
        None, "--owner",
        help="Owner name. Default: derived from file stem.",
    ),
    branch: str = typer.Option(
        "main", "--branch", "-b",
        help="Line (branch) name. Default: main.",
    ),
    root: Optional[Path] = typer.Option(None, "--root", "-r", help="Memento root directory."),
    name: str = typer.Option(
        "", "--name",
        help="Anchor filename stem. Empty = auto uid-based name.",
    ),
    description: str = typer.Option(
        "", "--description",
        help="Anchor one-line note.",
    ),
    out_dir: Optional[Path] = typer.Option(
        None, "--out-dir", "-o",
        help="Output directory. Default: agent .py parent.",
    ),
):
    """Dump the agent's current cognitive conditions to a self-explaining anchor.

    Freezes instruction (composed, window folded in) + tool protocol + model
    config into an .anchor.yml. No model call, no side effects. A consumer
    curls the anchor's `ref` to reconstruct the payload shape.
    """
    agent_py_path = Path(agent_path).resolve()
    agent = _build_agent(agent_path)
    resolved_owner = owner or _owner_from_path(agent_py_path)
    memento = _resolve_memento(root, resolved_owner)

    try:
        anchor = agent.dump_anchor(
            memento=memento,
            line_name=branch,
            name=name,
            description=description,
        )
    except NotImplementedError:
        print_info("dump-anchor not yet implemented in this agent version.")
        raise typer.Exit(code=0)

    target_dir = (out_dir or agent_py_path.parent).resolve()
    anchor_name = name or f"agent-{anchor.meta.uid[:8]}"
    path = anchor.dump_to_dir(target_dir, anchor_name)
    print_success(f"anchor: {path}")


@agent_app.command("export-context", short_help="Export current context as markdown.")
def agent_export_context(
    agent_path: str = typer.Argument(..., help="Path to *.agent.py file."),
    owner: Optional[str] = typer.Option(
        None, "--owner",
        help="Owner name. Default: derived from file stem.",
    ),
    branch: str = typer.Option(
        "main", "--branch", "-b",
        help="Line (branch) name. Default: main.",
    ),
    root: Optional[Path] = typer.Option(None, "--root", "-r", help="Memento root directory."),
):
    """Export the agent-perspective current context as markdown.

    Includes system prompt, folded window, and recent staging moments.
    """
    agent = _build_agent(agent_path)
    resolved_owner = owner or _owner_from_path(Path(agent_path).resolve())
    memento = _resolve_memento(root, resolved_owner)
    if memento is None:
        print_info("no memento root found; export may be incomplete.")
        raise typer.Exit(code=0)
    try:
        md = agent.export_context_md(memento, branch)
    except NotImplementedError:
        print_info("export-context not yet implemented in this agent version.")
        raise typer.Exit(code=0)
    echo(md)


@agent_app.command("describe", short_help="Agent-perspective line summary.")
def agent_describe(
    agent_path: str = typer.Argument(..., help="Path to *.agent.py file."),
    owner: Optional[str] = typer.Option(
        None, "--owner",
        help="Owner name. Default: derived from file stem.",
    ),
    branch: str = typer.Option(
        "main", "--branch", "-b",
        help="Line (branch) name. Default: main.",
    ),
    root: Optional[Path] = typer.Option(None, "--root", "-r", help="Memento root directory."),
):
    """Show an agent's semantic summary of a memento line.

    Contrast with 'moss memento branch log/window' which gives the structural
    view (commit / moment / trailer). This gives the agent's own interpretation.
    """
    agent = _build_agent(agent_path)
    resolved_owner = owner or _owner_from_path(Path(agent_path).resolve())
    memento = _resolve_memento(root, resolved_owner)
    if memento is None:
        print_info("no memento root found; describe requires a memento line.")
        raise typer.Exit(code=0)
    try:
        summary = agent.describe_line(memento, branch)
    except NotImplementedError:
        print_info("describe not yet implemented in this agent version.")
        raise typer.Exit(code=0)
    echo(summary)
