
"""Memento command group — the trajectory, read and written from a shell.

Semantics are git's, deliberately: someone who knows ``git log``/``git show``/
``git commit`` should not have to learn a second vocabulary to see their own memory.

    status                 where I am, what is mounted, how the last hook run went
    log [--from --until]   the anchors in a range, one line each
    show <coord>           one anchor in detail — note, directory, commit_view output
    commit -m "..."        freeze the span since the last anchor
    branch list|show|create
    hook list|run|log

Two things are deliberately absent. ``read`` — the raw span lives in the agent session
store, not in this index, so reading it belongs to the ghost channel that holds the
session, not to a shell that can only see the trajectory files. ``chat`` — a real round of
inference is a capability of the model side, never a command a person runs.

Every command resolves its root the same way: ``--root`` if given — before or after the
subcommand, both work — else ``.memento/ego`` in the working directory. Nothing is
searched for: a trajectory is not something to guess at, so a directory without one is an
error that names what it looked at, not a reason to go hunting for someone else's memory.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import typer

from ghoshell_moss.cli.utils import echo, print_error, print_info, print_success
from ghoshell_moss.memento import new_local_memento
from ghoshell_moss.memento.hooks import (
    HOOK_DIR_NAME,
    HOOK_SLOTS,
    LOG_FILE_NAME,
    hook_env,
    hook_paths,
    log_results,
    render_view_text,
    run_slot,
)

__all__ = ["memento_app"]

memento_app = typer.Typer(
    short_help="The cognitive trajectory — log, show, commit, branch, hook.",
    help=(
        "memento is the trajectory index: branches, anchors (commits) and the notes that "
        "say what a span was. The commands follow git's semantics on purpose."
    ),
    no_args_is_help=True,
)

branch_app = typer.Typer(short_help="The lines (branches) of the trajectory.", no_args_is_help=True)
hook_app = typer.Typer(short_help="The lifecycle slots — what is mounted, and what ran.", no_args_is_help=True)
memento_app.add_typer(branch_app, name="branch")
memento_app.add_typer(hook_app, name="hook")

DEFAULT_EGO_DIR = ".memento/ego"

# The ``--root`` given before the subcommand. A callback option is the only place typer lets
# it sit there; the commands keep their own ``--root`` so both orders read naturally. It is
# reset on every invocation, so one process running several commands never inherits a root.
_group_root: Optional[Path] = None


@memento_app.callback()
def _main(root: Optional[Path] = typer.Option(None, "--root", help="memento root directory.")):
    """Memento: the trajectory index behind your own memory."""
    global _group_root
    _group_root = root


# ── root resolution ─────────────────────────────────────────────────────────


def _resolve_root(root: Optional[Path]) -> Path:
    """One root, or a refusal that names what was looked at.

    ``--root`` wins wherever it was written. Without it the answer is ``.memento/ego`` under
    the working directory — the one place a person standing here means. There is no search:
    picking a ghost home for you would mean silently reading a trajectory nobody asked for.
    """
    given = root if root is not None else _group_root
    if given is not None:
        resolved = given.expanduser().resolve()
        if not resolved.is_dir():
            print_error(f"not a directory: {resolved}")
            raise typer.Exit(code=2)
        return resolved
    cwd_root = (Path.cwd() / DEFAULT_EGO_DIR).resolve()
    if not cwd_root.is_dir():
        print_error(
            f"no trajectory at {cwd_root}; pass --root <dir> "
            "(e.g. --root .moss/ghosts/<name>/.memento/ego)"
        )
        raise typer.Exit(code=2)
    return cwd_root


def _branch(m, name: Optional[str]):
    """The named branch, or the only one there is; failure is a message, not a traceback."""
    refs = m.list_branches()
    if not refs:
        print_error("no branch in this trajectory yet — `memento branch create <name>`")
        raise typer.Exit(code=1)
    if name is None:
        if len(refs) > 1:
            names = ", ".join(r.name for r in refs)
            print_error(f"several branches ({names}); say which with --branch")
            raise typer.Exit(code=1)
        name = refs[0].name
    branch = m.get_branch(name)
    if branch is None:
        print_error(f"no branch `{name}`")
        raise typer.Exit(code=1)
    return branch


def _hook_roots(branch, root: Path) -> list[Path]:
    """Branch first, then the ego root — both run: a branch adds to the root, never replaces it."""
    return [branch.path, root]


def _run(coro):
    try:
        return asyncio.run(coro)
    except Exception as error:  # a CLI must not dump a traceback at a person
        print_error(f"{type(error).__name__}: {error}")
        raise typer.Exit(code=2) from error


# ── status ──────────────────────────────────────────────────────────────────


@memento_app.command("status")
def status(root: Optional[Path] = typer.Option(None, "--root", help="memento root directory.")):
    """Where I am: the branch, the latest anchor, the slots, the last hook run."""
    resolved = _resolve_root(root)
    m = new_local_memento(resolved)
    refs = m.list_branches()
    echo(f"root      {resolved}")
    if not refs:
        echo("branches  (none)")
        return
    for ref in refs:
        branch = m.get_branch(ref.name)
        if branch is None:
            continue
        commits = branch.commits()
        latest = ""
        if commits:
            view = branch.get_commit(commits[-1].seq)
            if view is not None:
                latest = view.coord
        echo(f"branch    {ref.name}  commits={len(commits)}  latest={latest or '(empty)'}")
        for slot in HOOK_SLOTS:
            mounted = hook_paths(_hook_roots(branch, resolved), slot)
            if mounted:
                for path in mounted:
                    echo(f"  slot    {slot:<12} {path}")
            else:
                echo(f"  slot    {slot:<12} (not mounted)")


# ── log / show ──────────────────────────────────────────────────────────────


@memento_app.command("log")
def log_cmd(
    branch: Optional[str] = typer.Option(None, "--branch", "-b"),
    from_date: Optional[str] = typer.Option(None, "--from", help="ISO timestamp, inclusive."),
    until_date: Optional[str] = typer.Option(None, "--until", help="ISO timestamp, inclusive."),
    limit: int = typer.Option(20, "--limit", "-n"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """The anchors, one line each — summaries only, no hook is ever run here."""
    resolved = _resolve_root(root)
    m = new_local_memento(resolved)
    target = _branch(m, branch)
    rows = target.commits()
    if not rows:
        print_info("no anchor yet")
        return
    for ref in rows[-limit:]:
        view = target.get_commit(ref.seq)
        if view is None:
            continue
        mark = " (broken)" if view.is_broken else ""
        echo(f"{view.coord:<12} {view.created.astimezone().strftime('%Y-%m-%d %H:%M')}  {view.title or '(no note)'}{mark}")


@memento_app.command("show")
def show_cmd(
    coord: str = typer.Argument(..., help="coordinate, e.g. 1-404"),
    hook: bool = typer.Option(True, "--hook/--no-hook", help="run the commit_view slot."),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """One anchor in detail: the note, its directory, and what commit_view adds."""
    resolved = _resolve_root(root)
    m = new_local_memento(resolved)
    view = m.resolve_commit(coord)
    if view is None:
        print_error(f"no commit at `{coord}`")
        raise typer.Exit(code=1)
    branch = m.get_branch_by_index(view.branch_index)
    echo(f"coord     {view.coord}")
    echo(f"created   {view.created.astimezone().isoformat()}")
    echo(f"message   {view.message or '(no note — the summarizer has not run)'}")
    commit_dir = view.memento_path(resolved).parent
    echo(f"dir       {commit_dir}{'' if commit_dir.is_dir() else '  (not created)'}")
    if not hook:
        return
    results = _run(
        run_slot(
            "commit_view",
            _hook_roots(branch, resolved) if branch is not None else [resolved],
            env_base=hook_env(
                "commit_view",
                root=resolved,
                branch=branch.ref.name if branch is not None else "",
                branch_dir=branch.path if branch is not None else None,
                coord=view.coord,
                commit_dir=commit_dir,
            ),
            log_root=resolved,
        )
    )
    text = render_view_text(results)
    if text:
        echo("")
        echo(text)


# ── commit ──────────────────────────────────────────────────────────────────


@memento_app.command("commit")
def commit_cmd(
    message: str = typer.Option("", "--message", "-m", help="the note; empty = no note."),
    branch: Optional[str] = typer.Option(None, "--branch", "-b"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """Freeze a new anchor on the branch, and run the on_commit slot beside it."""
    resolved = _resolve_root(root)
    m = new_local_memento(resolved)
    target = _branch(m, branch)
    ref = target.commit(message=message)
    view = target.get_commit(ref.seq)
    coord = view.coord if view is not None else f"{target.index}-{ref.seq}"
    print_success(f"committed {coord}")
    commit_dir = view.memento_path(resolved).parent if view is not None else None
    if commit_dir is not None:
        # 协议承诺: hook 拿到的目标目录已经存在, 它只管往里搬东西.
        commit_dir.mkdir(parents=True, exist_ok=True)
    results = _run(
        run_slot(
            "on_commit",
            _hook_roots(target, resolved),
            env_base=hook_env(
                "on_commit",
                root=resolved,
                branch=target.ref.name,
                branch_dir=target.path,
                coord=coord,
                commit_dir=commit_dir,
            ),
            log_root=resolved,
        )
    )
    for result in results:
        echo(f"  {result.summary}")


# ── branch ──────────────────────────────────────────────────────────────────


@branch_app.command("list")
def branch_list(root: Optional[Path] = typer.Option(None, "--root")):
    """Every line in this trajectory."""
    resolved = _resolve_root(root)
    m = new_local_memento(resolved)
    refs = m.list_branches()
    if not refs:
        print_info("no branch yet")
        return
    for ref in refs:
        branch = m.get_branch(ref.name)
        commits = branch.commits() if branch is not None else []
        latest = ""
        if branch is not None and commits:
            view = branch.get_commit(commits[-1].seq)
            latest = view.coord if view is not None else ""
        echo(f"{ref.name}  commits={len(commits)}  latest={latest or '(empty)'}")


@branch_app.command("show")
def branch_show(
    name: Optional[str] = typer.Argument(None),
    hook: bool = typer.Option(True, "--hook/--no-hook", help="run the branch_view slot."),
    limit: int = typer.Option(10, "--limit", "-n"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """A line as it is now: its description, its recent anchors, and what branch_view adds."""
    resolved = _resolve_root(root)
    m = new_local_memento(resolved)
    target = _branch(m, name)
    echo(f"branch    {target.ref.name}  index={target.index}")
    if target.ref.description:
        echo(f"about     {target.ref.description}")
    commits = target.commits()
    echo(f"anchors   {len(commits)}")
    for ref in commits[-limit:]:
        view = target.get_commit(ref.seq)
        if view is not None:
            echo(f"  {view.coord:<12} {view.title or '(no note)'}")
    if not hook:
        return
    results = _run(
        run_slot(
            "branch_view",
            _hook_roots(target, resolved),
            env_base=hook_env(
                "branch_view",
                root=resolved,
                branch=target.ref.name,
                branch_dir=target.path,
            ),
            log_root=resolved,
        )
    )
    text = render_view_text(results)
    if text:
        echo("")
        echo(text)


@branch_app.command("create")
def branch_create(
    name: str = typer.Argument(...),
    description: str = typer.Option("", "--description", "-d"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """Open a new line."""
    resolved = _resolve_root(root)
    m = new_local_memento(resolved)
    branch = m.create_branch(name, description)
    print_success(f"created branch `{branch.ref.name}` at {branch.path}")


# ── hook ────────────────────────────────────────────────────────────────────


@hook_app.command("list")
def hook_list(
    branch: Optional[str] = typer.Option(None, "--branch", "-b"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """What is mounted on each slot — the branch's own file first, then the ego default."""
    resolved = _resolve_root(root)
    m = new_local_memento(resolved)
    target = _branch(m, branch)
    for slot in HOOK_SLOTS:
        mounted = hook_paths(_hook_roots(target, resolved), slot)
        if not mounted:
            echo(f"{slot:<12} (not mounted)")
            continue
        for path in mounted:
            echo(f"{slot:<12} {path}")


@hook_app.command("run")
def hook_run(
    slot: str = typer.Argument(..., help="on_commit / commit_view / branch_view"),
    coord: str = typer.Option("", "--coord", help="the anchor to hand the hook."),
    timeout: float = typer.Option(20.0, "--timeout"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """Run one slot by hand — the way to see what a slot prints without committing."""
    if slot not in HOOK_SLOTS:
        print_error(f"unknown slot `{slot}`; known: {', '.join(HOOK_SLOTS)}")
        raise typer.Exit(code=2)
    resolved = _resolve_root(root)
    m = new_local_memento(resolved)
    target = _branch(m, branch)
    commit_dir = None
    if coord:
        view = m.resolve_commit(coord)
        if view is None:
            print_error(f"no commit at `{coord}`")
            raise typer.Exit(code=1)
        commit_dir = view.memento_path(resolved).parent
        commit_dir.mkdir(parents=True, exist_ok=True)
    results = _run(
        run_slot(
            slot,
            _hook_roots(target, resolved),
            env_base=hook_env(
                slot,
                root=resolved,
                branch=target.ref.name,
                branch_dir=target.path,
                coord=coord,
                commit_dir=commit_dir,
            ),
            timeout=timeout,
            log_root=resolved,
        )
    )
    if not results:
        print_info(f"nothing mounted on `{slot}`")
        return
    for result in results:
        echo(f"{result.summary}  {result.path}")
        if result.stdout.strip():
            echo(result.stdout.rstrip())
        if result.stderr.strip():
            print_error(result.stderr.rstrip())


@hook_app.command("log")
def hook_log(
    limit: int = typer.Option(20, "--limit", "-n"),
    root: Optional[Path] = typer.Option(None, "--root"),
):
    """The verdicts of past runs — no retry, so this is the only record there is."""
    resolved = _resolve_root(root)
    path = resolved / HOOK_DIR_NAME / LOG_FILE_NAME
    if not path.is_file():
        print_info("no hook has run yet")
        return
    rows = path.read_text(encoding="utf-8").splitlines()[-limit:]
    for row in rows:
        try:
            data = json.loads(row)
        except json.JSONDecodeError:
            continue
        when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(data.get("at", 0)))
        verdict = "ok" if data.get("ok") else (data.get("error") or f"exit {data.get('exit_code')}")
        echo(f"{when}  {data.get('slot', ''):<12} {verdict}")
