"""Desktop command group — bash-layer landing for the cognitive desktop.

Every invocation is stateless: open a Grounds owner, open the Ground rooted
at --in (or cwd), do one operation, sediment (via __aexit__), exit. L0 file
(DESKTOP.md) is the single source of truth across invocations -- and across
future CTML channel landing (K16 three parallel acceptance surfaces).

Dogfood observations (2026-07-13, running against MOSShell repo itself):

- Frame at repo-root grounds is dominated by the tree section — depth=2
  produces ~200 lines just for the tree on a real repo, dwarfing pin
  content. Tree currently does no ignore-list filtering (.git/, .venv/,
  __pycache__/, node_modules/ all show up). Consider a tree_ignore
  patterns field on GroundConvention, or dropping the tree for large
  grounds and let hint_children carry the discovery.
- Budget report is accurate but only counts pin blocks, not tree or
  hints or headers. Total frame may exceed budget silently. See
  _render.py:_render_budget_warning.
- Budget warning line lands after tree/hints but before pin blocks.
  Reader misses it if scanning bottom-up. Consider hoisting to top of
  frame when triggered.
- Boundary check (K12) is strict — pins that cross-cut multiple
  workstreams need a ground rooted at their common ancestor. A
  feature-scoped ground cannot pin sibling feature docs. This is
  correct-by-design but easy to trip on; init/help could hint at it.
- Multiple DESKTOP.md scatter across a repo (one per ground root).
  Track-vs-ignore policy is a per-repo decision — users may want
  .gitignore or explicit tracking of workstream-scoped DESKTOP.md.
- Instruction chain requires a workspace boundary; passing ground root
  as workspace kills upward walk. _find_repo_root heuristic (.git or
  .moss/ upward) works for git-tracked repos; unrooted directories
  fall back to fs root walk. Environment-driven workspace discovery
  would be more principled once MOSS Environment plugs in.
- Hash-based staleness (mtime trigger + hash truth) correctly filters
  touch-only false alarms. K17 semantics verified end-to-end via
  touch + update + edit + update sequence.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer

from ghoshell_moss.cli.utils import (
    echo,
    print_error,
    print_info,
    print_success,
)
from ghoshell_moss.contracts.desktop import PathOutsideRootError
from ghoshell_moss.core.desktop import (
    DEFAULT_L0_FILENAME,
    DefaultGrounds,
)
from ghoshell_moss.core.desktop._l0 import dump_l0_pins, load_l0

__all__ = ["desktop_app"]


desktop_app = typer.Typer(
    short_help="Cognitive desktop — pin addresses to a directory.",
    help=(
        "Cognitive desktop: pin addresses (path / path:80-140 / **/*.py) to a "
        "directory, get a per-frame view of pinned content with change tracking. "
        "State persists in DESKTOP.md per directory."
    ),
    no_args_is_help=True,
)


def _resolve_ground_root(path_opt: Optional[Path]) -> Path:
    """Resolve --in / positional path to an absolute directory root.

    None → cwd. Raises typer.Exit if the path is not a directory.
    """
    root = (path_opt or Path.cwd()).resolve()
    if not root.is_dir():
        print_error(f"not a directory: {root}")
        raise typer.Exit(code=2)
    return root


async def _open_ground(root: Path):
    """Open a DefaultGrounds owner + one Ground rooted at root.

    Returns (grounds, ground) — caller must `await grounds.__aexit__(...)`
    to sediment (or use the run_op helper below).

    workspace_root probe order:
    1. Project.discover().root — MOSS workspace anchor. Preferred; treats
       desktop as a MOSS capability per K16 layering.
    2. Nearest .git / .moss upward — fallback for repos outside MOSS
       workspaces (e.g. arbitrary git clones).
    3. None — falls through; instruction chain walks to fs root.
    """
    workspace = _probe_workspace_root(root)
    grounds = DefaultGrounds(workspace_root=workspace)
    await grounds.__aenter__()
    ground = await grounds.open(root)
    return grounds, ground


def _probe_workspace_root(start: Path) -> Optional[Path]:
    """Project.discover() first, then upward .git/.moss walk."""
    try:
        from ghoshell_moss.core.blueprint.project import Project
        return Project.discover().root
    except Exception:
        pass
    return _find_repo_root(start)


def _find_repo_root(start: Path) -> Optional[Path]:
    """Walk upward from start looking for a repo marker (.git or .moss/).

    Returns None if none found — instruction walk will proceed to fs root
    (safe on typical layouts; strays are rare).
    """
    current = start.resolve()
    while True:
        if (current / ".git").exists() or (current / ".moss").exists():
            return current
        if current == current.parent:
            return None
        current = current.parent


def _run_op(coro_fn) -> None:
    """Run an async op that owns a Grounds owner: enter, act, sediment, exit.

    coro_fn: async callable taking (grounds, ground). All errors surface;
    __aexit__ runs in the finally block to guarantee sediment.
    """
    async def _driver(root: Path):
        grounds, ground = await _open_ground(root)
        try:
            await coro_fn(grounds, ground)
        finally:
            await grounds.__aexit__(None, None, None)

    return _driver


# ---- init --------------------------------------------------------------


@desktop_app.command(
    "init",
    short_help="Create DESKTOP.md with defaults in the target directory.",
)
def cmd_init(
    path: Optional[Path] = typer.Argument(
        None, help="Directory to init (defaults to cwd)."
    ),
) -> None:
    """Create DESKTOP.md with default GroundConvention + empty pin section."""
    root = _resolve_ground_root(path)
    target = root / DEFAULT_L0_FILENAME
    if target.exists():
        print_error(f"already exists: {target}")
        raise typer.Exit(code=1)

    dump_l0_pins(root, [])
    print_success(f"initialized {target}")


# ---- status ------------------------------------------------------------


@desktop_app.command(
    "status",
    short_help="Show ground info: root, convention, pin count.",
)
def cmd_status(
    path: Optional[Path] = typer.Option(
        None,
        "--in",
        "-C",
        help="Ground root (defaults to cwd).",
    ),
) -> None:
    """Show the ground's persistent state without opening it as a runtime."""
    root = _resolve_ground_root(path)
    contents = load_l0(root)
    l0_path = root / DEFAULT_L0_FILENAME
    exists = "yes" if l0_path.is_file() else "no (defaults)"
    echo(f"ground root:      {root}")
    echo(f"DESKTOP.md:       {exists}")
    echo(f"instruction_files: {list(contents.convention.instruction_files)}")
    echo(f"upward_lookup:    {contents.convention.upward_lookup}")
    echo(f"tree_depth:       {contents.convention.tree_depth}")
    echo(f"context_budget:   {contents.convention.context_budget}")
    echo(f"pin count:        {len(contents.pins)}")
    if contents.pins:
        echo("")
        echo("pins (most recent first):")
        for p in contents.pins:
            marker = f"  {p.addr}"
            if p.note:
                marker += f"  — {p.note}"
            echo(marker)


# ---- pin --------------------------------------------------------------


@desktop_app.command(
    "pin",
    short_help="Pin an address to a ground.",
)
def cmd_pin(
    addr: str = typer.Argument(
        ...,
        help="Address: path / path:80-140 / **/*.py.",
    ),
    note: str = typer.Option(
        "",
        "--note",
        "-n",
        help="Annotation preserved with the pin.",
    ),
    path: Optional[Path] = typer.Option(
        None,
        "--in",
        "-C",
        help="Ground root (defaults to cwd).",
    ),
) -> None:
    """Pin an address. Auto-creates DESKTOP.md on sediment if missing."""
    root = _resolve_ground_root(path)

    async def _op(grounds, ground):
        try:
            pin = ground.pin(addr, note)
        except PathOutsideRootError as e:
            print_error(str(e))
            # 越界: 提示更宽的 ground 该开在哪里. 优先 workspace, 兜底不给建议.
            workspace = _probe_workspace_root(root)
            if workspace is not None and workspace != root:
                echo(
                    f"  hint: to reach this addr, open a wider ground:\n"
                    f"    moss desktop pin {addr} --in {workspace}"
                )
            raise typer.Exit(code=2)
        except ValueError as e:
            print_error(f"invalid addr: {e}")
            raise typer.Exit(code=2)
        print_success(
            f"pinned: {pin.addr}"
            + (f"  note={pin.note!r}" if pin.note else "")
        )

    asyncio.run(_run_op(_op)(root))


# ---- unpin ------------------------------------------------------------


@desktop_app.command(
    "unpin",
    short_help="Remove a pin.",
)
def cmd_unpin(
    addr: str = typer.Argument(..., help="Address to unpin."),
    path: Optional[Path] = typer.Option(
        None, "--in", "-C", help="Ground root (defaults to cwd)."
    ),
) -> None:
    root = _resolve_ground_root(path)

    async def _op(grounds, ground):
        try:
            ground.unpin(addr)
        except KeyError:
            print_error(f"not pinned: {addr}")
            raise typer.Exit(code=1)
        print_success(f"unpinned: {addr}")

    asyncio.run(_run_op(_op)(root))


# ---- update -----------------------------------------------------------


@desktop_app.command(
    "update",
    short_help="Acknowledge a pin's world change: re-observe, refresh seen state.",
)
def cmd_update(
    addr: str = typer.Argument(..., help="Address to update."),
    path: Optional[Path] = typer.Option(
        None, "--in", "-C", help="Ground root (defaults to cwd)."
    ),
) -> None:
    root = _resolve_ground_root(path)

    async def _op(grounds, ground):
        try:
            result = await ground.update(addr)
        except KeyError:
            print_error(f"not pinned: {addr}")
            raise typer.Exit(code=1)
        status = "changed" if result.changed else "unchanged"
        echo(f"{addr}: {status} ({result.diff_preview})")

    asyncio.run(_run_op(_op)(root))


# ---- pins -------------------------------------------------------------


@desktop_app.command(
    "pins",
    short_help="List active pins (most recent first).",
)
def cmd_pins(
    path: Optional[Path] = typer.Option(
        None, "--in", "-C", help="Ground root (defaults to cwd)."
    ),
) -> None:
    root = _resolve_ground_root(path)

    async def _op(grounds, ground):
        pins = ground.pins()
        if not pins:
            print_info("no pins")
            return
        for p in pins:
            line = f"{p.addr}"
            if p.note:
                line += f"  — {p.note}"
            if p.seen_hash:
                line += f"   [seen]"
            echo(line)

    asyncio.run(_run_op(_op)(root))


# ---- frame ------------------------------------------------------------


@desktop_app.command(
    "frame",
    short_help="Render the current frame (tree + pin contents + budget).",
)
def cmd_frame(
    path: Optional[Path] = typer.Option(
        None, "--in", "-C", help="Ground root (defaults to cwd)."
    ),
) -> None:
    root = _resolve_ground_root(path)

    async def _op(grounds, ground):
        text = await ground.context()
        echo(text)

    asyncio.run(_run_op(_op)(root))


# ---- instruction ------------------------------------------------------


@desktop_app.command(
    "instruction",
    short_help="Print the collected instruction chain (upward CLAUDE.md).",
)
def cmd_instruction(
    path: Optional[Path] = typer.Option(
        None, "--in", "-C", help="Ground root (defaults to cwd)."
    ),
) -> None:
    root = _resolve_ground_root(path)

    async def _op(grounds, ground):
        text = ground.instruction()
        if text:
            echo(text)
        else:
            print_info("no instruction files found (upward from ground root)")

    asyncio.run(_run_op(_op)(root))
