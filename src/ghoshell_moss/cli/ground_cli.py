"""Ground command group — bash-layer landing for the cognitive ground.

Every invocation is stateless: open a Grounds owner, open the Ground rooted
at --in (or cwd), do one operation, sediment (via __aexit__), exit. L0 file
(GROUND.md) is the single source of truth across invocations.
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
from ghoshell_moss.ground.contract import PathOutsideRootError
from ghoshell_moss.ground import (
    DEFAULT_L0_FILENAME,
    DefaultGrounds,
)
from ghoshell_moss.ground._l0 import dump_l0_pins, load_l0

__all__ = ["ground_app"]


ground_app = typer.Typer(
    short_help="Cognitive ground — pin addresses to a directory.",
    help=(
        "Cognitive ground: pin addresses (path / path:80-140 / **/*.py) to a "
        "directory, get a per-frame view of pinned content with change tracking. "
        "State persists in GROUND.md per directory."
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


@ground_app.command(
    "init",
    short_help="Create GROUND.md with defaults in the target directory.",
)
def cmd_init(
    path: Optional[Path] = typer.Argument(
        None, help="Directory to init (defaults to cwd)."
    ),
) -> None:
    """Create GROUND.md with default GroundConvention + empty pin section."""
    root = _resolve_ground_root(path)
    target = root / DEFAULT_L0_FILENAME
    if target.exists():
        print_error(f"already exists: {target}")
        raise typer.Exit(code=1)

    dump_l0_pins(root, [])
    print_success(f"initialized {target}")


# ---- status ------------------------------------------------------------


@ground_app.command(
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
    echo(f"GROUND.md:        {exists}")
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


@ground_app.command(
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
    """Pin an address. Auto-creates GROUND.md on sediment if missing."""
    root = _resolve_ground_root(path)

    async def _op(grounds, ground):
        try:
            pin = ground.pin(addr, note)
        except PathOutsideRootError as e:
            print_error(str(e))
            workspace = _probe_workspace_root(root)
            if workspace is not None and workspace != root:
                echo(
                    f"  hint: to reach this addr, open a wider ground:\n"
                    f"    moss ground pin {addr} --in {workspace}"
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


@ground_app.command(
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


@ground_app.command(
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


@ground_app.command(
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


@ground_app.command(
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


@ground_app.command(
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
