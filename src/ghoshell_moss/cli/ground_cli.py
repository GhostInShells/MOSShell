"""Ground command group — SPEC §9 四命令: spec / init / frame / observe.

Every invocation is stateless: open → act → sediment (via __aexit__) → exit.
GROUND.md is the single source of truth across invocations.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from ghoshell_moss.cli.utils import echo, print_error, print_info, print_success
from ghoshell_moss.ground import DEFAULT_L0_FILENAME, DefaultGroundSet
from ghoshell_moss.ground._hash import PinShadow, observe_sync
from ghoshell_moss.ground._l0 import dump_l0_pins, load_l0
from ghoshell_moss.ground.contract import Ground, GroundSet

__all__ = ["ground_app"]

ground_app = typer.Typer(
    short_help="Cognitive ground — pin addresses to a directory.",
    help=(
        "Cognitive ground: pin addresses (file/glob/frontmatter/ls) to a "
        "directory, get a per-frame view of pinned content with change tracking. "
        "State persists in GROUND.md per directory."
    ),
    no_args_is_help=True,
)


# -- helpers --------------------------------------------------------------


def _resolve_root(path: Path | None) -> Path:
    root = (path or Path.cwd()).resolve()
    if not root.is_dir():
        print_error(f"not a directory: {root}")
        raise typer.Exit(code=2)
    return root


def _probe_workspace(start: Path) -> Path | None:
    try:
        from ghoshell_moss.core.blueprint.project import Project

        return Project.discover().root
    except Exception:
        pass
    return _find_repo_root(start)


def _find_repo_root(start: Path) -> Path | None:
    current = start.resolve()
    while True:
        if (current / ".git").exists() or (current / ".moss").exists():
            return current
        if current == current.parent:
            return None
        current = current.parent


async def _run_one(root: Path, coro_fn):
    """open GroundSet + one Ground → act → sediment → exit."""
    workspace = _probe_workspace(root)
    async with DefaultGroundSet(workspace_root=workspace) as gs:
        ground = await gs.open(root)
        await coro_fn(gs, ground)


def _run(coro_fn) -> None:
    root = _resolve_root(_current_root())

    async def _driver():
        await _run_one(root, coro_fn)

    asyncio.run(_driver())


_current_root_path: Path | None = None


def _current_root() -> Path | None:
    return _current_root_path


# -- spec -----------------------------------------------------------------


@ground_app.command("spec", short_help="Print the GROUND.md format specification.")
def cmd_spec() -> None:
    """Print SPECIFICATION.md — the authoritative format contract."""
    import ghoshell_moss.ground

    spec_path = (
        Path(ghoshell_moss.ground.__file__).parent / "SPECIFICATION.md"
    )
    if spec_path.is_file():
        echo(spec_path.read_text(encoding="utf-8"))
    else:
        print_error("SPECIFICATION.md not found")


# -- init -----------------------------------------------------------------


@ground_app.command("init", short_help="Create GROUND.md with defaults.")
def cmd_init(
    path: Path | None = typer.Argument(
        None, help="Directory to init (defaults to cwd)."
    ),
) -> None:
    root = _resolve_root(path)
    target = root / DEFAULT_L0_FILENAME
    if target.exists():
        print_error(f"already exists: {target}")
        raise typer.Exit(code=1)

    dump_l0_pins(root, [])
    print_success(f"initialized {target}")


# -- frame ----------------------------------------------------------------


@ground_app.command("frame", short_help="Render the current frame.")
def cmd_frame(
    path: Path | None = typer.Option(
        None, "--in", "-C", help="Ground root (defaults to cwd)."
    ),
) -> None:
    root = _resolve_root(path)

    async def _op(gs: GroundSet, ground: Ground) -> None:
        text = await ground.context()
        echo(text)

    asyncio.run(_run_one(root, _op))


# -- observe --------------------------------------------------------------


@ground_app.command(
    "observe",
    short_help="Run pin observations only; emit per-pin diagnostics.",
)
def cmd_observe(
    path: Path | None = typer.Option(
        None, "--in", "-C", help="Ground root (defaults to cwd)."
    ),
) -> None:
    root = _resolve_root(path)

    async def _op(gs: GroundSet, ground: Ground) -> None:
        pins = ground.pins()
        if not pins:
            print_info("no pins")
            return

        from ghoshell_moss.ground._addr import Anchor

        anchor = Anchor(
            ground=ground.doc_path.parent.resolve(),
            cwd=ground.root,
        )

        for p in pins:
            obs = observe_sync(p, anchor)
            status = "exists" if obs.exists else "MISSING"
            mtime_str = f"{obs.mtime:.0f}" if obs.mtime else "-"
            hash_short = obs.hash[:12] if obs.hash else "-"
            echo(f"{p.label}: {status}  mtime={mtime_str}  hash={hash_short}")

    asyncio.run(_run_one(root, _op))
