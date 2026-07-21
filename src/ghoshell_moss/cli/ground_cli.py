"""Ground command group — spec / init / frame / meta / observe / validate.

Every invocation is stateless: open → act → sediment (via __aexit__) → exit.
GROUND.md is the single source of truth across invocations.

Pin management is done by editing GROUND.md's frontmatter directly (the
YAML format is defined in SPECIFICATION.md).  Use `validate` to check
your edits.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from ghoshell_moss.cli.utils import echo, print_error, print_info, print_success
from ghoshell_moss.ground import DEFAULT_L0_FILENAME, DefaultGroundSet
from ghoshell_moss.ground._hash import PinShadow, observe_sync
from ghoshell_moss.ground._l0 import dump_l0_pins, load_l0
from ghoshell_moss.ground._render import render_meta
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
    path: Path | None = typer.Argument(
        None, help="Ground root (defaults to cwd)."
    ),
) -> None:
    root = _resolve_root(path)

    async def _op(gs: GroundSet, ground: Ground) -> None:
        text = await ground.context()
        echo(text)

    asyncio.run(_run_one(root, _op))


# -- meta -----------------------------------------------------------------


@ground_app.command("meta", short_help="Show ground identity and pin TOC.")
def cmd_meta(
    path: Path | None = typer.Argument(
        None, help="Ground root (defaults to cwd)."
    ),
) -> None:
    """Show ground location, law chain, $id, and pin table of contents.

    Separated from ``frame`` so consumers who don't need ground protocol
    get a clean content-only frame.
    """
    root = _resolve_root(path)

    async def _op(gs: GroundSet, ground: Ground) -> None:
        chain = await ground.chain_text()
        text = render_meta(
            root=ground.root,
            doc_path=ground.doc_path,
            chain=chain,
            pins=ground.pins(),
            id_=ground.convention.id,
            label=ground.label,
        )
        echo(text)

    asyncio.run(_run_one(root, _op))


# -- observe --------------------------------------------------------------


@ground_app.command(
    "observe",
    short_help="Run pin observations only; emit per-pin diagnostics.",
)
def cmd_observe(
    path: Path | None = typer.Argument(
        None, help="Ground root (defaults to cwd)."
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


# -- validate -------------------------------------------------------------


_REQUIRED_ARGS: dict[str, frozenset[str]] = {
    "file": frozenset({"path"}),
    "glob": frozenset({"pattern"}),
    "frontmatter": frozenset({"path"}),
    "ls": frozenset({"path"}),
}
_KNOWN_VERBS = frozenset(_REQUIRED_ARGS.keys())


@ground_app.command("validate", short_help="Validate GROUND.md format and pin definitions.")
def cmd_validate(
    path: Path | None = typer.Argument(
        None, help="Ground root (defaults to cwd)."
    ),
) -> None:
    """Check GROUND.md frontmatter and pins for format errors."""
    import re

    import yaml

    root = _resolve_root(path)
    l0_path = root / DEFAULT_L0_FILENAME

    if not l0_path.is_file():
        print_error(f"no GROUND.md found at {root}")
        print_info("run 'moss ground init' to create one")
        raise typer.Exit(code=2)

    text = l0_path.read_text(encoding="utf-8")
    errors: list[str] = []
    warnings: list[str] = []

    # --- frontmatter parse ---
    fm_match = re.match(r"\A---\s*\n(.*?)\n---", text, re.DOTALL)
    if fm_match:
        try:
            fm_data = yaml.safe_load(fm_match.group(1)) or {}
        except yaml.YAMLError as e:
            errors.append(f"frontmatter YAML: {e}")
            for e2 in errors:
                print_error(e2)
            raise typer.Exit(code=2)
    else:
        fm_data = {}
        # no frontmatter is valid for bare-directory ground

    # --- pins ---
    pins_data = fm_data.get("pins")
    if pins_data is None:
        print_info("no pins in frontmatter — valid")
        return

    if not isinstance(pins_data, list):
        errors.append("frontmatter 'pins': must be a list")
        for e in errors:
            print_error(e)
        raise typer.Exit(code=2)

    if len(pins_data) == 0:
        return

    label_pattern = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_-]*$")
    seen_labels: set[str] = set()

    for i, entry in enumerate(pins_data):
        if not isinstance(entry, dict):
            errors.append(f"pin[{i}]: not a mapping")
            continue

        idx = f"pin[{i}]"

        # verb
        verb = entry.get("verb")
        if not verb:
            errors.append(f"{idx}: missing 'verb' field")
            continue
        if verb not in _KNOWN_VERBS:
            warnings.append(f"{idx}: unknown verb '{verb}' — will be skipped on load")

        # label
        label = entry.get("label")
        if not label:
            errors.append(f"{idx}: missing 'label' field")
        elif not isinstance(label, str):
            errors.append(f"{idx}: 'label' must be a string, got {type(label).__name__}")
        elif not label_pattern.match(label):
            errors.append(
                f"{idx}: label '{label}' does not match pattern "
                f"[a-zA-Z_][a-zA-Z0-9_-]* (max 63 chars)"
            )
        elif len(label) > 63:
            errors.append(f"{idx}: label '{label}' exceeds 63 char limit")
        else:
            if label in seen_labels:
                warnings.append(f"{idx}: duplicate label '{label}'")
            seen_labels.add(label)

        # verb-specific required arguments (K55: inside arguments map)
        args = entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {}
        if verb in _REQUIRED_ARGS:
            for field in _REQUIRED_ARGS[verb]:
                if field not in args:
                    errors.append(f"{idx} ({verb}): missing required argument '{field}'")

        # optional argument type checks
        if "range" in args and verb == "file":
            rv = args["range"]
            if not isinstance(rv, str) and not isinstance(rv, int):
                errors.append(f"{idx}: 'range' must be str or int, got {type(rv).__name__}")
            elif isinstance(rv, str) and not re.match(r"^\d+(-\d+)?$", rv):
                errors.append(f"{idx}: 'range' '{rv}' does not match pattern N or N-M")
        if "depth" in args and verb == "ls":
            if not isinstance(args["depth"], int):
                errors.append(f"{idx}: 'depth' must be int, got {type(args['depth']).__name__}")
        if "description" in entry:
            desc = entry["description"]
            if not isinstance(desc, str):
                errors.append(f"{idx}: 'description' must be a string")
            elif len(desc) > 280:
                warnings.append(f"{idx}: description exceeds 280 chars ({len(desc)})")

    # --- report ---
    for w in warnings:
        print_info(f"[WARN] {w}")
    for e in errors:
        print_error(e)

    if errors:
        raise typer.Exit(code=2)

    if warnings:
        print_success(f"GROUND.md is valid ({len(warnings)} warning(s))")
    else:
        print_success("GROUND.md is valid")
