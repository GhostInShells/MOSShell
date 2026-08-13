"""Ground command group — spec / init / render / meta / observe / validate.

Every invocation is stateless: open → act → sediment (via __aexit__) → exit.
GROUND.md is the single source of truth across invocations.

Pin management is done by editing GROUND.md's frontmatter directly (the
YAML format is defined in SPECIFICATION.md).  Use `validate` to check
your edits.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer

from ghoshell_moss.cli.utils import echo, print_error, print_info, print_simple_table, print_success
from ghoshell_moss.ground import DEFAULT_L0_FILENAME, DefaultGroundSet, GroundSet
from ghoshell_moss.ground._hash import observe_sync
from ghoshell_moss.ground._l0 import dump_l0_pins, load_l0
from ghoshell_moss.ground._render import render_meta
from ghoshell_moss.ground.contract import Ground, GroundSet

__all__ = ["ground_app"]

ground_app = typer.Typer(
    short_help="Cognitive ground — pin addresses to a directory.",
    help=(
        "Cognitive ground: pin addresses (file/glob/frontmatter/ls) to a "
        "directory, get a rendered view of pinned content with change tracking. "
        "State persists in GROUND.md per directory."
    ),
    no_args_is_help=True,
)


# -- helpers --------------------------------------------------------------


def _run_async(coro):
    """asyncio.run() with pydantic error wrapping — consistent with validate."""
    try:
        return asyncio.run(coro)
    except Exception as e:
        # pydantic ValidationError → clean [ERROR] like validate
        cls = type(e).__name__
        msg = str(e)
        # strip pydantic URL noise
        if "For further information visit" in msg:
            msg = msg.split("For further information visit")[0].strip()
        print_error(f"{cls}: {msg}")
        raise typer.Exit(code=2) from e


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


def _find_ancestor_ground(start: Path) -> Path | None:
    """从 start 的父目录向上找最近的 GROUND.md, 边界 $HOME (SPEC §7.5).

    返回 GROUND.md 所在目录 (场根), 找不到返回 None.
    start 自身不查 — caller 已确认 start 无 GROUND.md.
    """
    import os

    home = Path(os.environ.get("HOME", "/")).resolve()
    current = start.resolve().parent
    while True:
        if (current / DEFAULT_L0_FILENAME).is_file():
            return current
        if current == home or current == current.parent:
            return None
        current = current.parent


async def _run_one(root: Path, coro_fn):
    """open GroundSet + one Ground → act → sediment → exit."""
    workspace = _probe_workspace(root)
    async with DefaultGroundSet(workspace_root=workspace) as gs:
        ground = await gs.open(root)
        await coro_fn(gs, ground)


async def _run_one_with_template(root: Path, coro_fn, template: str):
    """open GroundSet + one Ground with template → act → sediment → exit."""
    workspace = _probe_workspace(root)
    async with DefaultGroundSet(workspace_root=workspace) as gs:
        ground = await gs.open(root, template=template)
        await coro_fn(gs, ground)


async def _template_preview(root: Path, template: str, *, json_flag: bool = False) -> None:
    """只读模板预览 — 用模板的 body + pins 渲染, 不写 GROUND.md.

    兼容场景: 一个只有 CLAUDE.md 的 Claude 项目, 看它作为
    claude-project 场会长什么样, 而不落任何盘.

    从子目录预览时, 先向上搜寻模板的约定文件 (law pin 的 filename),
    找到祖先场根, 再 walk 进去 — 保证 law 链不会因 boundary 太窄而空.
    """
    from ghoshell_moss.ground._render import render_walk
    from ghoshell_moss.ground._l0 import load_l0

    workspace = _probe_workspace(root)
    gs = DefaultGroundSet(workspace_root=workspace)

    names = [t.name for t in gs.templates()]
    if template not in names:
        print_error(f"template '{template}' not found")
        if names:
            print_info("available templates: " + ", ".join(names))
        else:
            print_info("no templates found (.grounds/ empty or missing)")
        raise typer.Exit(code=2)

    # 找出模板的 law filename(s), 向上搜寻真实的场根边界
    tmpl_info = gs._find_template(template)
    law_filenames: set[str] = set()
    if tmpl_info is not None:
        try:
            tmpl_data = load_l0(tmpl_info.path.parent, filename=tmpl_info.path.name)
            for p in tmpl_data.pins:
                if hasattr(p, 'verb') and p.verb == 'law':
                    law_filenames.add(p.arguments.filename)
        except Exception:
            pass

    # 搜寻有约定文件的祖先目录做场根
    ground_root = root.resolve()
    if law_filenames:
        current = ground_root.parent
        while current != current.parent:
            if any((current / fname).is_file() for fname in law_filenames):
                ground_root = current
            current = current.parent

    ground = await gs.open(ground_root, template=template, override=True)

    if root.resolve() == ground_root:
        view = await ground.render()
    else:
        # 子目录预览: walk 从 root 在场根内
        view = await render_walk(
            cwd=root,
            ground_root=ground_root,
            doc_path=ground_root / DEFAULT_L0_FILENAME,
            pins=ground.pins(),
            label=ground.convention.name,
            ignore=ground.ignore_spec,
        )

    if json_flag:
        echo(view.model_dump_json(indent=2, exclude_none=True))
    else:
        echo(str(view))
    # 只读: 不 close → 不触发 sediment → GROUND.md 不落盘


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
    template: str | None = typer.Option(
        None, "--template", "-t",
        help="Template name from .grounds/ to initialize from.",
    ),
) -> None:
    root = _resolve_root(path)
    target = root / DEFAULT_L0_FILENAME
    if target.exists():
        print_error(f"already exists: {target}")
        raise typer.Exit(code=1)

    if template is not None:
        # 预检模板存在性 — 找不到时报错并列出可用模板, 不静默生成空场
        workspace = _probe_workspace(root)
        names = [t.name for t in DefaultGroundSet(workspace_root=workspace).templates()]
        if template not in names:
            print_error(f"template '{template}' not found")
            if names:
                print_info("available templates: " + ", ".join(names))
            else:
                print_info("no templates found (.grounds/ empty or missing)")
            raise typer.Exit(code=2)

        async def _op(gs: GroundSet, ground: Ground) -> None:
            await ground.sediment()

        _run_async(_run_one_with_template(root, _op, template))
        print_success(f"initialized {target} from template '{template}'")
    else:
        dir_name = root.resolve().name
        body = (
            f"# {dir_name}\n"
            "\n"
            "Ground body — free-form markdown.  Pins are declared in\n"
            "frontmatter above.  Available verbs: file, glob, frontmatter,\n"
            "ls, exec, law.  Run `moss ground verbs` for argument reference.\n"
            "Edit this file, then `moss ground validate` to check.\n"
        )
        dump_l0_pins(root, [], body=body)
        print_success(f"initialized {target}")


# -- templates ------------------------------------------------------------


@ground_app.command("templates", short_help="List available templates from .grounds/.")
def cmd_templates() -> None:
    """List all templates discovered from .grounds/ directories."""
    workspace = _probe_workspace(Path.cwd())
    gs = DefaultGroundSet(workspace_root=workspace)
    tmpls = gs.templates()
    if not tmpls:
        print_info("no templates found")
        return

    rows = [
        [t.name, t.source, t.description[:80] if t.description else "-"]
        for t in tmpls
    ]
    print_simple_table(rows, headers=["name", "source", "description"])


# -- verbs ----------------------------------------------------------------


_VERB_HELP: dict[str, dict[str, str]] = {
    "file": {
        "path": "file path (required). Anchor syntax allowed.",
        "range": "line range: N or N-M (1-indexed).",
        "budget": "content char limit, truncates with marker.",
    },
    "glob": {
        "path": "glob pattern *, **, ? (required). Anchor prefix allowed.",
        "limit": "max matched paths.",
        "max_depth": "recursion depth cap for **.",
    },
    "frontmatter": {
        "path": "file path or glob pattern (required). Anchor syntax allowed.",
        "keys": "frontmatter keys to extract; absent = full block.",
        "budget": "content char limit.",
        "limit": "max matched files in pattern mode.",
        "max_depth": "recursion depth cap in pattern mode.",
    },
    "ls": {
        "path": "directory path (required). Anchor syntax allowed.",
        "depth": "traversal depth. default 2.",
        "limit": "max directory entries.",
        "max_depth": "recursion depth cap (min of depth and max_depth wins).",
    },
    "exec": {
        "ref": "relative path to executable in ground subtree (required).",
        "timeout": "seconds. default 10, max 60.",
        "budget": "stdout char limit.",
    },
    "law": {
        "filename": "convention filename (CLAUDE.md, AGENT.md...). Collected upward from cwd to ground root (required).",
        "budget": "total char limit across collected law, truncates with marker.",
        "lines": "total line limit across collected law, truncates with marker.",
    },
}


@ground_app.command("verbs", short_help="List known pin verbs and their arguments.")
def cmd_verbs() -> None:
    """Show each verb's argument table — a quick reference for editing GROUND.md."""
    for verb, args in _VERB_HELP.items():
        echo(f"\n[{verb}]")
        rows = [[k, v] for k, v in args.items()]
        print_simple_table(rows, headers=["argument", "description"])


# -- render ---------------------------------------------------------------


@ground_app.command("render", short_help="Render the ground view.")
def cmd_render(
    path: Path | None = typer.Argument(
        None, help="Directory to view from (defaults to cwd)."
    ),
    template: str | None = typer.Option(
        None, "--template", "-t",
        help="Read-only preview using a template's body + pins (no GROUND.md written).",
    ),
    json_flag: bool = typer.Option(
        False, "-j", "--json",
        help="Output as JSON (RenderedView schema) instead of markdown.",
    ),
) -> None:
    """Render the ground view for a directory.

    - Directory has GROUND.md: ground-root mode — body + all pins expanded.
    - No GROUND.md but an ancestor has one: walk mode — $CWD-anchored pins
      (e.g. ls/file) expanded, other pins folded to a TOC. No automatic
      directory listing — that comes only from a $CWD-anchored ls pin.
    - No ground up to $HOME: hint to init.
    - --template <name>: read-only preview — open with the template's body
      and pins, render, and do NOT write GROUND.md.
    - -j / --json: output RenderedView as JSON instead of markdown.
    """
    root = _resolve_root(path)

    if template is not None:
        asyncio.run(_template_preview(root, template, json_flag=json_flag))
        return

    if (root / DEFAULT_L0_FILENAME).is_file():
        # ground-root mode
        async def _op(gs: GroundSet, ground: Ground) -> None:
            view = await ground.render()
            if json_flag:
                echo(view.model_dump_json(indent=2, exclude_none=True))
            else:
                echo(str(view))

        _run_async(_run_one(root, _op))
        return

    ground_root = _find_ancestor_ground(root)
    if ground_root is None:
        print_info(f"no ground: no GROUND.md from {root} up to $HOME")
        print_info("run 'moss ground init' to create one here")
        raise typer.Exit(code=1)

    # walk mode — render from ancestor ground, cwd = root
    async def _walk_op() -> None:
        doc_path = ground_root / DEFAULT_L0_FILENAME
        workspace = _probe_workspace(root)
        async with DefaultGroundSet(workspace_root=workspace) as gs:
            ground = await gs.open(root, doc=doc_path)
            view = await ground.render(cwd=root)
            if json_flag:
                echo(view.model_dump_json(indent=2, exclude_none=True))
            else:
                echo(str(view))

    asyncio.run(_walk_op())


# -- meta -----------------------------------------------------------------


@ground_app.command("meta", short_help="Show ground identity and pin TOC.")
def cmd_meta(
    path: Path | None = typer.Argument(
        None, help="Ground root (defaults to cwd)."
    ),
) -> None:
    """Show ground location, law chain, $id, and pin table of contents.

    Separated from ``render`` so consumers who don't need ground protocol
    get a clean content-only view.
    """
    root = _resolve_root(path)

    if (root / DEFAULT_L0_FILENAME).is_file():
        ground_root = root
    else:
        ground_root = _find_ancestor_ground(root)
        if ground_root is None:
            print_info(f"no ground: no GROUND.md from {root} up to $HOME")
            print_info("run 'moss ground init' to create one here")
            raise typer.Exit(code=1)

    doc_path = ground_root / DEFAULT_L0_FILENAME

    async def _op() -> None:
        workspace = _probe_workspace(root)
        async with DefaultGroundSet(workspace_root=workspace) as gs:
            ground = await gs.open(root, doc=doc_path)
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

    asyncio.run(_op())


# -- observe --------------------------------------------------------------


@ground_app.command(
    "observe",
    short_help="Run pin observations only; emit per-pin diagnostics.",
)
def cmd_observe(
    path: Path | None = typer.Argument(
        None, help="Directory to observe from (defaults to cwd)."
    ),
) -> None:
    """Per-pin diagnostics — one line each: label, verb, status, resolved
    target, result size.  No raw mtime or hash values (shell domain, §6.1).

    Runs in ground-root mode when GROUND.md is present, walk mode otherwise
    (same resolution as ``render``).  Missing targets show the resolved
    absolute path so the failure is self-explanatory.
    """
    from ghoshell_moss.ground._addr import Anchor

    root = _resolve_root(path)

    if (root / DEFAULT_L0_FILENAME).is_file():
        ground_root = root
    else:
        ground_root = _find_ancestor_ground(root)
        if ground_root is None:
            print_info(f"no ground: no GROUND.md from {root} up to $HOME")
            raise typer.Exit(code=1)

    doc_path = ground_root / DEFAULT_L0_FILENAME

    async def _op() -> None:
        workspace = _probe_workspace(root)
        async with DefaultGroundSet(workspace_root=workspace) as gs:
            ground = await gs.open(root, doc=doc_path)
            pins = ground.pins()
            if not pins:
                print_info("no pins")
                return

            anchor = Anchor(ground=ground_root, cwd=root)
            rows: list[list[str]] = []
            for p in pins:
                obs = observe_sync(p, anchor)
                target = _pin_target_display(p, anchor)
                status, size = _obs_status_and_size(p, obs)
                rows.append([p.label, p.verb, status, target, size])
            print_simple_table(
                rows, headers=["label", "verb", "status", "target", "size"]
            )

    asyncio.run(_op())


def _pin_target_display(pin, anchor) -> str:
    """Resolved absolute path/spec — makes MISSING self-explanatory."""
    from ghoshell_moss.ground._addr import resolve_path
    from ghoshell_moss.ground._chain import collect_law_files
    from ghoshell_moss.ground.contract import (
        ExecPin, FilePin, FrontmatterPin, GlobPin, LawPin, LsPin,
    )

    try:
        if isinstance(pin, (FilePin, FrontmatterPin, LsPin)):
            return str(resolve_path(pin.arguments.path, anchor))
        if isinstance(pin, GlobPin):
            return str(resolve_path(pin.arguments.path, anchor))
        if isinstance(pin, ExecPin):
            # 场根子树内相对路径 — 显示 resolved 绝对路径,
            # missing 时用户一眼看清是哪个文件缺
            resolved = (anchor.ground / pin.arguments.ref).resolve()
            return str(resolved)
        if isinstance(pin, LawPin):
            files = collect_law_files(anchor, pin.arguments.filename)
            if not files:
                return f"{pin.arguments.filename} (none from cwd up to ground root)"
            return f"{pin.arguments.filename} x{len(files)} (cwd → ground root)"
    except Exception as e:
        return f"[unresolved: {e}]"
    return "-"


def _obs_status_and_size(pin, obs) -> tuple[str, str]:
    """(status, size) — observation carries size/unit natively."""
    from ghoshell_moss.ground.contract import ExecPin

    if not obs.exists:
        return ("missing", "-")

    # exec: 状态从 payload 头识别
    if isinstance(pin, ExecPin) and obs.payload is not None:
        head = obs.payload.splitlines()[0] if obs.payload else ""
        if head.startswith("[timeout"):
            return ("timeout", f"{obs.size}{obs.unit}" if obs.size else "-")
        if head.startswith("error:"):
            return ("error", head[:40])
        if "[exit " in obs.payload:
            return ("nonzero", f"{obs.size}{obs.unit}")

    if obs.size == 0:
        return ("empty", f"0 {obs.unit}")
    if obs.size is not None:
        unit_sep = "" if obs.unit == "B" else f" {obs.unit}"
        return ("ok", f"{obs.size}{unit_sep}")
    return ("ok", "-")


# -- validate -------------------------------------------------------------


_REQUIRED_ARGS: dict[str, frozenset[str]] = {
    "file": frozenset({"path"}),
    "glob": frozenset({"path"}),
    "frontmatter": frozenset({"path"}),
    "ls": frozenset({"path"}),
    "exec": frozenset({"ref"}),
    "law": frozenset({"filename"}),
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
        # 文件以 --- 开头但正则没匹配 → 大概率是未闭合 frontmatter
        if text.lstrip().startswith("---"):
            warnings.append(
                "file starts with '---' but no closing '---' found — "
                "frontmatter may be unclosed; all pins will be ignored"
            )
        # no frontmatter is valid for bare-directory ground

    # --- pins ---
    pins_data = fm_data.get("pins")
    if pins_data is None:
        if warnings:
            for w in warnings:
                print_info(f"[WARN] {w}")
        else:
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
            if isinstance(rv, int):
                if rv < 1:
                    errors.append(f"{idx}: range {rv} must be >= 1 (1-indexed)")
            elif isinstance(rv, str):
                if not re.match(r"^\d+(-\d+)?$", rv):
                    errors.append(f"{idx}: 'range' '{rv}' does not match pattern N or N-M")
                else:
                    if "-" in rv:
                        r_start, r_end = (int(x) for x in rv.split("-", 1))
                    else:
                        r_start = r_end = int(rv)
                    if r_start < 1:
                        errors.append(f"{idx}: range start {r_start} must be >= 1 (1-indexed)")
                    elif r_start > r_end:
                        errors.append(f"{idx}: range start {r_start} > end {r_end}")
            else:
                errors.append(f"{idx}: 'range' must be str or int, got {type(rv).__name__}")
        if "depth" in args and verb == "ls":
            if not isinstance(args["depth"], int):
                errors.append(f"{idx}: 'depth' must be int, got {type(args['depth']).__name__}")
        if "description" in entry:
            desc = entry["description"]
            if not isinstance(desc, str):
                errors.append(f"{idx}: 'description' must be a string")
            elif len(desc) > 280:
                warnings.append(f"{idx}: description exceeds 280 chars ({len(desc)})")

        # exec ref reachability — warn if target missing or not +x
        if verb == "exec" and "ref" in args:
            ref = args["ref"]
            if not isinstance(ref, str):
                pass  # type error caught elsewhere
            elif ref.startswith("/"):
                warnings.append(f"{idx}: exec ref is absolute — exec requires relative path in ground subtree")
            elif ".." in Path(ref).parts:
                warnings.append(f"{idx}: exec ref contains '..' — must stay in ground subtree")
            else:
                resolved = (root / ref).resolve()
                if not resolved.is_file():
                    warnings.append(f"{idx}: exec ref '{ref}' not found on disk ({resolved})")
                elif not os.access(resolved, os.X_OK):
                    warnings.append(f"{idx}: exec ref '{ref}' missing +x — will render [missing]")

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
