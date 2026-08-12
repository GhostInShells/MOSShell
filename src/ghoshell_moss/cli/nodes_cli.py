"""
moss nodes — discover, create, launch, and maintain node cells.

CLI is the files-layer operator for node cells. Five things:
  discovery:  list / show
  creation:   create / link / install
  startup:    run     (foreground blocking, CLI is owner)
  debug:      status
  cleanup:    kill / prune

CLI stays 100% in the Project layer (0% Matrix except subprocess.Popen itself).
Runtime governance (accept/deny/mesh view/attach) is agent-in-channel — not CLI.
Deep debug = write a debug cell and Matrix.discover() inside it.

Target for run/show/install is path only. Three-in-one resolve:
  - directory        → NodeManifest.read_from_directory
  - NODE.md file     → NodeManifest.read_from_file
  - .py script       → NodeManifest.from_script (upward NODE.md, ad-hoc fallback)
  - no arg           → NodeManifest.find_upward(cwd)

Future direction: the current implementation still carries CLI-specific glue
(singleton lock, signal forwarding, crash-fast messaging, address matching).
Ideally the CLI becomes a thin shell over blueprint abstractions —
each command a one-liner delegating to Project/NodeManager/NodeLauncher.
When blueprint grows enough surface (e.g. a Runner contract that owns
spawn+lock+wait+cleanup, an address-query helper), collapse the glue here.

Design record: .ai_partners/features/workstreams/2026/06/cells-cli/plan.md
"""

import contextlib
import os
import signal
import subprocess
import sys
import time
from importlib import resources
from pathlib import Path

import typer

from ghoshell_moss.core.blueprint.project import Project
from ghoshell_moss.core.blueprint.cell import (
    NodeManifest, NodeLauncher, CellAddressCodec, CellRuntimeInfo, ExecSpec,
)

from .utils import (
    print_simple_table, print_simple_panel,
    print_error, print_warning, print_info, print_success, echo,
)

nodes_app = typer.Typer(
    help="Discover, create, launch, and maintain MOSS node cells.",
    no_args_is_help=True,
)

_NODE_STUB_PACKAGE = 'ghoshell_moss.stubs.node'
_KILL_GRACE_SECONDS = 3.0     # SIGTERM → wait → SIGKILL for kill/prune
_RUN_GRACE_SECONDS = 5.0      # SIGTERM → wait → SIGKILL when CLI (owner) exits


# ===========================================================================
# target resolve — path only, three-in-one (no name lookup)
# ===========================================================================

def _resolve_target(target: str | None) -> NodeManifest:
    """Resolve target path to NodeManifest. Path only, no name lookup.

    Priority: no-arg → find_upward(cwd) → directory → .py → NODE.md.
    """
    if not target:
        found = NodeManifest.find_upward(Path.cwd())
        if found is None:
            print_error(
                "No NODE.md found upward from current directory. "
                "Provide a path or cd into a node directory."
            )
            raise typer.Exit(code=1)
        return found

    path = Path(target).resolve()
    if not path.exists():
        print_error(f"Path does not exist: {target}")
        raise typer.Exit(code=1)

    if path.is_dir():
        manifest = NodeManifest.read_from_directory(path)
        if manifest is None:
            print_error(f"No {NodeManifest.MANIFEST_FILENAME} found in directory: {path}")
            print_info("  Create one with: moss nodes create <name>")
            raise typer.Exit(code=1)
        return manifest

    if path.suffix == '.py':
        return NodeManifest.from_script(path)

    if path.name == NodeManifest.MANIFEST_FILENAME:
        return NodeManifest.read_from_file(path)

    print_error(f"Cannot resolve target: {target}")
    print_info(
        f"  Expected: directory, {NodeManifest.MANIFEST_FILENAME} file, or .py script."
    )
    raise typer.Exit(code=1)


# ===========================================================================
# discovery: list
# ===========================================================================

@nodes_app.command(name="list")
def list_nodes(
    installed: bool = typer.Option(
        False, "--installed",
        help="Only show nodes marked as installed.",
    ),
    include: list[str] | None = typer.Option(
        None, "--include",
        help="fnmatch pattern to include (repeatable, e.g. 'tools/*').",
    ),
    exclude: list[str] | None = typer.Option(
        None, "--exclude",
        help="fnmatch pattern to exclude (repeatable).",
    ),
):
    """List discovered node manifests (NODE.md scanning under project.nodes)."""
    project = Project.discover()
    manifests = project.nodes.list_nodes(
        refresh=True,
        installed=True if installed else None,
        include=include or None,
        exclude=exclude or None,
    )

    if not manifests:
        print_warning("No nodes found.")
        return

    rows: list[list[str]] = []
    for rel_path, m in manifests.items():
        rows.append([
            m.name,
            str(rel_path),
            "persist" if m.persist else "one-shot",
            "yes" if m.installed else "no",
            (m.description or "")[:80],
        ])

    echo("")
    print_simple_table(
        data=rows,
        headers=["Name", "Path", "Type", "Installed", "Description"],
        title=f"Nodes ({len(rows)} found)",
    )


# ===========================================================================
# discovery: show — preserve truth (raw NODE.md)
# ===========================================================================

@nodes_app.command(name="show")
def show_node(
    path: str = typer.Argument(
        help="Path to node directory, NODE.md file, or .py script.",
    ),
):
    """Show a node's NODE.md verbatim + directory contents.

    Preserves file truth — what you see IS what's on disk. No field parsing,
    no re-rendering. Edit the file directly. Field structure reference:

        moss codex get-interface ghoshell_moss.core.blueprint.cell:NodeManifest
    """
    manifest = _resolve_target(path)

    if not manifest.file:
        print_warning(
            f"Ad-hoc node from {path} (no NODE.md on disk). "
            f"Nothing to show. Use 'moss nodes create' or 'moss nodes link'."
        )
        return

    node_file = Path(manifest.file)
    node_dir = node_file.parent

    # Directory listing
    entries = []
    for item in sorted(node_dir.iterdir()):
        marker = "/" if item.is_dir() else ""
        entries.append(f"  {item.name}{marker}")

    echo("")
    print_info(f"[show] Node file: {node_file}")
    print_info(f"       Directory: {node_dir}")
    print_info("       Contents:")
    for line in entries:
        echo(line)

    echo("")
    print_simple_panel(
        node_file.read_text(encoding='utf-8'),
        title=f"{NodeManifest.MANIFEST_FILENAME} (verbatim)",
    )

    install_md = node_dir / NodeManifest.INSTALL_FILENAME
    installed_marker = node_dir / NodeManifest.INSTALLED_FILE
    if install_md.exists() and not installed_marker.exists():
        echo("")
        print_warning(
            f"Not installed. {NodeManifest.INSTALL_FILENAME} declares steps; "
            f"'run' will refuse until installed."
        )
        print_info(f"  Read: {install_md}")
        print_info(f"  Then: moss nodes install {path}")


# ===========================================================================
# creation: create — scaffold from stub
# ===========================================================================

@nodes_app.command(name="create")
def create_node(
    path: Path = typer.Argument(
        help="Target directory for the new node (e.g. '.moss/nodes/tools/my-node').",
    ),
):
    """Create a new node from the stub template at the given path.

    The directory's last component becomes the node name. All other commands
    (run, show, install, link) already use path — this one does too.
    """
    target_dir = path.resolve()
    name = target_dir.name

    if target_dir.exists():
        print_error(f"Directory already exists: {target_dir}")
        raise typer.Exit(code=1)

    target_dir.mkdir(parents=True, exist_ok=False)

    stub_resources = resources.files(_NODE_STUB_PACKAGE)
    _copy_stub(stub_resources, target_dir, name=name)

    print_success(f"Node '{name}' created at {target_dir}")
    echo("")
    print_info(f"  Read {target_dir / 'README.md'} — what to fill in before running or sharing.")
    print_info(f"  Edit {target_dir / NodeManifest.MANIFEST_FILENAME} — name, exec, instruction body.")
    install_md = target_dir / NodeManifest.INSTALL_FILENAME
    if install_md.exists():
        print_info(f"  Read {install_md} — declares install steps.")
        print_info(f"       Delete it if no install is needed (then the node is installed by default).")
        print_info(f"       Otherwise run the steps, then: moss nodes install {path}")
    print_info(f"  Run: moss nodes run {path}")


def _copy_stub(stub_node, target_dir: Path, *, name: str) -> None:
    """Copy stub files into target_dir, replacing {name} placeholders in text files."""
    for item in stub_node.iterdir():
        if item.name == "__init__.py":
            continue
        target_item = target_dir / item.name
        if item.is_dir():
            target_item.mkdir(exist_ok=True)
            _copy_stub(item, target_item, name=name)
        else:
            content = item.read_bytes()
            try:
                text = content.decode('utf-8')
                text = text.replace("{name}", name)
                target_item.write_text(text)
            except UnicodeDecodeError:
                target_item.write_bytes(content)


# ===========================================================================
# creation: link — A workspace + B script (shortcut, absolute path, no distribution)
# ===========================================================================

@nodes_app.command(name="link")
def link_node(
    workspace_dir: Path = typer.Argument(
        help="Directory A: where the NODE.md shortcut is created (cell workspace).",
        exists=True, file_okay=False, dir_okay=True,
    ),
    script: Path = typer.Argument(
        help="Directory B: absolute path to the target script (any location).",
        exists=True, file_okay=True, dir_okay=False,
    ),
    name: str = typer.Option(
        "", "--name", "-n",
        help="Node name. Defaults to script filename stem.",
    ),
    command: str = typer.Option(
        "", "--command", "-c",
        help="Explicit exec.command (e.g. 'python', '/bin/bash', '/abs/path/to/interp'). "
             "Required — no auto-detection (WW-2: detection is a dead end).",
    ),
):
    """Create a NODE.md in A pointing to a script in B (shortcut, no distribution).

    A is the cell workspace (governance home — runtime files, ledger, logs go here).
    B is where the script lives (may be outside any MOSS workspace). The absolute
    path is written into NODE.md — if B moves, the link breaks (honest failure).

    Not for distribution — this shortcut is local-only. For distributable cells,
    use `moss nodes create` and put the code inside the cell directory.
    """
    a_dir = workspace_dir.resolve()
    b_script = script.resolve()
    node_name = name or b_script.stem

    if not command:
        print_error("--command is required (no auto-detection by extension).")
        print_info("  Common values:")
        print_info("    --command python        # for .py scripts (spawner's sys.executable)")
        print_info("    --command /bin/bash     # for .sh scripts")
        print_info("    --command /abs/binary   # for anything else")
        raise typer.Exit(code=1)

    target_dir = a_dir / node_name
    if target_dir.exists():
        print_error(f"Directory already exists: {target_dir}")
        raise typer.Exit(code=1)

    target_dir.mkdir(parents=True, exist_ok=False)

    manifest = NodeManifest(
        name=node_name,
        description=f"Link to {b_script}",
        singleton=True,
        exec=ExecSpec(command=command, args=str(b_script)),
        instruction=(
            f"This node is a link to an external script:\n\n"
            f"  {b_script}\n\n"
            f"The link uses an absolute path — if the script moves, this node "
            f"stops working. Not intended for distribution."
        ),
    )
    manifest.write_file(target_dir)

    print_success(f"Node '{node_name}' linked at {target_dir}")
    print_info(f"  exec.command: {command}")
    print_info(f"  exec.args:    {b_script}")
    try:
        rel = target_dir.relative_to(Path.cwd())
    except ValueError:
        rel = target_dir
    print_info(f"  Test: moss nodes run {rel}")


# ===========================================================================
# creation: install — mark installed
# ===========================================================================

@nodes_app.command(name="install")
def install_node(
    path: str = typer.Argument(help="Path to node directory or NODE.md."),
):
    """Mark a node as installed (touches .installed marker).

    Does NOT run install steps. Read INSTALL.md, run the steps via bash, then
    call this command.
    """
    manifest = _resolve_target(path)
    if not manifest.file:
        print_error("Cannot install an ad-hoc node (no NODE.md on disk).")
        raise typer.Exit(code=1)

    cell_dir = Path(manifest.file).parent
    install_md = cell_dir / NodeManifest.INSTALL_FILENAME
    if not install_md.exists():
        print_warning(f"No {NodeManifest.INSTALL_FILENAME} in {cell_dir}.")
        print_info("Node requires no installation — nothing to do.")
        return

    installed_file = cell_dir / NodeManifest.INSTALLED_FILE
    installed_file.touch()
    print_success(f"Node '{manifest.name}' marked as installed ({installed_file}).")


# ===========================================================================
# startup: run — foreground blocking, CLI is owner
# ===========================================================================

@nodes_app.command(
    name="run",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
def run_node(
    ctx: typer.Context,
    target: str = typer.Argument(
        None,
        help="Path to node dir / NODE.md / .py script. Omit to find_upward from cwd.",
    ),
):
    """Launch a node cell in the foreground. CLI is owner (Ctrl+C stops cleanly).

    Ctrl+C forwards SIGTERM to the child; 5s grace then SIGKILL bottom-line.
    Extra args after `--` are appended to the child argv:

        moss nodes run nodes/tools/foo -- --port 8000 --debug
    """
    project = Project.discover()
    env = project.env
    manifest = _resolve_target(target)

    if not manifest.installed:
        install_path = (
            Path(manifest.file).parent / NodeManifest.INSTALL_FILENAME
            if manifest.file else "(ad-hoc)"
        )
        print_error(f"Node '{manifest.name}' is not installed.")
        print_info(f"  See {install_path}, run install steps, then:")
        print_info("    moss nodes install <path>")
        raise typer.Exit(code=1)

    launcher = NodeLauncher.from_manifest(env, manifest)
    if ctx.args:
        launcher.run.extend(ctx.args)

    _print_launch_debug(launcher, env)

    try:
        with contextlib.ExitStack() as stack:
            # singleton 冲突: 只读探测给友好提示, 但**不抢锁** — 真锁归子进程
            # (enter_cell_lifecycle fast-fail). 父子进程共抢同一锁曾在 M7 死锁
            # (父抢 → 子等到超时), 见 cell-run-cycle FEATURE.md 与 workspace.py
            # FileLocker 注释. is_locked 是只读探测: _flock_ex_nb 拿到就释放,
            # 无 TOCTOU 死锁; 子进程 fast-fail 作兜底.
            if launcher.runtime.cell.singleton:
                probe = env.workspace.lock(launcher.runtime.locker_name())
                if probe.is_locked():
                    print_error(
                        f"Singleton conflict for '{manifest.name}': lock "
                        f"'{launcher.runtime.locker_name()}' held by another process."
                    )
                    print_info("  moss nodes status         # inspect what's running")
                    print_info("  moss nodes kill <address> # stop the running instance")
                    raise typer.Exit(code=1)

            proc = subprocess.Popen(
                launcher.run,
                cwd=str(launcher.cwd),
                env=launcher.env,
                start_new_session=True,
                # stdout/stderr default = inherit → directly to terminal
            )
            launcher.runtime.pid = proc.pid
            launcher.runtime.pgid = os.getpgid(proc.pid)
            launcher.runtime.write_to_runtime_dir(env.cell_runtimes_dir)

            _forward_signals(proc)

            try:
                proc.wait()
            finally:
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=_RUN_GRACE_SECONDS)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(launcher.runtime.pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        proc.wait()
                launcher.runtime.delete_invalid(env.cell_runtimes_dir)
    except typer.Exit:
        raise

    if proc.returncode != 0:
        echo("")
        print_error(
            f"Node exited abnormally (returncode={proc.returncode}). "
            f"See child stderr above for cause."
        )
    sys.exit(proc.returncode)


def _print_launch_debug(launcher: NodeLauncher, env) -> None:
    """Launch-debug section — printed before spawn.

    Operator sees: who / where / with what env / ledger location. First-order
    debug data for `moss nodes run`.
    """
    cell = launcher.runtime.cell
    ledger_path = CellRuntimeInfo.filepath(env.cell_runtimes_dir, cell.address)
    echo("")
    print_info("[run] Starting node cell")
    print_info(f"  address:   {cell.address}")
    print_info(f"  cwd:       {launcher.cwd}")
    print_info(f"  argv:      {' '.join(launcher.run)}")
    print_info(f"  ledger:    {ledger_path}")
    if cell.singleton:
        print_info(f"  singleton: true (lock '{launcher.runtime.locker_name()}')")
    else:
        print_info("  singleton: false")
    print_info("  env:")
    runtime_env = env.dump_runtime_scope()
    for key in sorted(runtime_env.keys()):
        value = runtime_env[key] or "(empty)"
        print_info(f"    {key:<30s}  {value}")
    print_info("--- child stdout/stderr below ---")


def _forward_signals(proc: subprocess.Popen) -> None:
    """SIGINT/SIGTERM → forward to child. Child owns graceful shutdown logic."""
    def _handler(signum, frame):
        try:
            proc.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            pass
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


def _match_address(info_address: str, query: str) -> bool:
    return CellAddressCodec(info_address).match(query)


def _find_runtime(runtime_dir: Path, query: str) -> list[CellRuntimeInfo]:
    """Collect all runtime infos matching query. Returns [] if none, [one] on unique."""
    return [
        info for info in CellRuntimeInfo.iter_runtime_info(runtime_dir)
        if _match_address(info.address, query)
    ]


def _resolve_single_runtime(
    runtime_dir: Path, query: str, *, action: str,
) -> CellRuntimeInfo | None:
    """Resolve query to exactly one runtime. Print operator hint on ambiguity."""
    matches = _find_runtime(runtime_dir, query)
    if not matches:
        print_error(f"No runtime entry found for '{query}'.")
        return None
    if len(matches) > 1:
        print_error(
            f"Ambiguous '{query}' — matches {len(matches)} runtime entries. "
            f"Use full address to {action}:"
        )
        for m in matches:
            print_info(f"  {m.address}")
        return None
    return matches[0]


# ===========================================================================
# debug: status
# ===========================================================================

@nodes_app.command(name="status")
def status_nodes(
    address: str = typer.Argument(
        "", help="Node address to inspect. Omit to list all runtime entries.",
    ),
):
    """Show runtime status of nodes (reads CellRuntimeInfo files, no matrix)."""
    project = Project.discover()
    runtime_dir = project.env.cell_runtimes_dir

    if address:
        _show_runtime_detail(project, runtime_dir, address)
    else:
        _list_runtime(runtime_dir)


def _list_runtime(runtime_dir: Path) -> None:
    infos = list(CellRuntimeInfo.iter_runtime_info(runtime_dir))
    if not infos:
        print_info("No runtime entries found.")
        return

    rows: list[list[str]] = []
    for info in infos:
        state = "alive" if info.is_alive() else "stale"
        rows.append([
            info.address,
            info.cell.name,
            info.cell.role,
            str(info.pid) if info.pid else "—",
            state,
            "yes" if info.cell.singleton else "no",
        ])
    echo("")
    print_simple_table(
        data=rows,
        headers=["Address", "Name", "Role", "PID", "State", "Singleton"],
        title=f"Runtime Nodes ({len(rows)} found)",
    )


def _show_runtime_detail(project: Project, runtime_dir: Path, address: str) -> None:
    matched = _resolve_single_runtime(runtime_dir, address, action="inspect")
    if matched is None:
        return

    state = "alive" if matched.is_alive() else "stale"

    # Best-effort NodeManifest.description reverse lookup via project.nodes
    manifest_desc = "—"
    try:
        for _, m in project.nodes.list_nodes(refresh=False).items():
            if m.name == matched.cell.name and m.category == matched.cell.category:
                manifest_desc = m.description or "—"
                break
    except Exception as e:
        print_warning(f"Manifest reverse lookup failed: {e}")

    echo("")
    print_simple_table(
        data=[
            ["address", matched.address],
            ["name", matched.cell.name],
            ["role", matched.cell.role],
            ["category", matched.cell.category or "—"],
            ["description", manifest_desc],
            ["state", state],
            ["pid", str(matched.pid)],
            ["pgid", str(matched.pgid)],
            ["start_time", str(matched.start_time)],
            ["singleton", "yes" if matched.cell.singleton else "no"],
            ["project_id", matched.cell.project_id or "—"],
            ["home", matched.cell.home or "—"],
            ["parent", matched.cell.parent_address or "—"],
            ["providing", ", ".join(matched.cell.providing) or "—"],
            ["event_level", matched.cell.event_level.name if matched.cell.event_level else "—"],
            ["ledger", str(CellRuntimeInfo.filepath(runtime_dir, matched.address))],
        ],
        headers=["Property", "Value"],
        title=f"Runtime Node: {matched.address}",
    )


# ===========================================================================
# cleanup: kill — maintenance action, no default defense
# ===========================================================================

@nodes_app.command(name="kill")
def kill_node(
    address: str = typer.Argument(help="Node address to kill."),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Immediate SIGKILL (skip graceful SIGTERM window).",
    ),
):
    """Kill a running node. Default: SIGTERM → 3s grace → SIGKILL. --force: immediate SIGKILL."""
    project = Project.discover()
    runtime_dir = project.env.cell_runtimes_dir

    matched = _resolve_single_runtime(runtime_dir, address, action="kill")
    if matched is None:
        raise typer.Exit(code=1)

    if matched.pgid > 0:
        _graceful_terminate(matched.pgid, force=force)

    matched.delete_invalid(runtime_dir)
    print_success(f"Killed {matched.address} (pid={matched.pid}).")


def _graceful_terminate(pgid: int, *, force: bool) -> None:
    """SIGTERM + short grace → SIGKILL, or --force = immediate SIGKILL.

    Shared by kill and prune. Grace window is _KILL_GRACE_SECONDS.
    """
    if force:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return

    deadline = time.time() + _KILL_GRACE_SECONDS
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)   # signal 0 = liveness probe
        except ProcessLookupError:
            return
        time.sleep(0.1)

    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# ===========================================================================
# cleanup: prune — orphan killer (kills alive by default)
# ===========================================================================

@nodes_app.command(name="prune")
def prune_nodes(
    keep_alive: bool = typer.Option(
        False, "--keep-alive",
        help="Only remove dead entries; leave live orphans running.",
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="Immediate SIGKILL for live orphans (skip graceful window).",
    ),
):
    """Prune stale runtime entries. Default: kills alive orphans (they hold singleton locks).

    --keep-alive: only remove dead entries.
    """
    project = Project.discover()
    runtime_dir = project.env.cell_runtimes_dir

    infos = list(CellRuntimeInfo.iter_runtime_info(runtime_dir))
    if not infos:
        print_info("No runtime entries to prune.")
        return

    killed = 0
    removed = 0
    skipped = 0
    for info in infos:
        if info.is_alive():
            if keep_alive:
                skipped += 1
                continue
            if info.pgid > 0:
                _graceful_terminate(info.pgid, force=force)
            killed += 1
        info.delete_invalid(runtime_dir)
        removed += 1

    msg = f"Pruned {removed} entries ({killed} killed alive)"
    if skipped:
        msg += f", {skipped} live entries kept"
    print_success(msg + ".")
