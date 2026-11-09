"""The dsh node — one dsh process, its page announced, a minimal channel.

This is the first node of the dsh family: it owns a dsh 0.2.0 web process through the
engine's launcher, announces the dsh web surface as a webview (a screen adopts it by
itself), and keeps the cell alive with a small channel.

Two responsibilities, deliberately separated:

- **install** — :func:`prepare_home` initializes a node home once: the dsh home
  (``.dsh``), the memento envelope (``.memento``) and the node's ``.gitignore``. It is
  idempotent and never reprovisions an existing ``.dsh``: only the plugin bundle is
  replaced, and settings / storages / workspace.json are left alone. Invoked by the
  install step in ``INSTALL.md`` (``--install``), not by the runtime.
- **serve** — :class:`DshNodeRuntime` starts the launcher and :func:`main` provides the
  channel. The runtime assumes the home was installed; a missing environment is the
  probe's job (``check.py``) before spawn, not a second informal check here.

The agents surface is a **gated child** of this channel: the node declares it and the ghost opens
it with one action (`mount_child`). Progressive disclosure done by the channel — the node stays
small until something is actually being driven.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from ghoshell_dsh.v0_2_0 import DSH_VERSION
from ghoshell_dsh.v0_2_0.install import install_plugin, installed_patch
from ghoshell_dsh.v0_2_0.launcher import DshLauncher, DshLauncherConfig, resolve_dsh_binary

from ghoshell_moss.contracts.subprocesses import Subprocesses
from ghoshell_moss.deepseek_harness.v0_2_0.channel import workspace_row
from ghoshell_moss.deepseek_harness.v0_2_0.hive import DshHive
from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel
from ghoshell_moss.core.blueprint.states_channel import PrimeChannel, new_prime_channel
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.services.webview import WebViewDeclaration, WebViewServer

__all__ = [
    "DEFAULT_PORT",
    "DshNodeHome",
    "DshNodeRuntime",
    "main",
    "new_dsh_node_channel",
    "parse_args",
    "prepare_home",
    "run_node",
]

CHANNEL_NAME = "dsh"
"""The channel's own name — the mesh path comes from the cell alias, not from this."""

DEFAULT_PORT = 23080
"""The dsh node family's port band starts here (23080, 23081, ...)."""

DEFAULT_PROFILE = "web"
DEFAULT_TITLE = f"dsh {DSH_VERSION}"

DSH_HOME_DIR = ".dsh"
MEMENTO_DIR = ".memento"

GITIGNORE = """\
# Runtime state of the dsh node — never committed.
.dsh/
.memento/
.installed
.env
__pycache__/
*.pyc

# Cell runtime logs (stdout/stderr of each launch).
runtime/*
!runtime/.gitignore
"""


@dataclass(frozen=True, slots=True)
class DshNodeHome:
    """What an initialized node home is made of."""

    root: Path
    dsh_home: Path
    memento: Path
    gitignore: Path
    plugin_patch: Path
    binary: str


def prepare_home(root: str | Path, *, profile: str = DEFAULT_PROFILE) -> DshNodeHome:
    """Initialize a node home once — idempotent, and never reprovisioning an existing home.

    A missing ``.dsh`` is provisioned by dsh itself first (``--dump-config``, no boot).
    An existing one is somebody's home already: only the plugin bundle is replaced, and
    settings / storages / workspace.json are not touched. The rest of the layout is
    created when absent.
    """
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    dsh_home = root / DSH_HOME_DIR
    memento = root / MEMENTO_DIR
    memento.mkdir(exist_ok=True)
    gitignore = root / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(GITIGNORE, encoding="utf-8")
    binary = resolve_dsh_binary()
    patch = install_plugin(
        home=dsh_home,
        profile=profile,
        binary=binary,
        init=not dsh_home.exists(),
    )
    return DshNodeHome(
        root=root,
        dsh_home=dsh_home,
        memento=memento,
        gitignore=gitignore,
        plugin_patch=patch,
        binary=binary,
    )


class DshNodeRuntime:
    """The cell's dsh process — started on channel startup, stopped with the cell."""

    def __init__(
        self,
        root: str | Path,
        *,
        profile: str = DEFAULT_PROFILE,
        port: int = DEFAULT_PORT,
        subprocesses: Subprocesses | None = None,
        title: str = DEFAULT_TITLE,
        logger: logging.Logger | None = None,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._profile = profile
        self._port = int(port)
        self._title = title
        self._subprocesses = subprocesses
        self._logger = logger or logging.getLogger("moss.dsh_node")
        self._launcher: DshLauncher | None = None
        self._url: str = ""

    @property
    def root(self) -> Path:
        return self._root

    @property
    def launcher(self) -> "DshLauncher | None":
        """The live launcher once started — the connection the hive is built on."""
        return self._launcher

    @property
    def title(self) -> str:
        return self._title

    @property
    def port(self) -> int:
        """The live port once started; the configured one before that."""
        if self._launcher is not None:
            return self._launcher.config.port
        return self._port

    async def start(self) -> str:
        """Bring the dsh up; returns its web URL (with the launch token)."""
        if self._launcher is not None:
            return self._url
        dsh_home = self._root / DSH_HOME_DIR
        config = DshLauncherConfig(
            home=dsh_home,
            profile=self._profile,
            port=self._port,
            plugin_patch=installed_patch(dsh_home),
            args=["--no-open"],
            readiness_timeout=60.0,
        )
        launcher = DshLauncher(config, subprocesses=self._subprocesses)
        await launcher.__aenter__()
        self._launcher = launcher
        self._url = launcher.web_url() or f"{config.base_url}/?token={launcher.token()}"
        return self._url

    async def close(self) -> None:
        launcher, self._launcher = self._launcher, None
        if launcher is not None:
            await launcher.__aexit__(None, None, None)

    def web_url(self) -> str:
        return self._url

    def status(self) -> str:
        running = self._launcher is not None and self._launcher.is_running()
        lines = [
            f"dsh {DSH_VERSION}",
            f"home: {self._root}",
            f"dsh home: {self._root / DSH_HOME_DIR}",
            f"memento: {self._root / MEMENTO_DIR}",
            f"profile: {self._profile}",
            f"port: {self.port}",
            f"running: {running}",
            f"web: {self._url or '-'}",
        ]
        return "\n".join(lines)

    def notice(self, name: str = "") -> str:
        """The one-line self-description. ``name`` is the hive's, i.e. this node's own name.

        The name already carries the engine version by convention (``dsh_0_2_0_rc_2``), so the
        notice does not repeat it: one identity, stated once, in the same words the channel path
        and every callback use.
        """
        running = self._launcher is not None and self._launcher.is_running()
        state = "running" if running else "stopped"
        head = name or f"dsh {DSH_VERSION}"
        return f"{head} · {state} · :{self.port} · home {self._root.name}"


def new_dsh_node_channel(
    runtime: DshNodeRuntime,
    hive: "DshHive | None" = None,
    *,
    name: str = CHANNEL_NAME,
) -> PrimeChannel:
    """The node's own channel — and the parent that *owns* the agents organ.

    ``gate=True`` declares the agents channel without mounting it. The framework already ships the
    verbs that govern this (``mount_child`` / ``unmount_child``), so nothing here reimplements
    them: the child is listed in the notice as closed, and the ghost opens it with one action.
    """
    chan = new_prime_channel(
        gate=True,
        name=name,
        description="a dsh hive — one dsh web process, its page as a window, and the agents it holds.",
    )

    @chan.build.notice
    def _notice() -> str:
        line = runtime.notice(name)
        if hive is not None:
            # One number, and only one: whether opening the child is worth an action.
            line += f" · agents: {len(hive.agents().list_sessions())}"
        return line

    @chan.build.instruction
    def _instruction() -> str:
        return (
            "A dsh node: it owns one dsh web process — its page is a window on the screen — and "
            "holds the agent sessions you drive through it.\n\n"
            "- `status` — home, profile, port, running state, page URL.\n"
            "- `web_url` — the page URL, token included.\n"
            "- `workspaces` — the declared places an agent may be created in, each with the "
            "directory it bounds. `create_workspace(path, name)` declares a new one.\n"
            "- `agents` — the agents, as a child channel. It stays closed until `mount_child` "
            "opens it; open it when you actually need to drive one."
        )

    if hive is not None:
        agents_channel = hive.agents().as_channel()

        @chan.build.virtual_children
        def _children() -> dict[str, MutableChannel]:
            # Declared, not mounted: the catalog. Mounting is the command below.
            return {"agents": agents_channel} if agents_channel is not None else {}

    @chan.build.command(name="workspaces", blocking=False, always_observe=True)
    async def _workspaces() -> str:
        """Cold read: the declared places, each with the directory it bounds.

        A place is the permission boundary an agent acts inside, so the path is part of the answer,
        not decoration — and the id is what `open` resolves behind the name.
        """
        if hive is None:
            return "failed: this node holds no agents runtime"
        places = hive.workspaces()
        if not places:
            return "(no workspaces)"
        return "\n".join(workspace_row(item) for item in places)

    @chan.build.command(name="create_workspace", blocking=False, always_observe=True)
    async def _create_workspace(path: str, name: str = "") -> str:
        """Declare a place inside the node's home — nothing is predefined.

        A workspace is a directory an agent may act inside; creating one is a runtime act, not
        something the environment hands you. The call returns only after the engine has reported
        the place back, so a following `open` can use it immediately.

        :param path: resolved against the node's home. Outside it is refused, never clamped.
        :param name: what callers address it by from then on; empty = the engine names it.
        :return: `ok: <name> · <id>`, or `failed: <reason>`.
        """
        if hive is None:
            return "failed: this node holds no agents runtime"
        try:
            place = await hive.create_workspace(path, name)
        except Exception as exc:  # noqa: BLE001 — a refused place is a fact, not a crash
            return f"failed: {exc}"
        return f"ok: {place.name or place.id} · {place.id}"

    @chan.build.command(name="status", blocking=False, always_observe=True)
    async def _status() -> str:
        """Home, profile, port, running state, and the dsh web URL."""
        return runtime.status()

    @chan.build.command(name="web_url", blocking=False, always_observe=True)
    async def _web_url() -> str:
        """The dsh web URL, including the per-launch token."""
        return runtime.web_url() or "(not running)"

    return chan


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="dsh-node", description="dsh 0.2.0 node")
    parser.add_argument("--home", default="", help="node home; empty = the cell's home")
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="dsh process profile")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"dsh web port (the node family starts at {DEFAULT_PORT})",
    )
    parser.add_argument("--title", default=DEFAULT_TITLE, help="the webview's title")
    parser.add_argument(
        "--install", action="store_true",
        help="initialize the node home and exit (the INSTALL.md step)",
    )
    return parser.parse_args(argv)


async def main(matrix: Matrix) -> None:
    """The cell main: start dsh, serve the channel, announce its page — governed by the Matrix.

    Every dependency comes from the Matrix container rather than from this file: the subprocess
    manager, the logger, and the signal channel. The hive's runtime is entered inside the same
    ``async with``, so the node's own lifetime governs it (倒挂): when the cell goes down, the
    agents go down with it, not the other way round.
    """
    args = parse_args(sys.argv[1:])
    root = Path(args.home).expanduser() if args.home else Path(matrix.home)
    runtime = DshNodeRuntime(
        root,
        profile=args.profile,
        port=args.port,
        subprocesses=matrix.processes,
        title=args.title,
        logger=matrix.logger,
    )
    url = await runtime.start()
    print(f"[dsh-node] home: {runtime.root}", flush=True)
    print(f"[dsh-node] dsh web: {url}", flush=True)

    # The node's name comes from its own declaration (``NODE.md`` -> the cell), never from the
    # alias a launcher happened to pick: one name for the declaration, the channel and every
    # callback, so a result can always be walked back to the thing that produced it.
    hive_name = getattr(matrix.this, "name", "") or CHANNEL_NAME
    hive = DshHive(
        runtime.launcher,
        name=hive_name,
        root=matrix.project_home,
        # The signal channel belongs to the Matrix session; the hive only holds it.
        send_signal=matrix.session.add_signal,
        logger=matrix.logger,
    )
    channel = new_dsh_node_channel(runtime, hive, name=hive_name)
    try:
        async with hive:
            await hive.refresh_if_stale()
            await WebViewServer.serve(
                matrix,
                WebViewDeclaration(
                    url=url,
                    title=runtime.title,
                    description="the dsh web surface owned by this node",
                    icon="brain",
                ),
            )
            await matrix.provide_channel(channel)
    finally:
        await runtime.close()
        print("[dsh-node] stopped", flush=True)


def run_node() -> None:
    """Node entry — install a home, or start the cell and serve."""
    args = parse_args(sys.argv[1:])
    if args.install:
        root = Path(args.home).expanduser() if args.home else Path.cwd()
        home = prepare_home(root, profile=args.profile)
        print(f"[dsh-node] home initialized: {home.root}")
        print(f"  dsh home:  {home.dsh_home}")
        print(f"  memento:   {home.memento}")
        print(f"  gitignore: {home.gitignore}")
        print(f"  plugin:    {home.plugin_patch}")
        print(f"  binary:    {home.binary}")
        return
    Matrix.discover().run(main)


if __name__ == "__main__":
    run_node()
