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

The Hive surface is not part of this generation. The channel exposes environment reads
only.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from ghoshell_dsh.v0_2_0 import DSH_VERSION
from ghoshell_dsh.v0_2_0.install import install_plugin, installed_patch
from ghoshell_dsh.v0_2_0.launcher import DshLauncher, DshLauncherConfig, resolve_dsh_binary

from ghoshell_moss.contracts.subprocesses import Subprocesses
from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel
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
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        self._profile = profile
        self._port = int(port)
        self._title = title
        self._subprocesses = subprocesses
        self._launcher: DshLauncher | None = None
        self._url: str = ""

    @property
    def root(self) -> Path:
        return self._root

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

    def notice(self) -> str:
        running = self._launcher is not None and self._launcher.is_running()
        state = "running" if running else "stopped"
        return f"dsh {DSH_VERSION} · {state} · :{self.port} · {self._root.name}"


def new_dsh_node_channel(runtime: DshNodeRuntime) -> MutableChannel:
    """The minimal channel: where this node is, and how to reach its page."""
    chan = new_channel(
        name=CHANNEL_NAME,
        description="dsh node — one dsh web process, its page announced as a window.",
    )

    @chan.build.notice
    def _notice() -> str:
        return runtime.notice()

    @chan.build.instruction
    def _instruction() -> str:
        return (
            "A dsh 0.2.0 node. The cell owns a dsh web process and announces its page as a "
            "window. Use `status` for home / profile / port / web, and `web_url` for the full "
            "URL (it carries the per-launch token).\n\n"
            "This generation exposes environment reads only — the delegation surface (hive) "
            "is not wired yet."
        )

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
    """The cell main: start dsh, announce its page, serve the channel until closed."""
    args = parse_args(sys.argv[1:])
    root = Path(args.home).expanduser() if args.home else Path(matrix.home)
    runtime = DshNodeRuntime(
        root,
        profile=args.profile,
        port=args.port,
        subprocesses=matrix.processes,
        title=args.title,
    )
    url = await runtime.start()
    print(f"[dsh-node] home: {runtime.root}", flush=True)
    print(f"[dsh-node] dsh web: {url}", flush=True)
    await WebViewServer.serve(
        matrix,
        WebViewDeclaration(
            url=url,
            title=runtime.title,
            description="the dsh web surface owned by this node",
            icon="brain",
        ),
    )
    channel = new_dsh_node_channel(runtime)
    try:
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
