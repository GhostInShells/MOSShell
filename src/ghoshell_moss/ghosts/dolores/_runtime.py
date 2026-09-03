import asyncio
import contextlib
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

import yaml
from typing_extensions import Self

from ghoshell_moss.contracts.logger import get_moss_logger

from ghoshell_moss.core.blueprint.ghost import Ghost, GhostMeta
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.mindflow import Thinking
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.ground import DefaultGroundSet, Ground
from ghoshell_moss.message import Message

if TYPE_CHECKING:
    from ghoshell_moss.deepseek_harness.launcher import DshLauncher

    from ._ego import DoloresConfig, DoloresEgo, DoloresEgoConfig
    from ._meta import DoloresMeta
    from ghoshell_moss.core.blueprint.shell_trajectory import MShellContextFacade

__all__ = ["Dolores"]

from ._prompts import dolores_inception, dolores_protocol_notice, dolores_terminology


class Dolores(Ghost):
    """Dolores — the second Ghost prototype runtime (DSH reasoning-core integration).

    Its lifecycle mounts a DshLauncher (reasoning core, via matrix.processes) and a DefaultGroundSet
    (ghost_home cognitive field), and holds an ego (DoloresEgo — the session/transaction bridge).
    think() delegates to ego.run_thinking() to drive dsh reasoning, yielding logos segment by segment;
    the articulator is managed here.
    """

    def __init__(
        self,
        *,
        meta: "DoloresMeta",
        home: Path | None = None,
        session: Session | None = None,
        matrix: Matrix | None = None,
        shell: MOSShell | None = None,
        base_instruction: str | None = None,
    ):
        self._meta = meta
        self._home = home
        self._session = session
        self._matrix = matrix
        self._shell = shell
        self._base_instruction = base_instruction
        # launcher / ground are lazy — __init__ touches no httpx / matrix.processes / shell (side-effect free).
        self._dsh_launcher: "DshLauncher | None" = None
        self._ground_set: DefaultGroundSet | None = None
        self._root_ground: Ground | None = None
        # ground render cache — rendered async in __aenter__, read synchronously by memories().
        self._ground_text: str | None = None
        self._exit_stack = contextlib.AsyncExitStack()
        self._ego: "DoloresEgo | None" = None
        self._facade: "MShellContextFacade | None" = None

    # ── Ghost ABC ──────────────────────────────────

    @property
    def meta(self) -> GhostMeta:
        return self._meta

    def system_prompt(self) -> str:
        """instruction = baseline + prototype meta + identity + terminology + protocol + dolores layer.

        baseline is the base_instruction the factory fetched from the container (CTML + project + mode).
        The ghost sections are derived from structured meta, not hardcoded — future cognition is built
        from the directory. Terminology/protocol sections (fence semantics) are not configurable; the
        dolores layer can be replaced via the ego config's inception_template. All sections are static
        (cache-stable) — not steered (tail), not injected per frame.
        """
        parts: list[str] = []
        if self._base_instruction:
            parts.append(self._base_instruction)
        parts.append(self._meta.prototype_instruction())
        parts.append(self._meta.identity_instruction())
        parts.append(dolores_terminology())
        parts.append(dolores_protocol_notice())
        parts.append(self._dolores_instruction())
        return "\n\n".join(parts)

    def _dolores_instruction(self) -> str:
        """The dolores persona/etiquette layer — replaced by a template file if the ego config has one; slots carry runtime paths."""
        template: str | None = None
        if self._home is not None:
            rel = self._load_ego_config().inception_template
            if rel:
                path = self._home / rel
                if path.exists():
                    template = path.read_text(encoding="utf-8")
                else:
                    self.logger.warning(
                        "inception_template %s not found in ghost home; using default", rel
                    )
        env = self._matrix.env if self._matrix is not None else None
        return dolores_inception(
            ghost_home=str(self._home) if self._home is not None else "",
            project_home=str(env.project_path) if env is not None else "",
            mode_home=str(env.mode_home) if env is not None else "",
            template=template,
        )

    async def ground_instruction(self) -> str | None:
        """ground slot — render the held root ground (ghost_home cognitive field) to text.

        The root ground is opened and held long-term in __aenter__ to keep snapshot change-tracking
        (single owner). Called by the epoch cycle; this step only prepares the element, it does not
        actively wire it.
        """
        if self._root_ground is None:
            return None
        view = await self._root_ground.render()
        return str(view)

    def memories(self) -> list[Message]:
        """The ghost's dynamic memory — the ground renders first (existential, at the front).

        The ground text is rendered async in __aenter__ and cached to _ground_text; this method reads
        the cache synchronously, so the ego can fetch the freshest memory via the closure on
        create_session. Clones share the same closure.
        """
        if self._ground_text:
            return [Message.new(tag="ground").with_content(self._ground_text)]
        return []

    async def think(self, thinking: Thinking) -> AsyncIterator[str]:
        """Delegate to ego.run_thinking() to drive dsh reasoning — lifecycle/ending/CTML parsing all live in the run.

        This side only holds the async-with boundary and passes logos through (for the mindflow
        broadcast observability surface). Errors (enter/consume/cancel) propagate naturally through
        async-with, governed by run.__aexit__.
        """
        if self._ego is not None:
            async with self._ego.run_thinking(thinking) as run:
                async for delta in run.logos():
                    yield delta
        else:
            yield ""

    async def __aenter__(self) -> Self:
        await self._exit_stack.__aenter__()
        # file IO is offloaded to a thread; session.output stays on the main loop (avoid cross-thread).
        action = await asyncio.to_thread(self._sync_stubs)
        # always override the plugin stub (active dev artifact, not version-gated) so the latest lands in ghost home.
        await asyncio.to_thread(self._sync_dsh_plugin)
        if action is not None and self._session is not None:
            self._session.output(
                "system",
                f"dolores ghost home {action} (VERSION={self._meta.VERSION})",
                log=f"dolores stubs {action}",
            )
        # open and hold the root ground (ghost_home cognitive field). Stub sync runs first (GROUND.md
        # already written); the GroundSet lifecycle is managed by the exit stack; the memory ground
        # section must render before ego creation.
        if self._home is not None:
            self._ground_set = await self._exit_stack.enter_async_context(
                DefaultGroundSet(workspace_root=self._home)
            )
            self._root_ground = await self._ground_set.open(self._home)
        # render the ground text, cached for synchronous read by memories() (ego create_session consumes it via closure).
        self._ground_text = await self.ground_instruction()
        if self._matrix is not None:
            await self._exit_stack.enter_async_context(self._dsh())
            # ego wiring: create and hold the ego session (via plugin RPC), after dsh is ready.
            # dependency inversion: the ego does not back-ref the ghost; runtime context is injected via ctx/launcher/memories closure.
            from ._ego import DoloresEgo, DoloresEgoContext
            from ghoshell_moss.core.blueprint.shell_trajectory import MShellContextFacade

            self._facade = MShellContextFacade(self._shell)
            ctx = DoloresEgoContext(
                project_home=self._home,
                project_name=f"Ghost {self._meta.name()} Home",
                name=self._meta.name(),
                instruction=self.system_prompt(),
                facade=self._facade,
            )
            self._ego = await self._exit_stack.enter_async_context(
                DoloresEgo(
                    launcher=self.dsh_launcher,
                    ctx=ctx,
                    config=self._load_ego_config(),
                    memories=self.memories,
                )
            )
            # bind the self-wake signal outlet to the MOSS session — matrix.session.add_signal routes to mindflow.
            self._ego.bind_signal_broadcast(self._matrix.session.add_signal)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._facade is not None:
            self._facade.discard()
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    # ── dsh startup ─────────────────────────────────

    @property
    def dsh_launcher(self) -> "DshLauncher":
        """dsh launcher handle — raises a clear error before startup."""
        if self._dsh_launcher is None:
            raise RuntimeError("dsh launcher not started. Call __aenter__ first.")
        return self._dsh_launcher

    @property
    def logger(self) -> logging.Logger:
        """MOSS runtime logger — taken via matrix (current node); falls back when there is no matrix."""
        if self._matrix is not None:
            return self._matrix.logger
        return get_moss_logger()

    def _build_dsh_launcher(self) -> "DshLauncher":
        from ghoshell_moss.deepseek_harness.launcher import DshLauncher

        dsh = self._load_config().dsh
        home = self._resolve_dsh_home(dsh.home)
        launcher_config = dsh.model_copy(update={"home": home})
        return DshLauncher(launcher_config, subprocesses=self._matrix.processes)

    @contextlib.asynccontextmanager
    async def _dsh(self):
        launcher = self._build_dsh_launcher()
        self._dsh_launcher = launcher
        launcher.on_exit(self._on_dsh_exit)
        self._session.output("system", log="starting dsh")
        async with launcher:
            self._session.output(
                "system",
                f"dsh ready at {launcher.config.base_url}",
                log="dsh ready",
            )
            yield

    def _on_dsh_exit(self, exit_info) -> None:
        if self._session is None:
            return
        if exit_info.self_shutdown or exit_info.exit_code in (0, None):
            self._session.output("system", log="dsh exited")
        else:
            self._session.output(
                "error",
                log=f"dsh exited code {exit_info.exit_code}: {exit_info.stderr}",
            )

    # ── stub sync ───────────────────────────────────

    def _sync_stubs(self) -> str | None:
        """Sync the skeleton into ghost home. Returns 'init' | 'override' | None (no-op).

        No-op when VERSION matches; init when missing, override when mismatched (fully overwrites the
        skeleton files, never touches dynamic data files in home). Also materializes dirs + dsh_home.
        """
        if self._home is None:
            return None
        target = self._meta.VERSION
        current = self._load_config().version
        if current == target:
            return None
        action = "override" if current else "init"
        shutil.copytree(self._meta.stubs_dir(), self._home, dirs_exist_ok=True)
        self._materialize_dirs()
        self._sync_dsh_home()
        self._write_version(target)
        return action

    def _load_config(self) -> "DoloresConfig":
        from ._ego import DoloresConfig

        marker = self._home / ".dolores.yml"
        if not marker.exists():
            return DoloresConfig()
        data = yaml.safe_load(marker.read_text(encoding="utf-8")) or {}
        return DoloresConfig(**data)

    def _write_version(self, version: str) -> None:
        """Write back the version (stub-sync marker); the rest of the config is reloaded from the current file and kept as-is."""
        config = self._load_config()
        config.version = version
        marker = self._home / ".dolores.yml"
        marker.write_text(
            yaml.safe_dump(
                config.model_dump(mode="json", exclude_defaults=True, exclude_none=True),
                allow_unicode=True,
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    def _load_ego_config(self) -> "DoloresEgoConfig":
        return self._load_config().ego

    def _materialize_dirs(self) -> None:
        for d in self._load_config().dirs:
            (self._home / d).mkdir(parents=True, exist_ok=True)

    def _sync_dsh_home(self) -> None:
        shutil.copytree(
            self._meta.dsh_stubs_dir(),
            self._home / ".dsh",
            dirs_exist_ok=True,
        )

    def _sync_dsh_plugin(self) -> None:
        """Copy the plugin stub into ghost home — always override, not version-gated.

        The plugin is an active dev artifact (dsh kernel-privilege bridge) that changes far more often
        than the skeleton files, which are version-gated; the plugin is pulled fresh on every startup.
        """
        if self._home is None:
            return
        target = self._home / ".dsh" / "profiles" / "web" / "plugin.ts"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._meta.dsh_plugin_stub(), target)

    def _resolve_dsh_home(self, home: str | Path | None) -> Path:
        if home is None:
            return self._home / ".dsh"
        p = Path(home)
        return p if p.is_absolute() else (self._home / p)
