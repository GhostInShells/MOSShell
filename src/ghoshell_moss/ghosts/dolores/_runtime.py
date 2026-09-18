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
from ghoshell_moss.core.blueprint.mindflow import Mindflow, Thinking
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.host import MossSystemPrompter
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.ground import DefaultGroundSet, Ground
from ghoshell_moss.message import Message

if TYPE_CHECKING:
    from ghoshell_moss.core.blueprint.channel_builder import MutableChannel
    from ghoshell_moss.core.blueprint.shell_trajectory import MShellContextFacade
    from ghoshell_moss.deepseek_harness.launcher import DshLauncher
    from ghoshell_moss.memento._fs_memento import FsMemento

    from ._ego import DoloresConfig, DoloresEgo, DoloresEgoConfig
    from ._ego_memento import EgoMementoManager
    from ._meta import DoloresMeta

__all__ = ["Dolores"]

from ._prompts import (
    dolores_inception, dolores_memento, dolores_output_protocol_notice, dolores_terminology,
    DOLORES_INSTRUCTION_END,
)

# ego 轨迹索引的根目录 (相对 ghost_home); branch 名由 memento config 决定.
_EGO_MEMENTO_DIR = ".memento/ego"


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
        moss_prompter: "MossSystemPrompter | None" = None,
        base_instruction: str | None = None,
    ):
        self._moss_prompter: MossSystemPrompter = moss_prompter
        self._meta = meta
        self._home = home
        self._session = session
        self._matrix = matrix
        self._shell = shell
        self._base_instruction = base_instruction
        # launcher / ground are lazy — __init__ touches no httpx / matrix.processes / shell (side-effect free).
        self._dsh_launcher: DshLauncher | None = None
        self._ground_set: DefaultGroundSet | None = None
        self._root_ground: Ground | None = None
        # ground render cache — rendered async in __aenter__, read synchronously by memories().
        self._ground_text: str | None = None
        # memento: ego 轨迹索引 + 旁路服务. manager 由 ghost 持有, 同时注入 ego.
        self._memento: FsMemento | None = None
        self._memento_manager: EgoMementoManager | None = None
        self._exit_stack = contextlib.AsyncExitStack()
        self._ego: DoloresEgo | None = None
        self._facade: MShellContextFacade | None = None
        # Held by the ghost (not the runtime) so the ghost can wire faculties onto it and
        # hand the same instance to controllers. Materialized on first mindflow() call.
        self._mindflow: Mindflow | None = None
        # reflexive control channel — built lazily in channel(), registered by the runtime as 'ghost'.
        self._channel: MutableChannel | None = None
        # the cognition epoch is opened once, on the first thinking (see think()).
        self._epoch_opened: bool = False

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
        if self._moss_prompter is not None:
            parts.append(self._moss_prompter.moss_meta_instruction())

        parts.append(dolores_terminology())
        parts.append(dolores_memento())
        parts.append(self._dolores_inception())
        parts.append(self._meta.prototype_instruction())
        parts.append(self._meta.identity_instruction())
        if self._base_instruction:
            parts.append(self._base_instruction)

        if self._moss_prompter is not None:
            parts.append("<!-- current moss project -->\n" + self._moss_prompter.project_instruction())
            parts.append("<!-- current moss mode -->\n" + self._moss_prompter.mode_instruction())

        parts.append(dolores_output_protocol_notice())
        parts.append(DOLORES_INSTRUCTION_END)
        return "\n---\n".join([part.strip() for part in parts]) + "\n<!-- deepseek harness instruction -->"

    def _dolores_inception(self) -> str:
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

    def channel(self) -> "MutableChannel | None":
        """The ghost's reflexive control channel — its own organs as sub-channels.

        The runtime calls this once, after __aenter__ (the ground set and the memento manager are
        held by then), and registers the result as the ``ghost`` channel. Cached so every caller
        gets the same instance: the channel closes over live resources (shared GroundSet, memento
        manager), and rebuilding it would hand out a second view of the same organs.
        """
        if self._home is None or self._ground_set is None:
            return None
        if self._channel is None:
            from .channel import build_dolores_channel

            self._channel = build_dolores_channel(
                groundset=self._ground_set,
                workspace_root=self._home,
                memento_manager=self._memento_manager,
                memento_root=self._home / _EGO_MEMENTO_DIR,
            )
        return self._channel

    def memories(self) -> list[Message]:
        """The ghost's dynamic memory — one ``<memory>`` container, not loose messages.

        Children: the ground frame (existential field, rendered async in __aenter__ and cached) and
        the memento ``<branch>`` view (the trajectory outline). Each is optional; with neither
        present the container is dropped rather than sent empty.

        The cognition epoch is deliberately *not* here: it opens on the first thinking, after the
        runtime wires the shell facade (see think()), so at session-creation time there is no epoch
        to report yet. It reaches the model through the ego's own epoch slot instead.

        Read by the ego's create_session through this closure (clones share it).
        """
        children: list[Message] = []
        if self._ground_text:
            children.append(Message.new(tag="ground").with_content(self._ground_text))
        if self._memento_manager is not None:
            view = self._memento_manager.view_message()
            if view is not None:
                children.append(view)
        if not children:
            return []
        return [Message.new(tag="memory").with_messages(*children)]

    def mindflow(self) -> Mindflow:
        """Return the mindflow Dolores owns, materializing it on first call.

        The ghost is the owner, not the runtime: it hands back the same instance the GhostRuntime
        will register (container + runtime) and start, so a faculty can hold a handle to the live
        perception/thought/action arbitration and wire nuclei/signals onto it. Default composition
        (InputSignalNucleus) is preserved; the runtime start is left to MOSS.
        """
        if self._mindflow is None:
            from ghoshell_moss.core.mindflow import new_default_mindflow

            self._mindflow = new_default_mindflow(logger=self.logger)
        return self._mindflow

    async def think(self, thinking: Thinking) -> AsyncIterator[str]:
        """Delegate to ego.run_thinking() to drive dsh reasoning — lifecycle/ending/CTML parsing all live in the run.

        This side only holds the async-with boundary and passes logos through (for the mindflow
        broadcast observability surface). Errors (enter/consume/cancel) propagate naturally through
        async-with, governed by run.__aexit__.

        Opening the cognition epoch happens here, on the first thinking: the epoch's baseline comes
        from the shell facade, wired by the runtime *after* ghost.__aenter__ (ghost_runtime step 5),
        so opening it earlier would snapshot an empty baseline. The epoch is then delivered by the
        ego's own epoch slot on the first frame.
        """
        if not self._epoch_opened:
            self._epoch_opened = True
            self.mindflow().moments.new_epoch([])
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
        # always override the ego preset (independent composition, follows dsh 升级 rebase).
        await asyncio.to_thread(self._sync_dsh_preset)
        if action is not None and self._session is not None:
            self._session.output(
                "system",
                f"dolores ghost home {action} (VERSION={self._meta.VERSION})",
                log=f"dolores stubs {action}",
            )
        # load the ghost-home .env into the process environment (supplement only, outer env wins) —
        # the source of startup-time config items (dsh web auto-open, etc.). The ghost owns this file.
        if self._home is not None:
            self._load_env()
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
            # memento: the ego trajectory index at ghost_home/.memento/ego; the branch is the ego line.
            if self._home is not None:
                from ghoshell_moss.memento import new_local_memento

                branch_name = self._load_config().memento.branch_name
                self._memento = new_local_memento(self._home / _EGO_MEMENTO_DIR)
                if self._memento.get_branch(branch_name) is None:
                    self._memento.create_branch(branch_name)
            # ego wiring: create and hold the ego session (via plugin RPC), after dsh is ready.
            # dependency inversion: the ego does not back-ref the ghost; runtime context is injected via ctx/launcher/memories closure.
            from ._ego import DoloresEgo, DoloresEgoContext
            from ghoshell_moss.core.blueprint.shell_trajectory import MShellContextFacade

            self._facade = MShellContextFacade(self._shell)
            ctx = DoloresEgoContext(
                project_home=self._matrix.env.project_path,
                project_name=f"{self._meta.name()} @ {self._matrix.env.project_name}",
                name=self._meta.name(),
                mode=self._matrix.env.mode_name,
                instruction=self.system_prompt(),
                facade=self._facade,
            )
            # memento 旁路服务: ghost 级持有, 注入 ego (锚点写入 / 阈值判定都过它).
            # 生命周期入 exit stack —— 旁路任务归它治理, ghost 关停时随栈取消.
            if self._memento is not None:
                from ._ego_memento import EgoMementoManager

                self._memento_manager = await self._exit_stack.enter_async_context(
                    EgoMementoManager(
                        connection=self.dsh_launcher,
                        memento=self._memento,
                        config=self._load_config().memento,
                        logger=self.logger,
                    )
                )
            self._ego = await self._exit_stack.enter_async_context(
                DoloresEgo(
                    launcher=self.dsh_launcher,
                    ctx=ctx,
                    config=self._load_ego_config(),
                    memories=self.memories,
                    memento_manager=self._memento_manager,
                )
            )
            # bind the self-wake signal outlet to the MOSS session — matrix.session.add_signal routes to mindflow.
            self._ego.bind_signal_broadcast(self._matrix.session.add_signal)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._facade is not None:
            self._facade.discard()
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    # ── startup (born) ───────────────────────────────

    async def startup(self) -> None:
        """Lifecycle hook (born) — read the mode startup doc and emit a self-wake signal.

        Called by GhostRuntime after mindflow wiring (signal routing registered, main loops
        started). Reads ``startup/{mode}.startup.yml`` (fallback ``default.startup.yml``);
        when both command and instruction are empty the boot is silent (no signal).
        """
        if self._matrix is None:
            return
        loaded = await asyncio.to_thread(self._load_startup)
        if loaded is None:
            return
        command, instruction = loaded
        if not command and not instruction:
            return
        from .nucleus import new_dolores_ego_signal

        signal = new_dolores_ego_signal(kind="startup", command=command, instruction=instruction)
        self._matrix.session.add_signal(signal)

    def _resolve_startup_doc(self) -> Path | None:
        """Resolve the current mode's startup doc, falling back to the default."""
        if self._home is None:
            return None
        startup_dir = self._home / "startup"
        mode_name = self._matrix.env.mode_name if self._matrix is not None else ""
        if mode_name:
            doc = startup_dir / f"{mode_name}.startup.yml"
            if doc.exists():
                return doc
        default = startup_dir / "default.startup.yml"
        if default.exists():
            return default
        return None

    def _load_startup(self) -> tuple[str, str] | None:
        """Parse the resolved startup doc into (command, instruction)."""
        doc = self._resolve_startup_doc()
        if doc is None:
            return None
        try:
            data = yaml.safe_load(doc.read_text(encoding="utf-8")) or {}
        except Exception as e:
            self.logger.warning("startup doc %s parse failed: %s", doc, e)
            return None
        command = str(data.get("command") or "").strip()
        instruction = str(data.get("instruction") or "").strip()
        return command, instruction

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
            url = launcher.web_url() or f"{launcher.config.base_url}/?token={launcher.token()}"
            self._session.output(
                "system",
                f"dsh ready at {url}",
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

    def _load_env(self) -> None:
        """Load ``<ghost_home>/.env`` into the process environment (override=False, outer env wins)."""
        import dotenv

        dotenv.load_dotenv(self._home / ".env", override=False)

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
        target = self._home / ".dsh" / "profiles" / "web" / "moss-dolores-ghost-plugin.ts"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self._meta.dsh_plugin_stub(), target)

    def _sync_dsh_preset(self) -> None:
        """Copy the repo-owned dolores-ego preset into ghost home — always override.

        The preset is the independent (non-standard) agent composition, authored in-repo under
        dsh_preset/. Copied fresh every startup so dsh 升级 rebase 落在仓库文件上, 不在运行态.
        """
        if self._home is None:
            return
        shutil.copytree(
            self._meta.dsh_preset_dir(),
            self._home / ".dsh" / ".agent-presets",
            dirs_exist_ok=True,
        )

    def _resolve_dsh_home(self, home: str | Path | None) -> Path:
        if home is None:
            return self._home / ".dsh"
        p = Path(home)
        return p if p.is_absolute() else (self._home / p)
