import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

import yaml

from ._meta import DoloresProviderSpec
from typing_extensions import Self

from ghoshell_moss.contracts.logger import get_moss_logger

from ghoshell_moss.core.blueprint.ghost import Ghost, GhostEvent, GhostMeta, Logos
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
    from ._startup import StartupDoc

__all__ = ["Dolores"]

from ._prompts import (
    dolores_inception, dolores_memory, dolores_output_protocol_notice, dolores_terminology,
    DOLORES_INSTRUCTION_END,
)

# ego 轨迹索引的根目录 (相对 ghost_home); branch 名由 memento config 决定.
_EGO_MEMENTO_DIR = ".memento/ego"

# settings.yaml 里承载默认模型的 section 名 (dsh-agent-default-model 的 settings namespace).
# 该段由 ghost 在首次 seed 时按 DoloresMeta.default_model 一次性写入, 之后归 ghost home 自治
# (dsh web Models 面可编辑). 多模态是现代模型默认, 不再做视觉强改.
_SETTINGS_MODEL_SECTION = "agent-default-model"
_SETTINGS_FILE = "settings.yaml"
# cordis patch — 承载 provider 覆盖的声明段. 同样 seed-once.
_CORDIS_PATCH = "profiles/web/cordis.patch.yml"
_PI_AI_ID = "llm-pi-ai"
# 可安全写成 YAML plain scalar 的形状; 其余转成 JSON 双引号串 (合法 YAML 标量).
_PLAIN_SCALAR = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _yaml_scalar(value: str) -> str:
    return value if _PLAIN_SCALAR.fullmatch(value) else json.dumps(value)


def _seed_model_section(text: str, provider: str, model: str) -> str | None:
    """Append the ``agent-default-model`` section if it is absent — otherwise return ``None``.

    **Seed-once, never overwrite.** The settings doc is owned by dsh (and by whoever is driving
    the Models page); once the section exists, the ghost home owns the choice. Returning None
    tells the caller to leave the file untouched. Textual on purpose: dsh's YAML 1.2 emitter and
    PyYAML's YAML 1.1 round-trip disagree on tokens like ``off`` (bool vs string), so we never
    rewrite bytes we did not place.
    """
    header = f"{_SETTINGS_MODEL_SECTION}:"
    if any(line.rstrip() == header for line in text.splitlines()):
        return None
    lines = text.splitlines()
    if lines and lines[-1].strip():
        lines.append("")
    lines.append(header)
    lines.append(f"  provider: {_yaml_scalar(provider)}")
    lines.append(f"  model: {_yaml_scalar(model)}")
    return "\n".join(lines) + "\n"


def _render_provider_block(providers: list[DoloresProviderSpec]) -> dict:
    """Project the meta's provider declarations into the dict shape pi-ai's cordis patch expects.

    Reads ``base_url_env`` *at seed time* from ``os.environ`` and writes the resolved URL into
    the patch; ``api_key_env`` is passed through as a name (dsh resolves it at runtime). The
    patch file lives inside the gitignored ghost home, so a literal URL here never reaches the
    repo — but the meta itself only carries variable *names*, keeping repo-visible Python clean.
    """
    out: dict[str, dict] = {}
    for spec in providers:
        entry: dict = {"apiKeyEnv": spec.api_key_env}
        if spec.base_url_env:
            resolved = os.environ.get(spec.base_url_env)
            if resolved:
                entry["baseURL"] = resolved
        if spec.models:
            entry["models"] = list(spec.models)
        if spec.display_name:
            entry["displayName"] = spec.display_name
        out[spec.route] = entry
    return {"providers": out}


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
        # branch view render cache — rendered async in __aenter__ (it runs the branch_view slot),
        # read synchronously by memories(). One render per boot: the plan is a snapshot of the
        # directory at the moment the session opens, which is exactly what boot memory is.
        self._memento_view: Message | None = None
        self._exit_stack = contextlib.AsyncExitStack()
        self._ego: DoloresEgo | None = None
        self._facade: MShellContextFacade | None = None
        # Held by the ghost (not the runtime) so the ghost can wire faculties onto it and
        # hand the same instance to controllers. Materialized on first mindflow() call.
        self._mindflow: Mindflow | None = None
        # reflexive control channel — built lazily in channel(), registered by the runtime as 'ghost'.
        self._channel: MutableChannel | None = None
        # startup doc (StartupDoc) — loaded once in __aenter__, read by channel() and startup().
        self._startup_doc: "StartupDoc | None" = None
        # the cognition epoch is opened on the first thinking and again after every compact (see think()).
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
        parts.append(dolores_memory())
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

    def _frame_root(self) -> "Path | None":
        """Frame 发现根目录 — 边界锁在 project home 的 .ai_partners/frames.

        无 matrix (纯测试 / 未入网) 时返回 None, channel() 不挂 frame 器官.
        """
        if self._matrix is None:
            return None
        return self._matrix.env.project_path / ".ai_partners" / "frames"

    def _commit_anchor(self, note: str) -> str | None:
        """memento channel 的落锚后端 —— 封段归 ego (turn span 记账在它手里), 这里只透传.

        ``channel()`` 在 ``__aenter__`` 之后才被调用, ego 一定已经在了; 真不在就是装配顺序被
        改坏了, 直接抛 —— 静默返回 None 会被模型读成"没有新东西可封", 那是谎报.
        """
        if self._ego is None:
            raise RuntimeError("commit_anchor called before the ego session was created")
        return self._ego.commit_anchor(note)

    def _model_organ(self) -> "ModelOrgan":
        """ghost.model 的后端 —— 读目录 / 读现状 / 声明, 三件都落在 ego 身上.

        读目录走 ego 已经持有的那个 dsh session (``session/modelCatalog``), 不新开连接:
        它就是"我现在这颗脑子"的那条连接, 另开一条会答出另一份状态.
        """
        from .model_channel import ModelOrgan

        async def read_catalog():
            if self._ego is None:
                raise RuntimeError("model catalog asked before the ego session was created")
            return await self._ego.session.models()

        def read_identity():
            if self._ego is None:
                return None
            return self._ego.model_identity

        def declare(provider: str, model: str, effort: str) -> str:
            if self._ego is None:
                raise RuntimeError("model declared before the ego session was created")
            return self._ego.declare_model(provider, model, effort)

        return ModelOrgan(
            read_catalog=read_catalog,
            read_identity=read_identity,
            declare=declare,
        )

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
                commit_anchor=self._commit_anchor,
                frame_root=self._frame_root(),
                init_frame=self._startup_doc.frame if self._startup_doc is not None else None,
                model=self._model_organ(),
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
        if self._memento_view is not None:
            children.append(self._memento_view)
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

    async def think(self, thinking: Thinking) -> AsyncIterator[Logos | GhostEvent]:
        """Delegate to ego.run_thinking() to drive dsh reasoning — lifecycle/ending/CTML parsing all live in the run.

        This side only holds the async-with boundary and passes the stream through: ``GhostEvent``
        (the forwarded dsh events) reaches the observability output, ``Logos`` the mindflow broadcast.
        Errors (enter/consume/cancel) propagate naturally through async-with, governed by run.__aexit__.

        Opening the cognition epoch happens here, on the first thinking: the epoch's baseline comes
        from the shell facade, wired by the runtime *after* ghost.__aenter__ (ghost_runtime step 5),
        so opening it earlier would snapshot an empty baseline. The epoch is then delivered by the
        ego's own epoch slot on the first frame.

        A compact decided by the previous valid run is executed first, and a new epoch is opened
        right after it: compaction rewrites the model-visible head, so the cognition must restart on
        a new epoch too. The ghost memory rides that epoch's recap slot — it is no longer written
        into the surface history, which the compact would shadow.
        """
        memory: list[Message] | None = None
        if self._ego is not None:
            # 遗留优先: 它意味着「上次 compact 可能已经换了 surface、epoch 还没开」, 比本帧再压
            # 一次更急. 先取走它, 取不到才发起本帧的 compact —— 否则本帧失败的 compact 会把遗留
            # 清成 None, memory 在模型视野里彻底消失.
            memory = self._ego.take_compact_memory()
            if memory is not None:
                # 上次 compact 在 RPC 中间被取消, 本帧补上: 这是一条可观测的恢复路径, 不能静默.
                self.logger.warning(
                    "dolores: completing a compact cancelled mid-flight — opening epoch with "
                    "%d memory messages",
                    len(memory),
                )
            else:
                memory = await self._ego.compact_if_pending()
        if memory is not None:
            self._epoch_opened = True
            self._open_cognition_epoch(memory=memory)
        elif not self._epoch_opened:
            self._epoch_opened = True
            self._open_cognition_epoch(memory=None)
        if self._ego is not None:
            async with self._ego.run_thinking(thinking) as run:
                async for delta in run.logos():
                    yield delta
        else:
            yield ""

    def _open_cognition_epoch(self, memory: "list[Message] | None") -> None:
        """Open a new cognition epoch, optionally carrying the ghost memory in its recap slot.

        The first epoch passes ``memory=None`` — create_session already seeded the memory as the
        surface head there. From the first compact on, memory lives only in the epoch recap: every
        later epoch re-delivers it, so the compact shadow can never drop it.

        The memory is captured **before** ``new_epoch`` and replayed through a closure: recap funcs
        run inside ``BaseMomentsObserver.new_epoch``, which swallows their exceptions — re-reading
        the memory there could silently lose it after the compact already dropped it from surface.
        """
        moments = self.mindflow().moments
        if memory is not None:
            moments.with_epoch_recap("memory", lambda: memory)
        moments.new_epoch([])

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
        # load the ghost-home .env into the process environment (supplement only, outer env wins) —
        # the source of startup-time config items (dsh web auto-open, etc.). The ghost owns this file.
        if self._home is not None:
            self._load_env()
        # startup doc: read once, cache. channel() and startup() both read this snapshot —
        # so the frame handed to build_dolores_channel and the signal handed to the ego
        # boot come from the same file.
        self._startup_doc = await asyncio.to_thread(self._load_startup)
        # assert the configured default model into the dsh settings document — dsh reads it at
        # startup, so it must land before the spawn below (and after .env, which carries the config).
        await asyncio.to_thread(self._sync_default_model)
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
                # 布局不在这里写: 两类 ground 与 on_commit 都是 stubs 里的种子文件,
                # 由 _seed_ground 在开机时补进 ghost home; 分支目录本身保持空, 渲染靠 walk.
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
                ghost_home=self._home,
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
                # 启动补漏: 上次运行留下的空 note (关停时被取消的旁路) 在这里补派任务. 只派发不等待,
                # 与运行期旁路同一条非阻塞路径 —— 不拖慢 boot.
                self._memento_manager.backfill()
                # 线自己渲染的那一页（计划 / 开着的 task）由 branch_view 槽位生成 —— 渲染在
                # 这里, 因为 memories() 是同步读; 它现在真的会进 boot memory, 不再只在 memento:view 里.
                self._memento_view = await self._memento_manager.aview_message()
            self._ego = await self._exit_stack.enter_async_context(
                DoloresEgo(
                    launcher=self.dsh_launcher,
                    ctx=ctx,
                    config=self._load_ego_config(),
                    memories=self.memories,
                    memento_manager=self._memento_manager,
                    # logger 逐层传下去: ghost → ego → run → 每个 tool 调用. 不传 = ego 掉回兜底的
                    # get_moss_logger(), 整个 ghost 的日志就从当前 node 的语境里漂走了.
                    logger=self.logger,
                    # boot 思考档声明: 来自 startup doc, 空 = 不表态 (档位归 dsh/UI).
                    default_thinking_effort=(
                        self._startup_doc.default_thinking_effort if self._startup_doc is not None else ""
                    ),
                )
            )
            # bind the self-wake signal outlet to the MOSS session — matrix.session.add_signal routes to mindflow.
            self._ego.bind_signal_broadcast(self._matrix.session.add_signal)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    # ── startup (born) ───────────────────────────────

    async def startup(self) -> None:
        """Lifecycle hook (born) — announce the ghost's own web face, then boot-wake.

        The doc is loaded once in ``__aenter__`` so both ``channel()`` (for ``init_frame``)
        and ``startup()`` (for command + instruction) read the same snapshot. Boot is
        silent when the doc is absent or when both command and instruction are empty.
        """
        if self._matrix is None:
            return
        # The surface is an organ, not a reading of the startup doc: it is announced
        # whether or not a boot doc exists.
        await self._serve_dsh_webview()
        doc = self._startup_doc
        if doc is None or (not doc.command and not doc.instruction):
            return
        from .nucleus import new_dolores_ego_signal

        signal = new_dolores_ego_signal(
            kind="startup",
            command=doc.command,
            instruction=self._with_dsh_surface(doc.instruction),
        )
        self._matrix.session.add_signal(signal)

    def _dsh_surface_url(self) -> str | None:
        """The dsh web surface URL (its per-launch token included), or None before it is up.

        One resolver for two consumers — the boot instruction and the webview
        declaration.  Two copies of "where the surface lives" would be two
        authorities over the same fact, and they would drift.
        """
        try:
            launcher = self.dsh_launcher
        except RuntimeError:
            return None
        return launcher.web_url() or f"{launcher.config.base_url}/?token={launcher.token()}"

    async def _serve_dsh_webview(self) -> None:
        """Announce the dsh web surface on the mesh, so a screen adopts it by itself.

        The ghost's reasoning surface is an organ like any node's — not something a
        human has to open, nor something the model has to materialize by hand with
        ``open(url=...)``.  Left unannounced, "the reasoning is on stage" depends on
        someone *remembering* to do it, which is exactly the kind of thing that
        silently stops happening.
        """
        from ghoshell_moss.services.webview import WebViewDeclaration, WebViewServer

        url = self._dsh_surface_url()
        if not url:
            return
        await WebViewServer.serve(
            self._matrix,
            WebViewDeclaration(
                url=url,
                title="dsh",
                description="what the ghost is thinking, as it thinks it",
                icon="brain",
            ),
        )

    def _with_dsh_surface(self, instruction: str) -> str:
        """Append the dsh web surface URL to the boot instruction.

        The URL carries a per-launch token, so it is only known after the launcher has
        spawned dsh — it can never live in the startup doc. Without it the ghost cannot
        reach its own web surface (where the human reads its messages), which is exactly
        what happened before this line existed.
        """
        url = self._dsh_surface_url()
        if not url:
            return instruction
        line = f"dsh web surface: {url}"
        return f"{instruction}\n\n{line}" if instruction else line

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

    def _load_startup(self) -> "StartupDoc | None":
        """Parse the resolved startup doc into a :class:`StartupDoc`.

        Returns ``None`` when there is no doc or the file fails to parse — a broken
        startup should never crash the ghost boot; missing fields default to empty.
        """
        from ._startup import StartupDoc

        doc = self._resolve_startup_doc()
        if doc is None:
            return None
        try:
            data = yaml.safe_load(doc.read_text(encoding="utf-8")) or {}
            return StartupDoc.model_validate(data)
        except Exception as e:
            self.logger.warning("startup doc %s parse failed: %s", doc, e)
            return None

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
        if self._session is not None:
            self._session.output("system", log="starting dsh")
        async with launcher:
            url = launcher.web_url() or f"{launcher.config.base_url}/?token={launcher.token()}"
            if self._session is not None:
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

        Three-way data contract (see dolores-ghost-home-governance.md):

        - config (``.dolores.yml``) → read-then-rewrite: only ``version`` is sync-owned; every other
          field is the ghost/user's and survives a version bump.
        - ground (the rest of ``stubs/``) → seed-once: a file is copied only when absent, so the
          ghost's own edits (identity / purpose / behaviors / GROUND.md) are never clobbered.
        - plugin (``dsh_plugin``) → always override, handled separately and not version-gated.

        The home directory itself is materialized first: a deleted/never-created ghost home is the
        normal way to reset an instance (delete the dir, restart → re-seed), so every write below
        must find its parent in place rather than fail on a missing path.
        """
        if self._home is None:
            return None
        self._home.mkdir(parents=True, exist_ok=True)
        target = self._meta.VERSION
        current = self._load_config().version
        if current == target:
            return None
        action = "override" if current else "init"
        self._write_version(target)   # seeds .dolores.yml on init, then read-rewrite version only
        self._seed_ground()
        self._materialize_dirs()
        self._sync_dsh_home()
        return action

    def _seed_ground(self) -> None:
        """Seed-once the ground skeleton — copy every stub file except ``.dolores.yml``, only when absent."""
        stubs = self._meta.stubs_dir()
        for src in stubs.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(stubs)
            if rel.as_posix() == ".dolores.yml":
                continue
            dst = self._home / rel
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

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
        # 旧 stub 曾把 dsh 可执行路径写进 dsh.binary; 该机制已删除 (统一走 DSH_BINARY env).
        # 老 ghost home 若还留着这个键, 拔掉并 warn, 而不是让 forbid-extra 把它变成启动失败.
        dsh_section = data.get("dsh")
        if isinstance(dsh_section, dict) and "binary" in dsh_section:
            legacy = dsh_section.pop("binary")
            self.logger.warning(
                "ignoring legacy dsh.binary=%r in %s; pin the dsh binary with DSH_BINARY "
                "in the ghost's .env instead",
                legacy,
                marker,
            )
        return DoloresConfig(**data)

    def _write_version(self, version: str) -> None:
        """Read-rewrite the version marker; every other ``.dolores.yml`` field is preserved.

        Seeds the stub config on a fresh home (init): the stub carries non-default config
        (``dirs``, ``memento.force_tokens``) that the pydantic defaults must not substitute.
        """
        marker = self._home / ".dolores.yml"
        if not marker.exists():
            shutil.copy2(self._meta.stubs_dir() / ".dolores.yml", marker)
        config = self._load_config()
        config.version = version
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
        """Seed-once the dsh home from stubs — copy each file only when absent.

        Prior behavior (``copytree`` with ``dirs_exist_ok=True``) clobbered ghost-home edits on
        every boot, defeating the point of letting the ghost own its own dsh profiles. Now any
        file the ghost (or dsh itself) has written survives; only the gaps are filled.
        """
        src_root = self._meta.dsh_stubs_dir()
        dst_root = self._home / ".dsh"
        for src in src_root.rglob("*"):
            if not src.is_file():
                continue
            dst = dst_root / src.relative_to(src_root)
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        # provider block is derived at seed time (reads env), not a static stub — handle it next.
        self._sync_providers()

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

    def _sync_default_model(self) -> tuple[str, str] | None:
        """Seed-once ``agent-default-model`` into dsh settings — only when the section is absent.

        Reads the pair from ``DoloresMeta.default_model``; env fallback was removed (that was
        the wrong seam — the ghost home's settings.yaml is the right one). After the first
        seed the section belongs to the ghost home (dsh web Models page can rewrite it live);
        later Python-side changes to ``default_model`` do **not** chase an existing home.
        """
        if self._home is None:
            return None
        pair = self._meta.default_model
        if pair is None:
            return None
        provider, model = pair[0].strip(), pair[1].strip()
        if not provider or not model:
            return None
        path = self._resolve_dsh_home(self._load_config().dsh.home) / _SETTINGS_FILE
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        seeded = _seed_model_section(text, provider, model)
        if seeded is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(seeded, encoding="utf-8")
        self.logger.info("dsh default model seeded as %s/%s", provider, model)
        return provider, model

    def _sync_providers(self) -> None:
        """Seed-once meta.providers into ``profiles/web/cordis.patch.yml``.

        The stub ships with its own patch entries; we append (not replace) an ``id: llm-pi-ai``
        override that carries the declared providers. Seed marker is literal ``id: llm-pi-ai``
        in the file — once present, we never append again. base_url_env is resolved from the
        process env *at this moment* and written literally; the patch sits inside the
        gitignored ghost home, so a literal URL here never reaches the repo.
        """
        providers = self._meta.llm_providers
        if self._home is None or not providers:
            return
        target = self._home / ".dsh" / _CORDIS_PATCH
        text = target.read_text(encoding="utf-8") if target.exists() else ""
        if re.search(r"^\s*-?\s*id:\s*" + re.escape(_PI_AI_ID) + r"\s*$", text, re.MULTILINE):
            return
        block = _render_provider_block(providers)
        if not block["providers"]:
            return
        rendered = yaml.safe_dump(
            [{"id": _PI_AI_ID, **block}], allow_unicode=True, sort_keys=False,
        )
        if text and not text.endswith("\n"):
            text += "\n"
        combined = text + "\n# llm-pi-ai providers — seeded once from DoloresMeta.providers.\n" + rendered
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(combined, encoding="utf-8")
        self.logger.info("dsh providers seeded for routes: %s", list(block["providers"].keys()))

    def _resolve_dsh_home(self, home: str | Path | None) -> Path:
        if home is None:
            return self._home / ".dsh"
        p = Path(home)
        return p if p.is_absolute() else (self._home / p)
