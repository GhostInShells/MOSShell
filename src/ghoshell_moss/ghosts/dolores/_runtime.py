import asyncio
import contextlib
import json
import logging
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

import yaml

from ._meta import DoloresProviderSpec
from .constants import DOLORES_META_DIRECTORY
from typing_extensions import Self

from ghoshell_container import IoCContainer

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
    from ghoshell_moss.deepseek_harness.v0_2_0 import DshLauncher
    from ghoshell_moss.memento._fs_memento import FsMemento

    from ._ego import DoloresConfig, DoloresEgo, DoloresEgoConfig
    from ._ego_memento import EgoMementoManager
    from ._meta import DoloresMeta
    from ._startup import StartupDoc

__all__ = ["Dolores"]

from ._prompts import (
    dolores_concurrent_mind, dolores_inception, dolores_memory, dolores_output_protocol_notice,
    dolores_terminology,
    DOLORES_INSTRUCTION_END,
)

# ego 轨迹索引的根目录 (相对 ghost_home); branch 名由 memento config 决定.
_EGO_MEMENTO_DIR = ".memento/ego"

# settings.yaml 里承载默认模型的 section 名 (dsh-agent-default-model 的 settings namespace).
# 该段由 ghost 在首次 seed 时按 DoloresMeta.default_model 一次性写入, 之后归 ghost home 自治
# (dsh web Models 面可编辑). 多模态是现代模型默认, 不再做视觉强改.
_SETTINGS_MODEL_SECTION = "agent-default-model"
_SETTINGS_FILE = "settings.yaml"
_PI_AI_ID = "llm-pi-ai"
# Shape that can be written as a YAML plain scalar; everything else becomes a JSON-quoted
# string (still a legal YAML scalar).
_PLAIN_SCALAR = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# -- the ghost plugin (a dsh bundle) -------------------------------------------
# Package name / entry id / install path are three faces of one fact: they correspond to
# dsh_plugin/{package.json,cordis.patch.yml}, so they move together.
_DSH_PLUGIN_NAME = "@local/moss-dolores-ghost"
_DSH_PLUGIN_ENTRY = "dolores-ghost-plugin"
# The session verbs (anchor / chat off record / session read) ship in the ghoshell-dsh bundle.
# It is a second package on purpose: those verbs belong to dsh sessions in general, not to this
# ghost, and a ghost that wants them installs them instead of reimplementing them.
_DSH_SERVICE_NAME = "@local/ghoshell-dsh"
# The previous generation copied a bare .ts into the profile dir and registered it by
# relative path in the profile patch. Migration removes exactly those two things.
_LEGACY_PLUGIN_FILE = "moss-dolores-ghost-plugin.ts"

# Host-plane suppressions (dsh web is not the first citizen). These are the *ghost's*
# policy, so they stay in the ghost's own profile patch instead of our bundle's patch:
# the profile layer is applied after every bundle, which is what keeps them revisable.
_SUPPRESSIONS_NOTE = (
    "# Suppress host-plane entries (dsh web is not the first citizen):\n"
    "#   ui-deliverables      -- ui:deliverable-file-references (markdown file refs, conflicts with CTML-first)\n"
    "#   file-reference-local -- context:file-reference (@path input syntax; ghost input comes from moments)\n"
    "# web-surface (app:web-surface) is shadowed empty by the plugin in agent scope; harness:source stays.\n"
)
_PROFILE_SUPPRESSIONS: tuple[str, ...] = ("ui-deliverables", "file-reference-local")

# Strip types with node's own API -- no toolchain, no dependency. `mode: strip` replaces type
# syntax with spaces, so byte offsets survive and a runtime stack trace still points at the
# source line.
_STRIP_TYPES_SCRIPT = (
    "const {stripTypeScriptTypes} = require('node:module');"
    "const fs = require('node:fs');"
    "process.stdout.write(stripTypeScriptTypes(fs.readFileSync(0, 'utf8'), {mode: 'strip'}));"
)


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


class _JsExpression(str):
    """A YAML `!!js` scalar -- an expression dsh evaluates when it composes the profile."""


yaml.SafeDumper.add_representer(
    _JsExpression,
    lambda dumper, value: dumper.represent_scalar("tag:yaml.org,2002:js", str(value)),
)


_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _env_name(declared: str) -> str:
    """Turn a ``$NAME`` declaration into the bare name the patch has to carry.

    The ``$`` is the marker that says "this is the *name* of a process-environment fact, not
    the fact". The value belongs to the machine, so it is never written down; a declaration
    without the marker is refused instead of guessed at, because silence here is exactly how
    a literal endpoint ended up frozen in a file.
    """
    text = declared.strip()
    if not text.startswith("$"):
        raise ValueError(
            f"provider env reference {declared!r} must be written as '${declared}': the "
            f"declaration names the environment variable, it never holds its value"
        )
    name = text[1:]
    if not _ENV_NAME.fullmatch(name):
        raise ValueError(
            f"provider env reference {declared!r} is not a valid environment variable name"
        )
    return name


def _render_provider_block(providers: list[DoloresProviderSpec]) -> dict:
    """Project the meta's provider declarations into pi-ai's `llm-pi-ai` config shape.

    Only **names** are written down. ``apiKeyEnv`` is a credential reference, and ``baseURL``
    stays an expression (``!!js process.env.<NAME>``) instead of the URL this process happens
    to see: a value frozen at seed time goes stale the moment the environment moves, and the
    home should carry names, not this machine's endpoint. A name that is unset at compose time
    evaluates to undefined, which is pi-ai's own "use the default endpoint".
    """
    out: dict[str, dict] = {}
    for spec in providers:
        entry: dict = {"apiKeyEnv": _env_name(spec.api_key_env)}
        if spec.base_url_env:
            entry["baseURL"] = _JsExpression(f"process.env.{_env_name(spec.base_url_env)}")
        if spec.models:
            entry["models"] = list(spec.models)
        if spec.display_name:
            entry["displayName"] = spec.display_name
        out[spec.route] = entry
    return {"providers": out}


def _patch_has_item(text: str, item_id: str) -> bool:
    """Whether a patch layer already carries a top-level item with this id."""
    pattern = r"^\s*-?\s*id:\s*" + re.escape(item_id) + r"\s*$"
    return re.search(pattern, text, re.MULTILINE) is not None


def _seed_suppressions(text: str) -> str:
    """Append the host-plane suppressions this ghost owns; unchanged text means already seeded.

    A fresh profile layer comes from dsh as a commented `[]`; that placeholder is replaced
    rather than appended to, because a bare `[]` followed by list items is not a document.
    """
    missing = [item for item in _PROFILE_SUPPRESSIONS if not _patch_has_item(text, item)]
    if not missing:
        return text
    lines = [line for line in text.splitlines() if line.strip() != "[]"]
    body = "\n".join(lines).rstrip("\n")
    block = "\n\n".join(f"- id: {item}\n  disabled: true" for item in missing)
    prefix = f"{body}\n\n" if body else ""
    return f"{prefix}{_SUPPRESSIONS_NOTE}{block}\n"


def _strip_legacy_insert(text: str) -> str | None:
    """Drop the top-level patch item that mounts the previous generation's relative-path plugin.

    Returns None when there is nothing to remove. Only a list item that carries our entry id
    **and** the legacy file name is removed -- an item the ghost has since rewritten is left
    alone, because this function's business is our own trace, not the file.
    """
    lines = text.splitlines(keepends=True)
    pattern = r"\s*-?\s*id:\s*" + re.escape(_DSH_PLUGIN_ENTRY) + r"\s*"
    index = next((i for i, line in enumerate(lines) if re.fullmatch(pattern, line)), None)
    if index is None:
        return None
    start = index
    while start > 0 and not re.match(r"^-(\s|$)", lines[start]):
        start -= 1
    if not re.match(r"^-(\s|$)", lines[start]):
        return None
    end = start + 1
    while end < len(lines) and not re.match(r"^-(\s|$)", lines[end]):
        end += 1
    if _LEGACY_PLUGIN_FILE not in "".join(lines[start:end]):
        return None
    return "".join(lines[:start] + lines[end:])


def _strip_types(target: Path) -> None:
    """Turn the bundle's typed entry into the JavaScript a `node_modules` entry must be.

    Node refuses to strip TypeScript under `node_modules`, so the installed copy has to be
    plain JavaScript while the source stays typed. `mode: strip` keeps every byte offset, so
    a runtime stack trace still points at the source line.
    """
    source = target / "index.ts"
    if not source.is_file():
        return
    code = _strip_typescript(source.read_text(encoding="utf-8"))
    (target / "index.js").write_text(code, encoding="utf-8")
    source.unlink()


def _strip_typescript(text: str) -> str:
    """Strip TypeScript syntax with node's own API -- no toolchain, no dependency."""
    node = os.environ.get("MOSS_NODE_BINARY", "").strip() or shutil.which("node")
    if not node:
        raise RuntimeError(
            "node was not found: the dsh plugin ships TypeScript and its installed copy must "
            "be JavaScript. Put node on PATH, or set MOSS_NODE_BINARY."
        )
    result = subprocess.run(
        [node, "-e", _STRIP_TYPES_SCRIPT],
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "node could not strip the plugin's TypeScript:\n" + result.stderr[-2000:]
        )
    return result.stdout


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
        container: IoCContainer | None = None,
    ):
        self._moss_prompter: MossSystemPrompter = moss_prompter
        self._meta = meta
        self._home = home
        self._session = session
        self._matrix = matrix
        self._shell = shell
        self._base_instruction = base_instruction
        # the IoC container that registered this ghost's providers (None for embedded tests).
        self._container = container
        # launcher / ground are lazy — __init__ touches no httpx / matrix.processes / shell (side-effect free).
        self._dsh_launcher: DshLauncher | None = None
        self._ground_set: DefaultGroundSet | None = None
        self._root_ground: Ground | None = None
        # ghost home root frame — rendered once at boot, carried by the instruction slot.
        self._ground_text: str | None = None
        # memento: ego 轨迹索引 + 旁路服务. manager 由 ghost 持有, 同时注入 ego.
        self._memento: FsMemento | None = None
        self._memento_manager: EgoMementoManager | None = None
        # memento view cache — rendered async in __aenter__ and before every compact; read
        # synchronously by memories(). It is a snapshot of the trajectory at the moment it is
        # formed, which is exactly what boot memory is.
        self._memento_view: Message | None = None
        # <memory instanced_at> — the snapshot's absolute anchor, stored/rendered in UTC.
        self._memory_instanced_at: datetime | None = None
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
        # the plugin bundles' patches, installed in __aenter__ — the launcher applies each with
        # --patch. Order matters: the ghost's own plugin, then the session-verb service.
        self._plugin_patches: list[Path] = []

    # ── Ghost ABC ──────────────────────────────────

    @property
    def meta(self) -> GhostMeta:
        return self._meta

    def home(self) -> Path:
        """The ghost home — the instance's persistent directory.

        Raises when this ghost was constructed without one (tests / embedded ghosts); a
        consumer that needs persistence should resolve ``GhostHome`` from the IoC and treat
        the missing home as a configuration error.
        """
        if self._home is None:
            raise RuntimeError("this ghost was constructed without a home")
        return self._home

    def home_meta_dir(self) -> Path:
        """The mechanism directory under the home — prompts, configs, and other meta assets."""
        return self.home() / DOLORES_META_DIRECTORY

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
        # The ghost home's own law. It is content, not a channel description — the ground
        # channel only carries the control surface. Rendered once at boot (see __aenter__).
        if self._ground_text:
            parts.append("<!-- ground: ghost home -->\n" + self._ground_text.strip())
        parts.append(self._dolores_inception())
        parts.append(self._meta.prototype_instruction())
        parts.append(self._meta.identity_instruction())
        if self._base_instruction:
            parts.append(self._base_instruction)

        if self._moss_prompter is not None:
            parts.append("<!-- current moss project -->\n" + self._moss_prompter.project_instruction())
            parts.append("<!-- current moss mode -->\n" + self._moss_prompter.mode_instruction())

        parts.append(dolores_output_protocol_notice())
        parts.append(dolores_concurrent_mind())
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
        """Render the ghost home field once, at boot, for the instruction slot.

        The ghost home answers *who am I* — it is cognition, not a control surface, so it has
        no channel: it is rendered here and injected as text. This is the boot snapshot; the
        ghost-ground-governance rework moves the render into ``memories()`` so it rides the
        memento clock instead.
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

    def _commit_anchor(self, note: str) -> str:
        """memento channel 的落锚后端 —— 排队归 ego (turn span 记账在它手里), 这里只透传.

        ``channel()`` 在 ``__aenter__`` 之后才被调用, ego 一定已经在了; 真不在就是装配顺序被
        改坏了, 直接抛 —— 静默返回一句假回执会被模型读成"没有新东西可封", 那是谎报.
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
            from .mindset.prompter import PromptManager

            prompt_manager = (
                self._container.force_fetch(PromptManager)
                if self._container is not None else None
            )
            self._channel = build_dolores_channel(
                memento_manager=self._memento_manager,
                prompt_manager=prompt_manager,
                memento_root=self._home / _EGO_MEMENTO_DIR,
                commit_anchor=self._commit_anchor,
                frame_root=self._frame_root(),
                init_frame=self._startup_doc.frame if self._startup_doc is not None else None,
                model=self._model_organ(),
            )
        return self._channel

    def memories(self) -> list[Message]:
        """The ghost's dynamic memory — one ``<memory>`` container, not loose messages.

        Child: the memento ``<memento>`` view (the trajectory outline). The ground is not here —
        it is content for the instruction (see ``system_prompt``), and duplicating it in the
        memory snapshot would make one edit invalidate two surfaces.

        ``instanced_at`` is when *this* memory instance was formed — the anchor every time inside
        it is read against. It is not "now": the block is a snapshot, and the number stops being
        now the moment it is written.

        The cognition epoch is deliberately *not* here: it opens on the first thinking, after the
        runtime wires the shell facade (see think()), so at session-creation time there is no epoch
        to report yet. It reaches the model through the ego's own epoch slot instead.

        Read by the ego's create_session through this closure (clones share it).
        """
        children: list[Message] = []
        if self._memento_view is not None:
            children.append(self._memento_view)
        if not children:
            return []
        attributes = {}
        if self._memory_instanced_at is not None:
            attributes["instanced_at"] = self._memory_instanced_at.strftime("%Y-%m-%dT%H:%M:%SZ")
        return [Message.new(tag="memory", attributes=attributes).with_messages(*children)]

    async def refresh_memory(self, until: str | None = None) -> None:
        """Re-render the memory snapshot — the **only** writer of the cache ``memories()`` reads.

        Called at two points, and both are boundaries where the snapshot would otherwise lie:

        - boot (``__aenter__``): no ``until`` → the whole trajectory.
        - compaction: ``until`` = the cut commit's coordinate → everything *before* it. The cut
          commit's span stays verbatim in the surface, so the snapshot must stop just there;
          summarising it too would put two versions of the same span in context.

        Why a cache at all: ``memories()`` is a sync closure handed to dsh at session creation,
        while the view needs an async render (disk reads plus the branch projection). So the
        async side owns the render, the sync side only reads what it left behind.

        Skipping this at compaction is not a small staleness bug: the surface gets replaced by
        the recap, so a boot-time snapshot would drop every commit made since boot from the
        model's view — the head would jump straight to the last summary with nothing between.
        """
        if self._memento_manager is None:
            return
        self._memento_view = await self._memento_manager.aview_message(until=until)
        self._memory_instanced_at = datetime.now(timezone.utc)

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
        # env first: the dsh home below is provisioned and the plugin installed by way of the
        # dsh binary, which the ghost home's .env is where we learn about.
        if self._home is not None:
            self._load_env()
        # the dsh home and this ghost's plugin bundle — initialized on every start, no parameters.
        if self._matrix is not None:
            await asyncio.to_thread(self._prepare_dsh_home)
        if action is not None and self._session is not None:
            self._session.output(
                "system",
                f"dolores ghost home {action} (VERSION={self._meta.VERSION})",
                log=f"dolores stubs {action}",
            )
        # startup doc: read once, cache. channel() and startup() both read this snapshot —
        # so the frame handed to build_dolores_channel and the signal handed to the ego
        # boot come from the same file.
        self._startup_doc = await asyncio.to_thread(self._load_startup)
        # assert the configured default model into the dsh settings document — dsh reads it at
        # startup, so it must land before the spawn below (and after .env, which carries the config).
        await asyncio.to_thread(self._sync_default_model)
        # open and hold the ghost home ground. Stub sync runs first (GROUND.md already written);
        # the GroundSet lifecycle is managed by the exit stack.
        # 这一层正在重做 (ghost-ground-governance FEATURE): ghost home 只作为 cognition 渲染,
        # 不挂 channel; project home 的场等 step 5 以新的 channel 形态回来, 现在不构造.
        if self._home is not None:
            self._ground_set = await self._exit_stack.enter_async_context(
                DefaultGroundSet(workspace_root=self._home, root_label="ghost_home")
            )
            self._root_ground = self._ground_set.root
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
                # 布局不在这里写: memento 根与 branch 由 new_local_memento 建。目录里不再种子
                # ground 或进程 hook —— 那套文件即契约的机制已整体移除。
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
                from .mindset.prompter import PromptManager

                prompt_manager = (
                    self._container.force_fetch(PromptManager)
                    if self._container is not None else None
                )
                self._memento_manager = await self._exit_stack.enter_async_context(
                    EgoMementoManager(
                        connection=self.dsh_launcher,
                        memento=self._memento,
                        config=self._load_config().memento,
                        prompt_manager=prompt_manager,
                        logger=self.logger,
                    )
                )
                # memory 的唯一写入点(开机这一次 + 每次 compact 前一次). memories() 是同步闭包,
                # 只能读缓存; 渲染在这里和 compact 前发生.
                await self.refresh_memory()
            self._ego = await self._exit_stack.enter_async_context(
                DoloresEgo(
                    launcher=self.dsh_launcher,
                    ctx=ctx,
                    config=self._load_ego_config(),
                    memories=self.memories,
                    memory_refresh=self.refresh_memory,
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
        from ghoshell_moss.deepseek_harness.v0_2_0 import DshLauncher

        dsh = self._load_config().dsh
        home = self._resolve_dsh_home(dsh.home)
        # the bundles' patches are the whole registration -- dsh composes them after the profile
        # layer, so the profile's own patch is never edited to mount a plugin.
        launcher_config = dsh.model_copy(
            update={"home": home, "plugin_patch": None, "plugin_patches": list(self._plugin_patches)}
        )
        # A logger travels with the launcher: without one the launcher logs on
        # ``logging.getLogger("ghoshell_dsh...")``, which has no handler here — its WARNING/ERROR
        # fall through Python's lastResort to the terminal and never reach moss.log, so a mux
        # fault arrives as "the ghost went deaf" with nothing written down. That was the second
        # half of the 2026-11-12 1008: the cause was visible, but only on a terminal nobody read.
        return DshLauncher(launcher_config, subprocesses=self._matrix.processes, logger=self.logger)

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

        - config (``.dolores.yml``) -> read-then-rewrite: only ``version`` is sync-owned; every
          other field is the ghost/user's and survives a version bump.
        - ground (the rest of ``stubs/``) -> seed-once: a file is copied only when absent, so the
          ghost's own edits (identity / purpose / behaviors / GROUND.md) are never clobbered.
        - dsh home -> *not* ours to copy any more: dsh provisions its own layout, and the plugin
          is a bundle installed by ``_prepare_dsh_home``.

        The home directory itself is materialized first: a deleted/never-created ghost home is the
        normal way to reset an instance (delete the dir, restart -> re-seed), so every write below
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

    # -- dsh home: provision + plugin bundle -----------------------------------

    def _prepare_dsh_home(self) -> None:
        """Make the dsh home whole at every start -- no parameters, no manual step.

        Four idempotent steps, in order:

        1. **provision**: when the profile has never existed, dsh creates it
           (``--dump-config`` composes without booting). The layout belongs to dsh, so a dsh
           upgrade is also a layout upgrade; we no longer copy a home skeleton of our own.
        2. **install the plugin bundle**: a content-fingerprinted copy under
           ``profiles/node_modules``, copied only when the shipped bytes changed. That is the
           whole upgrade story -- replace the package, restart the ghost.
        3. **retire the previous generation's trace**: a bare ``.ts`` beside the profile and a
           relative-path entry in the profile's own patch layer.
        4. **seed what is genuinely ours** into that profile layer: the host-plane
           suppressions (when absent), then the provider block (when absent).
        """
        if self._home is None:
            return
        from ghoshell_moss.deepseek_harness.v0_2_0 import (
            bundle_dir, installed_patch, refresh_bundle, resolve_dsh_binary,
        )

        config = self._load_config().dsh
        home = self._resolve_dsh_home(config.home)
        profile = config.profile or "web"
        binary = resolve_dsh_binary()
        refresh_bundle(
            source=self._meta.dsh_plugin_bundle(),
            name=_DSH_PLUGIN_NAME,
            home=home,
            profile=profile,
            binary=binary,
            init=True,
            transform=_strip_types,
            # verify by package name: the loader entry id is also carried by the legacy layer
            # that migration has not removed yet, so it would not prove anything on the first
            # migrated boot.
            verify_marker=_DSH_PLUGIN_NAME,
        )
        self._plugin_patches = [installed_patch(home, _DSH_PLUGIN_NAME)]
        # The session verbs (anchor / chat off record) live in the ghoshell-dsh bundle, not in the
        # ghost's own plugin. Installing it here is what makes a ghost home able to talk to a past
        # context without inventing a second implementation of the same verb.
        #
        # A failure here degrades, it does not stop the ghost: without the bundle the ego still
        # boots, and only the verbs that ride on it answer with an error. That is the difference
        # between a missing capability and a dead body.
        try:
            refresh_bundle(
                source=bundle_dir(),
                name=_DSH_SERVICE_NAME,
                home=home,
                profile=profile,
                binary=binary,
                init=True,
                verify_marker=_DSH_SERVICE_NAME,
            )
        except Exception:  # noqa: BLE001 — a missing verb must not take the ego down
            self.logger.exception("dolores: cannot install the %s bundle", _DSH_SERVICE_NAME)
        else:
            self._plugin_patches.append(installed_patch(home, _DSH_SERVICE_NAME))
        self._retire_legacy_plugin(home, profile)
        self._seed_profile_suppressions(home, profile)
        self._sync_providers(home, profile)

    def _retire_legacy_plugin(self, home: Path, profile: str) -> None:
        """Remove the plugin registration an older build planted in the profile's own layer.

        Left in place it would either load the plugin twice or point at a file that no longer
        exists. Removal is by id, and only for a block that still names the legacy file: the
        rest of that patch layer belongs to the ghost, so it is never rewritten wholesale.
        """
        profile_dir = home / "profiles" / profile
        legacy = profile_dir / _LEGACY_PLUGIN_FILE
        if legacy.is_file():
            legacy.unlink()
            self.logger.info("dolores: retired the legacy plugin copy at %s", legacy)
        patch = profile_dir / "cordis.patch.yml"
        if not patch.is_file():
            return
        text = patch.read_text(encoding="utf-8")
        stripped = _strip_legacy_insert(text)
        if stripped is None:
            return
        patch.write_text(stripped, encoding="utf-8")
        self.logger.info("dolores: retired the legacy plugin entry from %s", patch)

    def _seed_profile_suppressions(self, home: Path, profile: str) -> None:
        """Seed the host-plane suppressions once; an already seeded layer is left alone."""
        patch = home / "profiles" / profile / "cordis.patch.yml"
        text = patch.read_text(encoding="utf-8") if patch.is_file() else ""
        seeded = _seed_suppressions(text)
        if seeded == text:
            return
        patch.parent.mkdir(parents=True, exist_ok=True)
        patch.write_text(seeded, encoding="utf-8")

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

    def _sync_providers(self, home: Path, profile: str) -> None:
        """Seed-once meta.providers into the profile's own patch layer.

        We append (never replace) an ``id: llm-pi-ai`` entry carrying the declared providers;
        the seed marker is that literal id, so once it is there we never append again. The
        provider map rides **under `config:`** -- the loader hands a plugin ``entry.config`` and
        ignores every other key on the entry, so a flat ``providers:`` would seed a file that
        reads correctly and does nothing.
        """
        providers = self._meta.llm_providers
        if not providers:
            return
        target = home / "profiles" / profile / "cordis.patch.yml"
        text = target.read_text(encoding="utf-8") if target.exists() else ""
        if re.search(r"^\s*-?\s*id:\s*" + re.escape(_PI_AI_ID) + r"\s*$", text, re.MULTILINE):
            return
        block = _render_provider_block(providers)
        if not block["providers"]:
            return
        rendered = yaml.safe_dump(
            [{"id": _PI_AI_ID, "config": block}], allow_unicode=True, sort_keys=False,
        )
        if text and not text.endswith("\n"):
            text += "\n"
        combined = text + "\n# llm-pi-ai providers -- seeded once from DoloresMeta.providers.\n" + rendered
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(combined, encoding="utf-8")
        self.logger.info("dsh providers seeded for routes: %s", list(block["providers"].keys()))

    def _resolve_dsh_home(self, home: str | Path | None) -> Path:
        if home is None:
            return self._home / ".dsh"
        p = Path(home)
        return p if p.is_absolute() else (self._home / p)
