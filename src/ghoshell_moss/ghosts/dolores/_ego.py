"""DoloresEgo — the ego / continuity layer: thinking transaction + external-activity self-wake.

The Python half of the ego surface, paired with the dsh-side plugin (which owns the kernel:
ego/create, thinking enter/exit, tool-result, perStep lock). This module owns the ego session
state, moment→wire serialization, and the self-wake watcher.

Two lifecycle lines:

- long-lived (ghost lifetime): a background watcher on turn/start + user/message emits self-wake
  signals; suppressed while a thinking transaction is running.
- short-lived (per thinking): run_thinking() returns a DoloresRun — an async-with transaction
  boundary plus an event stream; its lifecycle (enter/exit/observe/perStep) lives there.

Moment serialization fills three enter-injected slots, assembled Python-side (the plugin is a dumb
transport that only receives content blocks):

- context (echoes/dynamic/executing) → <moment>, injected as background;
- inputs (percepts + optional <hint>) → <inputs>, steered to drive a turn;
- epoch (on epoch change) → <epoch index=N> recap + baseline, injected as background.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.blueprint.moment import Moment
from ghoshell_moss.core.blueprint.mindflow import Signal, Thinking
from ghoshell_moss.deepseek_harness.launcher import DshLauncher, DshLauncherConfig
from ghoshell_moss.deepseek_harness.session import DshSession
from ghoshell_moss.deepseek_harness.types.session_events import (
    AssistantMessageEvent,
    RequestHeader,
    SessionEvent,
    TurnEnd,
)
from ghoshell_moss.message import Content, Message
from ghoshell_moss.memento.abcd import CommitRef

from ._ego_memento import CommitDecision, EgoMementoConfig, EgoMementoManager
from ._prompts import dolores_model_notice
from .nucleus import new_dolores_ego_signal

if TYPE_CHECKING:
    from ._run import DoloresRun
    from ghoshell_moss.core.blueprint.shell_trajectory import MShellContextFacade

__all__ = ["DoloresConfig", "DoloresEgo", "DoloresEgoConfig", "DoloresEgoContext"]

# Route constants — cross-language contract with the dsh plugin.
_DOLORES_EGO_CREATE = "/moss-api/ghost/dolores/ego/create"
_DOLORES_THINKING_ENTER = "/moss-api/ghost/dolores/thinking/enter"
_DOLORES_THINKING_EXIT = "/moss-api/ghost/dolores/thinking/exit"
_DOLORES_TOOL_RESULT = "/moss-api/ghost/dolores/tool-result"
# compact: 原地把 surface 前缀替换成 memory (ground + memento view), cut 到 cut_turn 边界.
_DOLORES_COMPACT = "/moss-api/ghost/dolores/compact"

# thinking/exit confirmation timeout (fail-safe): degrade instead of hanging the exit if the plugin stalls.
_EXIT_RPC_TIMEOUT = 5.0


class DoloresEgoConfig(BaseModel):
    """Ego session config, loaded from the ``ego:`` section of .dolores.yml.

    Field defaults are the fallback — a missing YAML key uses the default rather than a manual .get().
    """

    session_title: str = Field(
        default="Ψ · {timestamp}",
        description="session title template ({timestamp} placeholder), the human-readable session name.",
    )
    permission: str = Field(
        default="workspace-write",
        description="sandbox mode: read-only | workspace-write | danger-full-access.",
    )
    inception_template: str = Field(
        default="",
        description=(
            "instruction template path (relative to ghost home). Replaces only the persona/etiquette "
            "layer; the protocol sections are not replaceable. Empty = built-in default. "
            "Slots: {ghost_home} / {project_home} / {mode_home}."
        ),
    )


class DoloresConfig(BaseModel):
    """Top-level .dolores.yml config. Field defaults are the fallback."""

    model_config = ConfigDict(extra="ignore")

    version: str = Field(
        default="",
        description="stub-sync version marker (matches DoloresMeta.VERSION).",
    )
    dirs: list[str] = Field(
        default_factory=list,
        description="subdirectories to materialize in ghost home.",
    )
    dsh: DshLauncherConfig = Field(
        default_factory=DshLauncherConfig,
        description="dsh launcher config (binary/profile/port/...).",
    )
    ego: DoloresEgoConfig = Field(
        default_factory=DoloresEgoConfig,
        description="ego session config.",
    )
    memento: EgoMementoConfig = Field(
        default_factory=EgoMementoConfig,
        description="memento 旁路服务配置 (branch / view 边界 / K·T 阈值).",
    )


@dataclasses.dataclass(frozen=True, slots=True)
class DoloresEgoContext:
    """Static context captured before the ego enters its lifecycle — assembled by the ghost, injected to avoid a back-ref.

    Values are read once at ego construction; nothing needs to reach back into the ghost afterward.
    All references are injected via typed objects / variables / closures — no back-ref is held.

    - project_home: working dir of the ego session.
    - project_name: workspace title (`{name} @ {project}`).
    - name: ghost name, used for title/identity.
    - mode: mode name, used for the session title.
    - instruction: assembled system prompt.
    - facade: shell context surface (used to refresh meta on interleaved_ctml).
    - ghost_home: ghost home dir — 旁路 (note/chat) session 归组的 home workspace; None = 不建 home workspace.
    """

    project_home: Path
    project_name: str
    name: str
    mode: str
    instruction: str
    facade: "MShellContextFacade"
    ghost_home: Path | None = None


class DoloresEgo:
    """The ego / continuity layer. See the module docstring."""

    def __init__(
            self,
            *,
            launcher: "DshLauncher",
            ctx: DoloresEgoContext,
            config: DoloresEgoConfig | None = None,
            logger: LoggerItf | None = None,
            memories: Callable[[], list[Message]] | None = None,
            memento_manager: EgoMementoManager | None = None,
            default_thinking_effort: str = "",
    ) -> None:
        """Construct before the ghost enters its lifecycle; side-effect free (no httpx / session / matrix.processes).

        All dependencies are injected via typed objects (launcher/config/logger), variables (ctx), or
        closures (memories / bind_signal_broadcast) — no ghost back-ref, no private-member access.

        :param launcher: dsh reasoning-core launcher, used for ego create and thinking enter/exit RPCs.
        :param ctx: one-shot session context (home/name/instruction/project_name).
        :param config: ego session config; None uses all defaults.
        :param logger: logger; None falls back to the MOSS logger.
        :param memories: closure returning the ghost's dynamic memory (existential layer); called on
            create_session for the freshest value. Clones share the same closure. None = no memory.
        :param memento_manager: ghost-held memento 旁路服务; 锚点写入与阈值判定都过它. None = 无 memento.
        :param default_thinking_effort: boot-time thinking depth declaration (from startup doc). Empty =
            declare nothing — the depth stays whatever dsh/UI holds.
        """
        self._launcher = launcher
        self._ctx = ctx
        self._facade = ctx.facade
        self._config = config or DoloresEgoConfig()
        self._memories = memories
        self._memento_manager = memento_manager
        self._session: "DshSession | None" = None
        self._ego_session_id: str | None = None
        # anti-bypass token: returned by ego/create, carried by thinking enter/exit, verified by the plugin to reject non-ego calls.
        self._thinking_token: str | None = None
        # thinking 配对序 (单调, 每 session): enter/exit 带同一个, 插件只关自己那一次的 gate.
        self._thinking_seq: int = 0
        self._exit_stack = contextlib.AsyncExitStack()
        # logger: prefer the injected one (MOSS runtime logger), else fall back.
        self._logger = logger or get_moss_logger()
        # self-wake gate: whether a thinking transaction is running; the turn/start watcher reads this.
        self._thinking_event = asyncio.Event()
        # self-wake signal outlet — injected by host/mindflow after bus wiring; this side never touches the nucleus directly.
        self._signal_broadcast: "Callable[[Signal], None] | None" = None
        # epoch tracking: remembers the last injected epoch id, compared on enter to decide whether to carry an <epoch> container.
        self._moment_epoch: str | None = None
        # 待采用的思考档声明: startup 的 boot 声明 (或 moss_reasoning 的现场声明) 放这里,
        # 由下一个**真正带上它**的思考帧下发, 采用即置回 None —— 没被采用的声明不会凭空蒸发.
        # 声明只被消费一次, 但档位本身不会一轮即丢: plugin 把它写进 request/header 的持久配置,
        # 后续帧仍从那里读到, 直到 ghost 或界面再改一次.
        self.default_thinking_effort: str | None = default_thinking_effort or None
        # 预期模型身份 (provider, model, reasoning_effort) — request/header 观测值与之不符时排一条
        # notice. None = 尚未观测过 (首次观测也算变化: 开局就得知道自己在哪一档).
        self._model_identity: tuple[str, str, str] | None = None
        # commit 运行时状态 (ego 持有; manager 不托管): 最后一个已完成 turn + 窗口基准 + 每窗口提醒位.
        self._last_turn: int = 0
        self._window_size: int = 0
        self._window_base: int = 0
        self._warned: bool = False
        # 待执行的 compact (上一有效 run 决定, 下一 run 启动执行) — 只在 run 边界写入/清空.
        self._pending_compact: bool = False
        # 待注入的 notice (warn / committed); thinking-enter 时排空, 与 moment 同级注入.
        self._notices: list[Message] = []
        # 连续缺 usage 计数: 首帧 (warm-up) 的 usage 是收尾才补, live 流里常缺一条, 属预期瞬态;
        # 连续两条都缺才说明窗口增量真喂不进来. 出现 usage 即清零.
        self._usage_missing_count: int = 0
        self._self_wake_unwired_warned: bool = False

    # ── long-lived: lifecycle ────────────────────────────────────────

    async def __aenter__(self) -> Self:
        """Enter the ghost lifecycle and create the ego session."""
        await self._exit_stack.__aenter__()
        await self.create_session()
        return self

    async def create_session(self) -> str:
        """Create the ego session (reusable — clones share the same memories closure).

        Injects instruction + memory (ghost dynamic memory, 1:1 into user messages) at session
        creation, establishing an initial surface below the instruction and above the conversation.
        Returns the ego session id.

        启动**不再回放源 session 的原文 tail**：seed 只有 memory (ground + memento view)。这样 dsh
        历史消息协议再变，启动都不碰（防「升级后 ghost 永远起不来」）。跨重启的连续性由 memento
        视图 + read/chat 承担。
        """
        payload = {
            "project_home": str(self._ctx.project_home),
            "project_name": self._ctx.project_name,
            "title": self._config.session_title.format(
                mode=self._ctx.mode,
                timestamp=datetime.now().strftime("%y-%m-%d %H:%M:%S"),
            ),
            "instruction": self._ctx.instruction,
            "messages": self._assemble_initial_messages(),
            "permission": self._config.permission,
        }
        if self._ctx.ghost_home is not None:
            payload["ghost_home"] = str(self._ctx.ghost_home)
            payload["home_title"] = f"{self._ctx.name} @ home"
        result = await self._launcher.call(_DOLORES_EGO_CREATE, payload)
        self._ego_session_id = result["sessionId"]
        self._thinking_token = result.get("thinkingToken")
        if self._thinking_token is None:
            # 每个 enter/exit 都带这个 token, 插件用它拒绝非 ego 调用. 拿不到它时这里一切正常,
            # 坏在下一步 —— 于是原因看不见, 只见症状. 现在就把它说出来.
            self._logger.warning(
                "ego/create returned no thinkingToken (session %s) — thinking enter/exit will "
                "carry no anti-bypass token and the plugin may reject them",
                self._ego_session_id,
            )
        self._session = self._launcher.create_session(self._ego_session_id)
        await self._exit_stack.enter_async_context(self._session)
        # long-lived: subscribe to turn/start + user/message for silent self-wake.
        # user/message covers direct UI input — it produces only user/message (not turn/start), so
        # self-wake is still needed to observe it.
        self._session.on_session_event("turn/start", self._on_session_activity)
        self._session.on_session_event("user/message", self._on_session_activity)
        # commit: 在 completed turn 边界推进 last_turn, 并按阈值决定是否强制提交锚点.
        self._session.on_session_event("turn/end", self._on_turn_end)
        # 窗口大小: assistant/message 带 usage, 记下最近一次调用的 prompt 大小 (_maybe_commit 求增量).
        self._session.on_session_event("assistant/message", self._on_assistant_message)
        # 模型身份: request/header 是变更时记录的下一次请求配置 (provider/model/reasoningEffort).
        self._session.on_session_event("request/header", self._on_request_header)
        return self._ego_session_id

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit: commit on a normal exit (封尾), then close the ego session.

        正常退出必 commit; 异常退出不管 —— 异常态的 turn 区间不可信.
        """
        if exc_type is None:
            try:
                self._commit()
            except Exception:
                self._logger.exception("ego exit commit failed — degraded")
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def session(self) -> "DshSession":
        """The held dsh session facade — raises a clear error before startup."""
        if self._session is None:
            raise RuntimeError("ego session not started. Call __aenter__ first.")
        return self._session

    @property
    def model_route(self) -> "tuple[str, str] | None":
        """(provider, model) this session actually requested with — the last ``request/header``.

        Deliberately **not** ``session.model_selection()``: that reads the model *catalog
        default*, and the catalog default is not session state — ``session/selectModel``
        rewrites it globally in the background. Measured 2026-10-15: one ``selectModel`` on a
        throwaway session moved the default every session reads, while the live session's own
        projection stayed put. So the catalog default must never be used to fill in "the
        session's current model": it would silently rewrite a route the human changed in the UI.

        None = no request observed yet (a session that has never run).
        """
        if self._model_identity is None:
            return None
        return (self._model_identity[0], self._model_identity[1])

    # ── short-lived: run_thinking (transaction) ──────────────────────

    def run_thinking(self, thinking: "Thinking") -> "DoloresRun":
        """Return the run object for a thinking transaction — consumed by ``async with``.

        The async-with boundary is the explicit lifecycle (see DoloresRun): enter binds listeners and
        starts the enter task; exit cancels, unbinds, re-sends exit, and aborts. The consumer pulls
        events and dispatches logos/turn/end; the articulator is managed by the caller.

        :param thinking: the mindflow Thinking — moment/effort/articulator/abort all come from it.
        """
        from ._run import DoloresRun
        return DoloresRun(
            ego=self,
            thinking=thinking,
            thinking_event=self._thinking_event,
            facade=self._facade,
            # 逐层传下去的 logger: 这个 run 属于当前 node, 日志也该落在当前 node 的 logger 上.
            logger=self._logger,
        )

    # ── context assembly ─────────────────────────────────────────────

    def _assemble_initial_messages(self) -> list[dict]:
        """Initial messages: ghost dynamic memory (memories closure) → plugin payload.

        Each item is ``{"text": ...}``, injected by the plugin as a user message. Memory maps 1:1 to
        dsh user messages (no folding).
        """
        if self._memories is None:
            return []
        return [
            {"text": msg.to_content_string()}
            for msg in self._memories()
            if not msg.is_empty()
        ]

    # ── background watcher (long-lived) ──────────────────────────────

    @property
    def is_thinking(self) -> bool:
        """self-wake gate — whether a thinking transaction is running (the run sets/clears the event)."""
        return self._thinking_event.is_set()

    def bind_signal_broadcast(self, broadcast: "Callable[[Signal], None]") -> None:
        """Inject the self-wake signal outlet (host/mindflow bus broadcast).

        Self-wake signals are produced here but delivered to the mindflow bus (routed by signal name).
        This gives the host a seam — the ego never holds the nucleus instance directly.
        """
        self._signal_broadcast = broadcast

    async def _on_session_activity(self, event: "SessionEvent") -> None:
        """External session activity callback (turn/start + user/message) — silent self-wake heartbeat.

        Gate: if a thinking transaction is running, this ghost is already driving, so don't wake.
        Otherwise external activity on the dsh side means the ghost should wake — emit a self-wake
        signal. Discardable: the nucleus builds a BACKGROUND impulse; if mindflow is busy the
        challenge fails and is dropped, so it only wakes when idle.
        """
        if self.is_thinking:
            return
        self._emit_self_wake()

    def _emit_self_wake(self) -> None:
        """Emit a self-wake signal (silent when no broadcast is wired, for tests/pre-wiring).

        Unwired is legitimate before host/mindflow wiring — but if it is *still* unwired at runtime,
        self-wake is dead and nothing anywhere says so. One warning, once, on the first dropped wake.
        """
        signal = new_dolores_ego_signal()
        if self._signal_broadcast is not None:
            self._signal_broadcast(signal)
        elif not self._self_wake_unwired_warned:
            self._self_wake_unwired_warned = True
            self._logger.warning(
                "self-wake dropped: no signal broadcast is wired — the ghost will not wake on "
                "external session activity"
            )

    # ── commit (锚点; 慢的 message 生产归 manager 的 sidecar) ──────────

    async def _on_assistant_message(self, event: SessionEvent) -> None:
        """assistant/message 回调 — 用 usage 刷新窗口大小 (覆盖, 非累加: 语义是最近一次调用的 prompt 大小)."""
        if self._memento_manager is None:
            return
        message = AssistantMessageEvent.from_session_event(event)
        if message is None:
            self._logger.warning("assistant/message did not parse — window size unchanged")
            return
        if message.usage is None:
            # 首帧 (warm-up) 的 usage 是收尾才补的, live 流里常缺一条, 属预期瞬态, 不当故障.
            # 连续两条都缺 usage 才说明窗口增量真的喂不进来 (K/T 阈值会静默失效), 此时才告警.
            self._usage_missing_count += 1
            if self._usage_missing_count == 2:
                self._logger.warning(
                    "assistant/message carried no usage twice in a row — the window size stays 0 "
                    "and the K/T commit thresholds can never fire; automatic anchoring is silently dead"
                )
            return
        self._window_size = self._memento_manager.window_size(message.usage)
        self._usage_missing_count = 0

    async def _on_request_header(self, event: SessionEvent) -> None:
        """request/header 回调 — 感知模型身份与思考档, 变化时排一条 notice (下一帧 enter 带出).

        首次观测也发: 思考档决定交互礼仪 (off 直答 / max 憋到最后), 模型开局就得知道自己在哪一档,
        否则会按错误的档位选礼仪. 上下文压缩后模型自己声明过的 effort 也不复记忆, 这条 notice 补上.
        """
        header = RequestHeader.from_session_event(event)
        if header is None:
            return
        config = header.header.config
        observed = (config.provider, config.model, config.reasoningEffort or "")
        previous = self._model_identity
        if observed == previous:
            return
        self._model_identity = observed
        self._notices.append(Message.new(tag="model_notice").with_content(
            dolores_model_notice(
                provider=observed[0],
                model=observed[1],
                effort=observed[2],
                previous_effort=None if previous is None else previous[2],
            )
        ))

    async def _on_turn_end(self, event: SessionEvent) -> None:
        """turn/end 回调 — 推进 last_turn, 再按阈值决定是否强制提交."""
        turn_end = TurnEnd.from_session_event(event)
        if turn_end is None:
            return
        self._last_turn = max(self._last_turn, turn_end.turn)
        self._maybe_commit()

    def _maybe_commit(self) -> None:
        """阈值判定: 到 T 强制 commit (封一段), 到 K 提醒一次 (notice 排队列, 下次 enter 注入)."""
        manager = self._memento_manager
        if manager is None:
            return
        growth = self._window_size - self._window_base
        decision = manager.evaluate(growth, self._warned)
        if decision is CommitDecision.FORCE:
            self._commit()
        elif decision is CommitDecision.WARN:
            self._warned = True
            self._notices.append(manager.warn_notice(growth))

    def _commit(self, message: str = "") -> CommitRef | None:
        """落一个锚点并排旁路 note: 区间 = (上个 commit 的 end_turn, 最后一个已完成 turn]. 无 memento 时 no-op.

        message 默认空 (authoritative message 归 sidecar); 提交后重置滑动窗口, notice 排队列.
        区间为空 (没有新追认的 turn, 如封尾时上一锚点就落在同一个 turn) → 不落锚点, 返回 None.
        """
        manager = self._memento_manager
        if manager is None:
            self._logger.warning("commit requested but no memento manager is wired — nothing anchored")
            return None
        if self._ego_session_id is None:
            self._logger.warning("commit requested before the ego session exists — nothing anchored")
            return None
        anchor = manager.commit(
            session_id=self._ego_session_id,
            start_turn=self._session_start_turn(),
            end_turn=self._last_turn,
            message=message,
        )
        if anchor is None:
            self._logger.info(
                "empty commit span (start=%s end=%s) — no anchor written",
                self._session_start_turn(), self._last_turn,
            )
            return None
        self._window_base = self._window_size
        self._warned = False
        self._notices.append(manager.committed_notice(anchor))
        manager.schedule_note(anchor.id)  # 慢腿: 空 note 归旁路 sidecar (已有 note 时它自己会让开)
        return anchor

    def commit_anchor(self, note: str = "") -> str | None:
        """主动落锚 (memento channel 的写面后端) —— 封一段, 返回坐标; 空区间返回 None.

        ``note`` 非空 = 这条记忆由 ghost 自己写, 旁路不再代笔 (判据在 ``schedule_note``);
        留空 = 只封段, 摘要照旧交给旁路. 区间上界是**最后一个已完成的 turn** —— 正在走的
        这一轮不在里面, 落锚落不住脚下这一步.
        """
        anchor = self._commit(message=note)
        if anchor is None or self._memento_manager is None:
            return None
        return self._memento_manager.coord(anchor)

    # ── compact (原地 surface replace; 上一有效 run 决定, 下一 run 启动执行) ──

    def on_run_exit(self, valid: bool) -> None:
        """run 边界回调 —— 只在**有效** run (无异常 / 无中断 abort) 之后确认是否要 compact.

        valid=False 时清空待执行位（异常 run 的上下文状态不可信，不能据此压）。valid=True 时按
        **绝对**窗口阈值 ``compact_tokens`` 判定：`window_size`（最近一次 assistant/message 的 usage）
        达标且存在 SCS（至少两个 commit）→ 置位，交给下一个 run 启动时执行。
        """
        if not valid:
            self._pending_compact = False
            return
        manager = self._memento_manager
        if manager is None:
            return
        if self._window_size >= manager.config.compact_tokens and manager.compact_cut_turn() is not None:
            self._pending_compact = True

    async def compact_if_pending(self) -> None:
        """run 启动时调用：有 pending 就原地 compact（surface 替换成 memory），无则 no-op.

        执行放在请求之前（下一 run 的 enter 尚未注入帧），surface 处于 idle，replace 安全。失败降级
        为告警 —— 主动 compact 是防溢出，不该反过来把一个 run 的启动拖死。
        """
        if not self._pending_compact:
            return
        self._pending_compact = False
        manager = self._memento_manager
        if manager is None:
            return
        cut_turn = manager.compact_cut_turn()
        if cut_turn is None:
            return
        memory = self._compact_memory_payload()
        if not memory:
            return
        try:
            await self._launcher.call(_DOLORES_COMPACT, {"memory": memory, "cut_turn": cut_turn})
            self._logger.info("dolores compact at turn %s (window=%s)", cut_turn, self._window_size)
        except Exception:
            self._logger.exception("dolores compact failed — degraded")

    def _compact_memory_payload(self) -> list[dict]:
        """当前 memory + 一条 compact 说明 → 替换内容（``{"text": ...}`` 列表，与 create 的 messages 同形）."""
        if self._memories is None:
            return []
        memory = [
            {"text": msg.to_content_string()}
            for msg in self._memories()
            if not msg.is_empty()
        ]
        if not memory:
            return []
        notice = (
            "Your context was compacted to stay within the model window. The memory view below is the "
            "compressed history (ground + memento commits); the recent turns after it are the live tail."
        )
        return [{"text": notice}, *memory]

    def _session_start_turn(self) -> int:
        """本 session 的区间下界 (**开**): 上个 commit 的 end_turn (同 session); 跨 session 从 0 重编号.

        下界开 → turn ``start_turn`` 归上一个锚点, 本锚点从它之后起; 于是相邻锚点严丝合缝
        (`(0,1] (1,2]`), 下界逐字抄旧的 ``end_turn``, 不需要 +1.
        """
        manager = self._memento_manager
        previous = manager.latest_ref() if manager is not None else None
        if previous is not None and previous.session_id == self._ego_session_id:
            return previous.end_turn
        return 0

    # ── RPC (narrow bridge to the plugin) ────────────────────────────

    async def rpc_tool_result(
            self,
            call_id: str,
            result: dict | list | str | None,
            moment: dict | None = None,
            cancel: bool = False,
            turn: int | None = None,
    ) -> None:
        """Unlock a pending tool: {callId, result, moment, cancel, turn}.

        result = the tool's return value for the model (a "{epoch}-{moment}" short id for
        fetch_next_moment). moment = the moment payload to inject — the *same* ``{context, inputs,
        moment_id}`` protocol as ``thinking/enter`` uses (see ``moment_payload``), never a
        context-only slice: the moment already drained its percepts, so dropping ``inputs`` here
        would destroy them. The plugin injects the moment then resolves the result. callId is
        passed through for routing.
        cancel = cut the turn as soon as the result lands (the turn ends without a final answer; the
        plugin binds it to the turn the tool was called in).
        turn = the turn the tool was called in. It lets the plugin bind a result that arrives before
        dsh dispatched the tool's execute to the right turn, and clean such early results per turn.
        """
        await self._launcher.call(_DOLORES_TOOL_RESULT, {
            "callId": call_id,
            "result": result,
            "moment": moment,
            "cancel": cancel,
            "turn": turn,
        })

    def moment_payload(self, moment: Moment, moment_id: str) -> dict:
        """moment → the one wire payload **every** delivery path carries: context + inputs.

        Both delivery paths — ``thinking/enter`` and ``tool-result`` — hand over the same moment
        data protocol. The earlier split (enter = context + inputs, tool-result = context only) was
        a silent loss channel, not a simplification: ``observe()`` *drains* whatever percepts were
        injected since the last observation into the moment it produces, and a path that then
        structurally drops ``inputs`` destroys that text outright. A frame carrying a buffered
        impulse (verdict ``buffered`` rides the next observation as percepts) could therefore be
        consumed by a mid-thought ``moss_interpret`` observation and never reach the model at all.

        The consume/deliver decoupling is the defect; the invariant is "no delivery path drops
        inputs". Slot mapping (context vs inputs) lives in ``_moment_payload``.
        """
        return self._moment_payload(moment, moment_id)

    def needs_observe(self, thinking: "Thinking") -> bool:
        """Whether this frame is a self-driven observe continuation (the previous round's echo required another look).

        The shell trajectory marks ``need_observe`` when an interpreter settles (a command finished, so the ghost
        should look back at what it did); the mindflow loop turns that into the next thinking frame. Such a frame
        often carries no percepts — it is the ghost's own continuation, not a new input — so the plugin needs the
        flag to know it must open a turn instead of buffering the frame as pure background.
        """
        previous = thinking.moment.previous
        return bool(previous is not None and previous.need_observe)

    def next_thinking_seq(self) -> int:
        """铸下一个 thinking 配对序 — 单调, 每 session 一次递增.

        enter 是可取消的 task, 它的 HTTP 请求一旦在飞, 取消就收不回来 —— 插件侧的到达顺序因此
        不保证与发起顺序一致. enter 与 exit 带同一个序, 插件只关自己那一次的 gate, 迟到的
        (序号更小的) enter 一律无视. 于是 exit(N) 不可能关掉 enter(N+1) 开的那道门.
        """
        self._thinking_seq += 1
        return self._thinking_seq

    async def enter_thinking(self, thinking: "Thinking", thinking_seq: int) -> None:
        """Inject moment (context/inputs) + epoch + effort + thinkingToken + thinkingSeq to start a thinking turn.

        ``thinkingSeq`` pairs this enter with its exit (see ``next_thinking_seq``).

        ``effort`` is the frame's single thinking effort, taken from the impulse's ``thinking_effort``
        (``'none'`` = don't drive a turn, ``''`` = no opinion, a depth = apply it). Resolution:

            effort = frame_effort if frame_effort != '' else pending declaration (if any)

        A frame that carries its own effort wins and the pending declaration is untouched; the
        declaration is consumed only on the frame that actually **uses** it, so a ``'none'`` frame
        (or any other frame with its own effort) can never silently eat it.

        The depth travel is one-shot on this side but not one-turn on the model side: the plugin
        writes it as a pending model selection and releases it once ``request/header`` adopts it,
        which leaves the depth in the durable request config. The UI/settings can therefore change
        it afterwards — the ghost never pins the session.
        """
        moment = thinking.moment
        moment_ref = f"{thinking.observer.epoch.index}-{moment.index}"
        frame_effort = thinking.effort()
        if frame_effort != "":
            effort = frame_effort
        elif self.default_thinking_effort is not None:
            effort = self.default_thinking_effort
            self.default_thinking_effort = None
        else:
            effort = ""
        payload = {
            "moment": self._moment_payload(moment, moment_ref),
            "epoch": self._epoch_payload(thinking),
            # notices (commit 提醒 / 已提交告知) — 与 moment 同级注入, 每帧排空.
            "notices": self._drain_notices(),
            "effort": effort,
            "thinkingToken": self._thinking_token,
            "thinkingSeq": thinking_seq,
            # observe continuation: empty inputs still drive a turn — see needs_observe().
            "needsObserve": self.needs_observe(thinking),
        }
        await self._launcher.call(_DOLORES_THINKING_ENTER, payload)

    def _drain_notices(self) -> list[str]:
        """排空 notice 队列 → 纯文本 (与 moment 同级, 由 plugin 挂载到本步历史; notice 无图)."""
        notices, self._notices = self._notices, []
        return [notice.to_content_string() for notice in notices if not notice.is_empty()]

    async def exit_thinking(self, thinking_seq: int) -> None:
        """Reverse the thinking state; the plugin does the relevant teardown.

        A non-idle agent is cancelled by the plugin. Blocks for confirmation with a fail-safe timeout
        so a stalled plugin degrades instead of hanging the exit. ``thinking_seq`` is the one this
        frame's enter carried — the plugin closes only that enter's gate.
        """
        try:
            await self._launcher.call(
                _DOLORES_THINKING_EXIT,
                {
                    "thinkingToken": self._thinking_token,
                    "thinkingSeq": thinking_seq,
                },
                timeout=_EXIT_RPC_TIMEOUT,
            )
        except Exception:
            self._logger.exception("thinking/exit failed — degraded; state may be stale")

    def _context_message(self, moment: Moment, moment_id: str) -> Message | None:
        """context slot — as_moment_message excluding percept/hint (echoes/dynamic/executing).

        Folded into one ``<moment moment_id=...>`` message, injected as background. moment_id is a
        "{epoch.index}-{moment.index}" composite id (not a uuid). None when there is no context content.
        """
        return moment.as_moment_message(
            always_return=False,
            with_moment_id=False,
            with_percepts=False,
            with_hint=False,
            attributes={'moment_id': moment_id},
        )

    def _inputs_message(self, moment: Moment) -> Message | None:
        """inputs slot — percepts + hint wrapped into one ``<inputs>`` message (steer, may be empty).

        Reuses moment.inputs_messages with executing excluded — executing belongs to the context
        slot, not the inputs slot. None when there are no percepts and no hint.
        """
        messages: list[Message] = list(moment.inputs_messages(with_command_executing=False))
        if not messages:
            return None
        return Message.new(tag='inputs').with_messages(*messages)

    def _moment_payload(self, moment: Moment, moment_id: str) -> dict:
        """moment → wire content of two messages: context + inputs.

        context = the <moment> fold (echoes/dynamic/executing, inject); inputs = the <inputs> fold
        (percepts + hint, steer). The mapping is done Python-side; the plugin receives two ready
        content blocks. Text is passed through; images are converted to base64 EncodedImageAttachment
        (multimodal preserved). moment_id = "{epoch.index}-{moment.index}".
        """
        context_msg = self._context_message(moment, moment_id)
        inputs_msg = self._inputs_message(moment)
        return {
            "context": [
                self._content_payload(content)
                for content in context_msg.as_contents(with_meta=True)
            ] if context_msg is not None else [],
            "inputs": [
                self._content_payload(content)
                for content in inputs_msg.as_contents(with_meta=True)
            ] if inputs_msg is not None else [],
            "moment_id": moment_id,
        }

    def _content_payload(self, content: Content | dict) -> dict[str, Any]:
        """MOSS content → dsh wire content. Images keep their base64, reshaped as EncodedImageAttachment."""
        if content.get("type") == "image":
            source = content.get("source") or {}
            return {
                "type": "image",
                "mediaType": source.get("media_type"),
                "data": source.get("data", ""),
            }
        return content

    def _epoch_payload(self, thinking: "Thinking") -> list[dict] | None:
        """epoch slot — <cognition_epoch> container content blocks, only on epoch change.

        Rendered as a single ``<cognition_epoch index=N>`` container: ``<recap>`` background +
        ``<baseline>`` start info (each baseline key rendered as ``<key>value</key>``). The tag names
        the thing itself — a span of the ghost's own cognition — rather than leaving the model to
        guess what an "epoch" is. The plugin is a dumb transport — it only receives content blocks,
        it does not parse structure. Returns on the first frame and on every epoch change; None when
        unchanged.
        """
        epoch = thinking.observer.epoch
        if epoch.id == self._moment_epoch:
            return None
        self._moment_epoch = epoch.id
        children: list[Message] = []
        if epoch.recap:
            children.append(Message.new(tag="recap").with_messages(*epoch.recap))
        if epoch.baseline:
            baseline_msgs = [
                Message.new(tag=key).with_content(value)
                for key, value in epoch.baseline.items()
                if value
            ]
            children.append(Message.new(tag="baseline").with_messages(*baseline_msgs))
        if not children:
            return None
        container = Message.new(tag="cognition_epoch", attributes={"index": str(epoch.index)}).with_messages(*children)
        return [
            self._content_payload(content)
            for content in container.as_contents(with_meta=True)
        ]
