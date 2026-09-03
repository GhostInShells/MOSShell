"""DoloresEgo — the ego / continuity layer: thinking transaction + external-activity self-wake.

The Python half of the ego surface, paired with the dsh-side plugin (which owns the kernel:
ego/create, thinking enter/exit, tool-result, perStep lock). This module owns the ego session
state, moment→wire serialization, and the self-wake watcher.

Two lifecycle lines:

- long-lived (ghost lifetime): a background watcher on turn/start + user/message emits self-wake
  signals; suppressed while a thinking transaction is running.
- short-lived (per thinking): run_thinking() returns a DoloresRun — an async-with transaction
  boundary plus an event stream; its lifecycle (enter/exit/yield/observe/perStep) lives there.

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
from ghoshell_moss.deepseek_harness.types.session_events import SessionEvent
from ghoshell_moss.message import Content, Message

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

# thinking/exit confirmation timeout (fail-safe): degrade instead of hanging the exit if the plugin stalls.
_EXIT_RPC_TIMEOUT = 5.0


class DoloresEgoConfig(BaseModel):
    """Ego session config, loaded from the ``ego:`` section of .dolores.yml.

    Field defaults are the fallback — a missing YAML key uses the default rather than a manual .get().
    """

    agent_preset: str = Field(
        default="standard",
        description="dsh agent preset name — the ego session's persona + tool set.",
    )
    session_title: str = Field(
        default="{name} at {date}",
        description="session title template ({name}/{date} placeholders), the human-readable session name.",
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


@dataclasses.dataclass(frozen=True, slots=True)
class DoloresEgoContext:
    """Static context captured before the ego enters its lifecycle — assembled by the ghost, injected to avoid a back-ref.

    Values are read once at ego construction; nothing needs to reach back into the ghost afterward.
    All references are injected via typed objects / variables / closures — no back-ref is held.

    - project_home: working dir of the ego session.
    - project_name: workspace title.
    - name: ghost name, used for title/identity.
    - instruction: assembled system prompt.
    - facade: shell context surface (used to refresh meta on append_ctml).
    """

    project_home: Path
    project_name: str
    name: str
    instruction: str
    facade: "MShellContextFacade"


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
        """
        self._launcher = launcher
        self._ctx = ctx
        self._facade = ctx.facade
        self._config = config or DoloresEgoConfig()
        self._memories = memories
        self._session: "DshSession | None" = None
        self._ego_session_id: str | None = None
        # anti-bypass token: returned by ego/create, carried by thinking enter/exit, verified by the plugin to reject non-ego calls.
        self._thinking_token: str | None = None
        self._exit_stack = contextlib.AsyncExitStack()
        # logger: prefer the injected one (MOSS runtime logger), else fall back.
        self._logger = logger or get_moss_logger()
        # self-wake gate: whether a thinking transaction is running; the turn/start watcher reads this.
        self._thinking_event = asyncio.Event()
        # self-wake signal outlet — injected by host/mindflow after bus wiring; this side never touches the nucleus directly.
        self._signal_broadcast: "Callable[[Signal], None] | None" = None
        # epoch tracking: remembers the last injected epoch id, compared on enter to decide whether to carry an <epoch> container.
        self._moment_epoch: str | None = None

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
        """
        result = await self._launcher.call(
            _DOLORES_EGO_CREATE,
            {
                "project_home": str(self._ctx.project_home),
                "project_name": self._ctx.project_name,
                "title": self._config.session_title.format(
                    name=self._ctx.name,
                    date=datetime.now().strftime("%Y-%m-%d"),
                ),
                "instruction": self._ctx.instruction,
                "messages": self._assemble_initial_messages(),
                "agent_preset": self._config.agent_preset,
                "permission": self._config.permission,
            },
        )
        self._ego_session_id = result["sessionId"]
        self._thinking_token = result.get("thinkingToken")
        self._session = self._launcher.create_session(self._ego_session_id)
        await self._exit_stack.enter_async_context(self._session)
        # long-lived: subscribe to turn/start + user/message for silent self-wake.
        # user/message covers direct UI input — after a yield the dsh loop blocks on tool result;
        # UI input produces only user/message (not turn/start), so self-wake is still needed to unlock the pending tool.
        self._session.on_session_event("turn/start", self._on_session_activity)
        self._session.on_session_event("user/message", self._on_session_activity)
        return self._ego_session_id

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit: close the ego session."""
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    @property
    def session(self) -> "DshSession":
        """The held dsh session facade — raises a clear error before startup."""
        if self._session is None:
            raise RuntimeError("ego session not started. Call __aenter__ first.")
        return self._session

    # ── short-lived: run_thinking (transaction) ──────────────────────

    def run_thinking(self, thinking: "Thinking") -> "DoloresRun":
        """Return the run object for a thinking transaction — consumed by ``async with``.

        The async-with boundary is the explicit lifecycle (see DoloresRun): enter binds listeners and
        starts the enter task; exit cancels, unbinds, re-sends exit, and aborts. The consumer pulls
        events and dispatches logos/turn/end; the articulator is managed by the caller.

        :param thinking: the mindflow Thinking — moment/effort/articulator/abort all come from it.
        """
        from ._run import DoloresRun
        return DoloresRun(ego=self, thinking=thinking, thinking_event=self._thinking_event, facade=self._facade)

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
        """Emit a self-wake signal (silent when no broadcast is wired, for tests/pre-wiring)."""
        signal = new_dolores_ego_signal()
        if self._signal_broadcast is not None:
            self._signal_broadcast(signal)

    # ── RPC (narrow bridge to the plugin) ────────────────────────────

    async def _rpc_tool_result(
            self,
            call_id: str,
            result: dict | list | str | None,
            moment: list[dict] | None = None,
    ) -> None:
        """Unlock a pending tool: {callId, result, moment}.

        result = the tool's return value for the model (a "{epoch}-{moment}" short id for
        fetch_next_moment). moment = the moment content parts to inject into context (text + image);
        the plugin injects the moment then resolves the result. callId is passed through for routing.
        """
        await self._launcher.call(_DOLORES_TOOL_RESULT, {
            "callId": call_id,
            "result": result,
            "moment": moment,
        })

    def moment_context_parts(self, moment: Moment, moment_id: str) -> list[dict]:
        """moment → content blocks for context injection (context slot, excluding percept/hint).

        The fetch_next_moment tool injects the moment into the next step's context (background, does
        not drive a turn) via the tool-result RPC. Content blocks (text + image) are preserved, not
        folded to a string. Empty when there is no context content.
        """
        context_msg = self._context_message(moment, moment_id)
        if context_msg is None:
            return []
        return [self._content_payload(content) for content in context_msg.as_contents(with_meta=True)]

    async def enter_thinking(self, thinking: "Thinking") -> None:
        """Inject moment (context/inputs) + epoch + effort + model + thinkingToken to start a thinking turn."""
        moment = thinking.moment
        moment_ref = f"{thinking.observer.epoch.index}-{moment.index}"
        payload = {
            "moment": self._moment_payload(moment, moment_ref),
            "epoch": self._epoch_payload(thinking),
            "effort": thinking.effort(),
            "model": await self._model_config(),
            "thinkingToken": self._thinking_token,
        }
        await self._launcher.call(_DOLORES_THINKING_ENTER, payload)

    async def exit_thinking(self, *, yielded: bool = False) -> None:
        """Reverse the thinking state; the plugin does the relevant teardown.

        yielded: whether this break is a yield (wait_next_moment) — the plugin then does NOT cancel
        (the tool stays blocked awaiting the next moment), rather than relying on the plugin's own
        pendingYield timing. Non-yield + non-idle agent is cancelled by the plugin. Blocks for
        confirmation with a fail-safe timeout so a stalled plugin degrades instead of hanging the exit.
        """
        try:
            await self._launcher.call(
                _DOLORES_THINKING_EXIT,
                {
                    "thinkingToken": self._thinking_token,
                    "yielded": yielded,
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

        Percept messages are flattened in source order (no extra ``<percepts>`` wrapper); an optional
        hint is appended last as a ``<hint>`` child. None when there are no percepts and no hint.
        """
        messages: list[Message] = list(moment.percepts_messages())
        if moment.hint:
            messages.append(Message.new(tag='hint').with_content(moment.hint))
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
        """epoch slot — <epoch> container content blocks, only on epoch change.

        Rendered as a single ``<epoch index=N>`` container: ``<recap>`` background + ``<baseline>``
        start info (each baseline key rendered as ``<key>value</key>``). The plugin is a dumb
        transport — it only receives content blocks, it does not parse structure. Returns on the
        first frame and on every epoch change; None when unchanged.
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
        container = Message.new(tag="epoch", attributes={"index": str(epoch.index)}).with_messages(*children)
        return [
            self._content_payload(content)
            for content in container.as_contents(with_meta=True)
        ]

    async def _model_config(self) -> dict:
        """Current model config (provider/model/reasoningEffort), pulled via session.models."""
        selection = await self.session.model_selection()
        return {
            "provider": selection.provider,
            "model": selection.model,
            "reasoningEffort": selection.reasoningEffort,
        }
