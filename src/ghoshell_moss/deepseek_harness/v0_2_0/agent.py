"""DshAgentSession — one dsh session, carried as an ``AgentSession``.

The engine owns the turn; this class owns the **translation**: one lifecycle phase per moment,
the final answer as plain text, and a state that never asks the network for what it already
mirrored from the event stream.

Nothing here knows about branches, ops or coordinates. If a capability needs a trajectory to
exist, it belongs to the layer above this one.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ghoshell_dsh.v0_2_0.session import DshSession
from ghoshell_dsh.v0_2_0.types.sessions import PromptContentPart
from ghoshell_dsh.v0_2_0.types.session_events import (
    AssistantMessageEvent,
    ToolCallEvent,
    ToolResultEvent,
    TurnEnd,
    TurnStart,
    UserMessageEvent,
)

from ghoshell_moss.agents.blueprint.agent import (
    AgentPhase,
    AgentSession,
    AgentSessionEvent,
    AgentState,
    EventHook,
    MessageMode,
)
from ghoshell_moss.message import Message, Text

__all__ = ["DshAgentSession"]

_PREVIEW_CAP = 400
"""How much text one event carries in ``description`` — enough to recognise, not to re-read."""

_PROMPT_MODE = {MessageMode.enqueue: "queue", MessageMode.steer: "steer"}
"""Our three boundaries mapped onto the engine's two prompt modes; ``inject`` takes another verb."""


def message_text(message: Message) -> str:
    """The plain text of a MOSS message — what dsh can take without knowing our content model.

    Text is the only content this layer forwards. An image or any other block would need a
    mapping into the engine's own part protocol; dropping it silently would be worse than
    stating the limit, so callers that need more go one level down to the engine session.
    """
    parts: list[str] = []
    for content in message.contents:
        text = content.text if isinstance(content, Text) else _content_text(content)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _content_text(content: object) -> str:
    if isinstance(content, dict):
        value = content.get("text", "")
        return value if isinstance(value, str) else ""
    value = getattr(content, "text", "")
    return value if isinstance(value, str) else ""


class DshAgentSession(AgentSession):
    """One dsh session, driven through the engine's own verbs."""

    def __init__(
        self,
        session: DshSession,
        *,
        hive: str,
        engine: str,
        alias: str = "",
        title: str = "",
        cwd: str = "",
        workspace_id: str = "",
        on_event: Callable[[AgentSessionEvent], None] | None = None,
    ) -> None:
        self._session = session
        self._hive = hive
        self._engine = engine
        self._alias = alias
        self._on_event = on_event
        self._event_hooks: list[EventHook] = []
        self._output_hooks: list[Callable[[str, AgentSessionEvent], Any]] = []
        self._disposers: list[Callable[[], None]] = []
        self._closed = False
        self._state = AgentState(
            id=session.session_id,
            title=title,
            preset="",
            cwd=cwd,
            workspace_id=workspace_id,
        )
        self._subscribe()

    # -- identity -------------------------------------------------------------- #

    @property
    def alias(self) -> str:
        """The runtime alias this session was opened under; empty when it has none."""
        return self._alias

    @property
    def session_id(self) -> str:
        return self._state.id

    @property
    def engine_session(self) -> DshSession:
        """The engine handle, for an implementation that needs to go one level down."""
        return self._session

    @property
    def closed(self) -> bool:
        return self._closed

    # -- the cheap read ---------------------------------------------------------- #

    def state(self) -> AgentState:
        return self._state.model_copy()

    def _update(self, **fields: Any) -> None:
        self._state = self._state.model_copy(update=fields)

    # -- event translation -------------------------------------------------------- #

    def _subscribe(self) -> None:
        session = self._session

        async def _on_user(event: UserMessageEvent) -> None:
            # A user-**role** message is not necessarily the human. dsh writes injected context
            # (instruction files, the runtime policy snapshot, agent presets) as user messages, so
            # one prompt arrives as three: the prompt plus two injections. Reading ``source.kind``
            # is the only thing that separates them, and without it every input count is wrong.
            source = getattr(event, "source", None)
            kind = getattr(source, "kind", "") or "user"
            if kind != "user":
                return
            text = _user_text(event)
            self._emit(
                AgentPhase.input,
                event_type(event),
                _preview(text),
                {"text": text},
            )

        async def _on_turn_start(event: TurnStart) -> None:
            self._update(phase=AgentPhase.running)
            self._emit(AgentPhase.running, event_type(event), "")

        async def _on_assistant(event: AssistantMessageEvent) -> None:
            text = assistant_text(event)
            payload = {"text": text, "message": event.model_dump(mode="json")}
            item = self._emit(AgentPhase.output, event_type(event), _preview(text), payload)
            self._update(phase=AgentPhase.idle)
            for hook in list(self._output_hooks):
                _call(hook, text, item)

        async def _on_tool_call(event: ToolCallEvent) -> None:
            name = getattr(event, "name", "") or getattr(event, "tool", "")
            self._emit(AgentPhase.tool_use, event_type(event), str(name), event.model_dump(mode="json"))

        async def _on_tool_result(event: ToolResultEvent) -> None:
            self._emit(AgentPhase.tool_use, event_type(event), "", event.model_dump(mode="json"))

        async def _on_turn_end(event: TurnEnd) -> None:
            self._update(phase=AgentPhase.idle)
            reason = getattr(getattr(event, "reason", None), "kind", "") or ""
            self._emit(AgentPhase.idle, event_type(event), reason)

        self._disposers = [
            session.on_session_event_model(UserMessageEvent, _on_user),
            session.on_session_event_model(TurnStart, _on_turn_start),
            session.on_session_event_model(AssistantMessageEvent, _on_assistant),
            session.on_session_event_model(ToolCallEvent, _on_tool_call),
            session.on_session_event_model(ToolResultEvent, _on_tool_result),
            session.on_session_event_model(TurnEnd, _on_turn_end),
        ]

    def _emit(
        self,
        phase: AgentPhase,
        name: str,
        description: str = "",
        payload: dict[str, Any] | None = None,
    ) -> AgentSessionEvent:
        item = AgentSessionEvent(
            hive=self._hive,
            engine=self._engine,
            session_id=self._state.id,
            preset=self._state.preset,
            event_name=name,
            description=description,
            phase=phase,
            payload=payload or {},
        )
        for hook in list(self._event_hooks):
            _call(hook, item)
        if self._on_event is not None:
            self._on_event(item)
        return item

    # -- observation --------------------------------------------------------------- #

    def on_session_event(self, hook: EventHook) -> Callable[[], None]:
        self._event_hooks.append(hook)

        def _dispose() -> None:
            if hook in self._event_hooks:
                self._event_hooks.remove(hook)

        return _dispose

    def on_output(self, callback: Callable[[str, AgentSessionEvent], Any]) -> Callable[[], None]:
        self._output_hooks.append(callback)

        def _dispose() -> None:
            if callback in self._output_hooks:
                self._output_hooks.remove(callback)

        return _dispose

    # -- driving -------------------------------------------------------------------- #

    async def send(self, *messages: Message, mode: MessageMode = MessageMode.enqueue) -> None:
        text = "\n".join(part for part in (message_text(m) for m in messages) if part)
        if not text:
            raise ValueError("dsh: nothing to send — the messages carried no text")
        if mode is MessageMode.inject:
            # inject never wakes the carrier: it is held until something else starts a turn.
            await self._session.inject(text, wait="accepted")
            return
        content = [PromptContentPart(type="text", text=text)]
        await self._session.prompt(content=content, mode=_PROMPT_MODE[mode])

    async def cancel(self, reason: str = "") -> None:
        await self._session.cancel()

    async def configure(self, patch: dict[str, Any]) -> str:
        applied: list[str] = []
        llm = patch.get("llm") or {}
        if llm:
            provider = llm.get("provider") or ""
            model = llm.get("model") or ""
            if not provider or not model:
                raise ValueError("dsh: an llm patch needs both `provider` and `model`")
            await self._session.select_model(
                provider=provider, model=model, reasoning_effort=llm.get("reasoningEffort")
            )
            applied.append(f"llm={provider}/{model}")
            config = dict(self._state.config)
            config["llm"] = {**dict(config.get("llm") or {}), **llm}
            self._update(config=config)
        sandbox = patch.get("sandbox")
        approval = patch.get("approval")
        if sandbox is not None or approval is not None:
            policy = await self._session.set_policy(sandbox=sandbox, approval=approval)
            applied.append(f"sandbox={policy.sandbox} approval={policy.approval}")
            config = dict(self._state.config)
            config["sandbox"] = policy.sandbox
            config["approval"] = policy.approval
            self._update(config=config)
        if not applied:
            return "nothing to apply"
        return ", ".join(applied)

    # -- teardown (the hive's act, called by it) --------------------------------------- #

    async def close(self) -> None:
        """Stop the carrier and stop translating. **Called by the hive**, never by itself."""
        if self._closed:
            return
        self._closed = True
        for dispose in self._disposers:
            _call(dispose)
        self._disposers.clear()
        self._emit(AgentPhase.closed, "session/closed")
        self._event_hooks.clear()
        self._output_hooks.clear()
        try:
            await self._session.close()
        finally:
            self._update(phase=AgentPhase.closed)

    # -- state priming (called by the hive after the engine answers) -------------------- #

    def prime(self, *, title: str = "", preset: str = "", cwd: str = "", workspace_id: str = "") -> None:
        """Fill the structural facts once, from the engine's own answer."""
        self._update(
            title=title or self._state.title,
            preset=preset or self._state.preset,
            cwd=cwd or self._state.cwd,
            workspace_id=workspace_id or self._state.workspace_id,
        )


def event_type(event: Any) -> str:
    """The engine's own name for this event — passed through, never translated."""
    getter = getattr(type(event), "event_type", None)
    if callable(getter):
        try:
            return str(getter())
        except Exception:  # noqa: BLE001 — a name is worth less than a working stream
            pass
    return type(event).__name__


def _user_text(event: UserMessageEvent) -> str:
    getter = getattr(event, "text", None)
    if callable(getter):
        try:
            text = str(getter() or "")
        except Exception:  # noqa: BLE001 — a shape we do not know is not worth a crash
            text = ""
        if text:
            return text
    return blocks_text(getattr(event, "content", None))


def blocks_text(blocks: Any) -> str:
    """Plain text of a block list — text blocks only, whatever the engine calls the field."""
    if not isinstance(blocks, list):
        return ""
    parts: list[str] = []
    for block in blocks:
        kind = getattr(block, "type", "") or (block.get("type") if isinstance(block, dict) else "")
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if kind == "text" and text:
            parts.append(str(text))
    return "\n".join(parts).strip()


def assistant_text(event: AssistantMessageEvent) -> str:
    """The assistant's prose for this turn — text blocks only.

    Public-internal: the event translation calls it, and the tests read it as the one place that
    decides "what counts as the answer".

    Reasoning and tool traffic travel in the same message; they are the agent's working, not its
    answer, so they never reach ``on_output`` or a signal.
    """
    message = getattr(event, "message", None)
    if message is None:
        return ""
    return blocks_text(getattr(message, "content", None))


def _preview(text: str) -> str:
    text = (text or "").strip()
    return text if len(text) <= _PREVIEW_CAP else text[:_PREVIEW_CAP] + "…"


def _call(fn: Callable[..., Any], *args: Any) -> None:
    """Call a hook that may be sync or async, without letting it break the stream."""
    try:
        result = fn(*args)
    except Exception:  # noqa: BLE001 — one bad subscriber must not stop the others
        return
    if _is_awaitable(result):
        from ghoshell_moss.core.blueprint.channel_builder import CommandUtil
        try:
            CommandUtil.create_task(result)
        except Exception:  # noqa: BLE001 — outside a channel runtime there is no task scope
            pass


def _is_awaitable(value: Any) -> bool:
    import inspect

    return inspect.isawaitable(value)
