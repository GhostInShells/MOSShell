"""DshHive — one dsh connection's sessions, held as an ``AgentSession`` roster.

This is the implementation of the MOSS hive blueprint on dsh 0.2.0. It **does not own the
process**: it is handed an entered ``DshConnection`` and holds sessions on it. Where that
connection comes from — a launcher this node started, or a dsh somebody else already runs — is
the node's decision, not the hive's.

Baseline rule inherited from the blueprint: everything held here is **runtime and in memory**.
Nothing is promised across a restart; coming back to a session is an explicit ``resume`` by id.

Two facts are cached rather than fetched, and the reason is the same for both: the blueprint's
read faces are synchronous because a driver must be able to look without awaiting. ``refresh()``
is the one async step that fills them.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from ghoshell_dsh.v0_2_0 import DSH_VERSION
from ghoshell_dsh.v0_2_0.launcher import DshConnection
from ghoshell_dsh.v0_2_0.surfaces import DshConnectionSurface
from ghoshell_dsh.v0_2_0.types.sessions import SessionCreateParams

from ghoshell_moss.agents.blueprint.agent import (
    AgentPreset,
    AgentSession,
    AgentSessionEvent,
    AgentState,
    Hive,
    MessageMode,
    Workspace,
)
from ghoshell_moss.core.blueprint.channel_builder import MutableChannel
from ghoshell_moss.core.blueprint.mindflow import Priority
from ghoshell_moss.core.mindflow.notify_nucleus import new_notify_signal
from ghoshell_moss.message import Message

from .agent import DshAgentSession

__all__ = ["DshHive"]

SignalSender = Callable[[Any], None]
"""How a signal reaches the ghost — injected, so this module never imports Matrix."""

_EVENT_HISTORY = 512
"""How many events the pull side keeps. A debug/observability tail, not a transcript."""


class DshHive(Hive):
    """A dsh connection's sessions, addressed inside one hive."""

    def __init__(
        self,
        connection: DshConnection,
        *,
        name: str = "dsh",
        project_root: str = "",
        send_signal: SignalSender | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._connection = connection
        self._surface = DshConnectionSurface(connection)
        self._name = name
        self._project_root = project_root
        self._send_signal = send_signal
        self._logger = logger or logging.getLogger("moss.dsh_hive")
        self._agents: dict[str, DshAgentSession] = {}
        self._aliases: dict[str, str] = {}
        self._events: deque[tuple[int, AgentSessionEvent]] = deque(maxlen=_EVENT_HISTORY)
        self._cursor = 0
        self._workspaces: list[Workspace] = []
        self._presets: list[AgentPreset] = []
        self._refreshed_at = 0.0
        self._closed = False

    # -- identity ------------------------------------------------------------------ #

    @property
    def name(self) -> str:
        return self._name

    @property
    def engine_name(self) -> str:
        return "dsh"

    @property
    def version(self) -> str:
        return DSH_VERSION

    @property
    def connection(self) -> DshConnection:
        """The engine connection this hive drives — bound here, not part of the blueprint."""
        return self._connection

    @property
    def project_root(self) -> str:
        """Where ``@``-file references in a send are resolved from."""
        return self._project_root

    # -- discovery (cached; see refresh) --------------------------------------------- #

    def workspaces(self) -> list[Workspace]:
        return list(self._workspaces)

    def presets(self) -> list[AgentPreset]:
        return list(self._presets)

    def preset(self, ref: str) -> AgentPreset:
        for item in self._presets:
            if item.id == ref or (item.name and item.name == ref):
                return item
        known = ", ".join(item.id for item in self._presets) or "(none)"
        raise KeyError(f"dsh hive {self._name}: no preset {ref!r}; available: {known}")

    def sessions(self) -> list[AgentState]:
        return [agent.state() for agent in self._agents.values()]

    def session(self, session_id: str) -> AgentSession | None:
        return self._agents.get(self._resolve(session_id))

    def alias_of(self, session_id: str) -> str:
        """The runtime alias a session was opened under, or ``""``."""
        for alias, known in self._aliases.items():
            if known == session_id:
                return alias
        return ""

    async def refresh_if_stale(self, *, ttl: float = 30.0) -> None:
        """The meta-refresh entry point: never raises, because a notice is not worth a crash."""
        try:
            await self.refresh(ttl=ttl)
        except Exception:  # noqa: BLE001
            self._logger.debug("dsh hive %s: refresh failed", self._name, exc_info=True)

    # -- the one async read step ------------------------------------------------------ #

    async def refresh(self, *, force: bool = False, ttl: float = 30.0) -> None:
        """Fill the two cached read faces. Idempotent, and safe to call on every meta refresh."""
        if not force and (time.monotonic() - self._refreshed_at) < ttl:
            return
        self._refreshed_at = time.monotonic()
        self._workspaces = [
            Workspace(id=item.workspaceId, path=item.path, title=item.title)
            for item in await self._connection.workspaces()
        ]
        roster = await self._surface.presets()
        self._presets = [
            AgentPreset(
                id=row.id,
                name=row.name or "",
                description=row.description or "",
                config_json_schema=_CONFIG_SCHEMA,
                **({"broken": row.broken} if row.broken else {}),
            )
            for row in roster.presets
        ]

    # -- lifecycle -------------------------------------------------------------------- #

    async def open(
        self,
        *,
        preset: str = "",
        cwd: str = "",
        title: str = "",
        name: str = "",
    ) -> AgentSession:
        self._ensure_open()
        value = await self._connection.client.session_create(
            SessionCreateParams(cwd=cwd or None, agentPreset=preset or None)
        )
        session_id = str(value.sessionId)
        agent = await self._attach(
            session_id, preset=value.agentPreset or preset, cwd=cwd, title=title, alias=name
        )
        if title:
            await agent.engine_session.rename(title=title)
        if name:
            self._alias(name, session_id)
        self._logger.info("dsh hive %s: opened %s (%s)", self._name, session_id, name or "-")
        return agent

    async def resume(
        self,
        session_id: str,
        *,
        name: str = "",
    ) -> AgentSession:
        """Take up a session this hive no longer holds — the only way back in."""
        self._ensure_open()
        agent = await self._attach(session_id, alias=name)
        if name:
            self._alias(name, session_id)
        return agent

    async def retire(self, to: str, reason: str = "") -> None:
        session_id = self._resolve(to)
        agent = self._agents.get(session_id)
        if agent is None:
            raise KeyError(f"dsh hive {self._name}: no session {to!r}")
        await self._teardown(agent)
        self._logger.info("dsh hive %s: retired %s (%s)", self._name, session_id, reason or "-")

    async def close(self) -> None:
        """Close every held session, in order. The connection belongs to whoever passed it in."""
        if self._closed:
            return
        self._closed = True
        for agent in list(self._agents.values()):
            await self._teardown(agent)
        self._aliases.clear()

    # -- driving ----------------------------------------------------------------------- #

    async def send(
        self,
        to: str,
        *messages: Message,
        mode: MessageMode = MessageMode.enqueue,
    ) -> None:
        await self._agent(to).send(*messages, mode=mode)

    async def cancel(self, to: str, reason: str = "") -> None:
        await self._agent(to).cancel(reason)

    # -- observation -------------------------------------------------------------------- #

    def events(self, since: int = 0) -> tuple[list[AgentSessionEvent], int]:
        items = [item for seq, item in self._events if seq >= since]
        return items, self._cursor + 1

    # -- presentation --------------------------------------------------------------------- #

    def as_channel(self) -> MutableChannel:
        from .channel import new_dsh_hive_channel

        return new_dsh_hive_channel(self)

    # -- internals ------------------------------------------------------------------------ #

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"dsh hive {self._name} is closed")

    async def _attach(
        self, session_id: str, *, preset: str = "", cwd: str = "", title: str = "", alias: str = ""
    ) -> DshAgentSession:
        existing = self._agents.get(session_id)
        if existing is not None:
            return existing
        session = self._connection.create_session(session_id, self._logger)
        # The session is a lifecycle object, not a plain facade: `__aenter__` starts the consume
        # loop that turns follow frames into dispatched events. Without it every event is queued
        # and never delivered — the stream looks alive and reports nothing.
        await session.__aenter__()
        agent = DshAgentSession(
            session,
            hive=self._name,
            engine=f"{self.engine_name}@{self.version}",
            alias=alias,
            title=title,
            cwd=cwd,
            on_event=self._collect,
        )
        if not cwd:
            # The engine is the authority on where this session may act; ask it once, at attach.
            try:
                cwd = await session.cwd(force=True) or ""
            except Exception:  # noqa: BLE001 — an unknown cwd is not a reason to refuse the session
                cwd = ""
        agent.prime(preset=preset, cwd=cwd, title=title)
        self._agents[session_id] = agent
        return agent

    async def _teardown(self, agent: DshAgentSession) -> None:
        await agent.close()
        self._agents.pop(agent.session_id, None)
        for alias, session_id in list(self._aliases.items()):
            if session_id == agent.session_id:
                self._aliases.pop(alias, None)

    def _alias(self, name: str, session_id: str) -> None:
        self._aliases[name] = session_id

    def _resolve(self, to: str) -> str:
        return self._aliases.get(to, to)

    def _agent(self, to: str) -> DshAgentSession:
        session_id = self._resolve(to)
        agent = self._agents.get(session_id)
        if agent is None:
            known = ", ".join(list(self._aliases) + list(self._agents)) or "(none)"
            raise KeyError(f"dsh hive {self._name}: no session {to!r}; known: {known}")
        return agent

    def _collect(self, item: AgentSessionEvent) -> None:
        """Every event goes into the pull tail; an output also reaches the ghost as a signal."""
        self._cursor += 1
        self._events.append((self._cursor, item))
        if item.phase.value == "output" and self._send_signal is not None:
            self._signal_output(item)

    def _signal_output(self, item: AgentSessionEvent) -> None:
        alias = ""
        agent = self._agents.get(item.session_id)
        if agent is not None:
            alias = agent.alias
        who = alias or item.session_id
        text = (item.description or "").strip() or "(no text)"
        signal = new_notify_signal(
            f"[dsh {who}] output: {text}",
            priority=Priority.NOTICE,
            description=f"dsh session {who} finished a turn",
            hint="output",
        )
        try:
            self._send_signal(signal)
        except Exception:  # noqa: BLE001 — a signal that cannot be sent must not kill the stream
            self._logger.debug("dsh hive %s: output signal dropped", self._name, exc_info=True)


# The patch surface below is dsh 0.2.0's own: engine knowledge, kept in the implementation.
_CONFIG_SCHEMA: dict[str, Any] = {"type": "object"}
