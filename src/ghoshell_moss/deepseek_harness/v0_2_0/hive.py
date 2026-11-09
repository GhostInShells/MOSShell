"""DshHive — the place, and the collection that lives in it.

Two objects, one boundary:

- ``DshHive`` is the **place**: it holds the connection, reads the engine's declared workspaces and
  presets, and hands out a collection. It never knows how many agents exist.
- ``DshAgents`` is the **collection**: it holds the roster keyed by the runtime name, opens and
  takes up sessions, and owns the channel. It never invents policy — a workspace must be named, a
  preset must be declared, and both are asked of the hive that owns it.

The line is ownership and lifetime. Anything that outlives an agent (the connection, the declared
places) belongs to the hive; anything that dies with the process (the roster) belongs to the
collection. A verb on the wrong side of that line is what made the first version unclear.

Two read faces are cached rather than fetched because the blueprint's reads are synchronous: a
driver must be able to look without awaiting, so ``refresh()`` is the one async step that fills
them, and the channel's meta refresh is what calls it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from contextlib import AsyncExitStack
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ghoshell_dsh.v0_2_0 import DSH_VERSION
from ghoshell_dsh.v0_2_0.launcher import DshConnection
from ghoshell_dsh.v0_2_0.surfaces import DshConnectionSurface
from ghoshell_dsh.v0_2_0.types.sessions import (
    SessionCreateParams,
    SessionListParams,
    SessionListValue,
)

from ghoshell_moss.agents.blueprint.agent import (
    AgentPreset,
    Agents,
    AgentSession,
    AgentSessionEvent,
    EventHook,
    SessionId,
    SessionName,
    Workspace,
    WorkspaceName,
)
from ghoshell_moss.agents.blueprint.hive import Hive
from ghoshell_moss.core.blueprint.mindflow import Priority
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.mindflow.notify_nucleus import new_notify_signal
from ghoshell_moss.message import Message

from .agent import DshAgentSession

__all__ = ["DshAgents", "DshHive"]

SignalSender = Callable[[Any], None]
"""How a signal reaches the ghost — injected, so this module never imports Matrix."""

_SNAPSHOT_WAIT = 5.0
"""How long ``open`` waits for the follow snapshot before falling back to an RPC read."""

_REFRESH_TTL = 30.0
"""How long the cached declared places/templates stay good enough to read without asking again."""


def _to_workspace(view: Any) -> Workspace:
    """The engine's workspace row → this layer's model. One conversion, one place."""
    return Workspace(
        id=view.workspaceId,
        name=view.title or Path(view.path).name,
        path=view.path,
        title=view.title,
    )


class DshHive(Hive):
    """One dsh connection's place: its declared workspaces, its declared presets."""

    def __init__(
        self,
        connection: DshConnection,
        *,
        name: str = "dsh",
        root: str | Path = "",
        send_signal: SignalSender | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._connection = connection
        self._surface = DshConnectionSurface(connection)
        self._name = name
        self._root = Path(root).expanduser().resolve() if root else Path.cwd().resolve()
        self._send_signal = send_signal
        self._logger = logger or logging.getLogger("moss.dsh_hive")
        self._workspaces: list[Workspace] = []
        self._presets: list[AgentPreset] = []
        self._refreshed_at = 0.0
        self._agents = DshAgents(self)
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "DshHive":
        """Register the agents runtime on this hive's own exit stack."""
        await self._stack.__aenter__()
        await self._stack.enter_async_context(self._agents)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._stack.__aexit__(exc_type, exc_val, exc_tb)

    # -- identity ---------------------------------------------------------------- #

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
        """The engine connection this place reads — bound here, not part of the blueprint."""
        return self._connection

    @property
    def root(self) -> Path:
        """**The maximum boundary** — no place may be declared outside it."""
        return self._root

    @property
    def project_root(self) -> str:
        """Where ``@``-file references in a send resolve from; the same directory as ``root``."""
        return str(self._root)

    @property
    def send_signal(self) -> SignalSender | None:
        return self._send_signal

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    # -- the policy: places and templates ----------------------------------------- #

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

    def workspace(self, ref: WorkspaceName) -> Workspace:
        """Resolve a declared place by name, id or title. Raises when it is not declared."""
        for item in self._workspaces:
            if ref in (item.name, item.id, item.title):
                return item
        known = ", ".join(item.name or item.id for item in self._workspaces) or "(none)"
        raise KeyError(
            f"dsh hive {self._name}: no workspace named {ref!r} inside root {self._root}; "
            f"declared here: {known}"
        )

    async def create_workspace(self, path: str, name: str = "") -> Workspace:
        """Declare a place inside ``root`` — resolved against it, and refused outside it.

        Nothing is predefined: a fresh node has no workspace, and creating one is a runtime act.
        The outer edge is the hive's — a place outside ``root`` is a permission this hive does not
        have, so it raises rather than clamping the path.
        """
        target = (self._root / path).expanduser().resolve()
        try:
            target.relative_to(self._root)
        except ValueError:
            raise PermissionError(
                f"dsh hive {self._name}: {target} is outside the hive root {self._root}"
            ) from None
        # A declared place has to exist to be a boundary; dsh refuses a path that does not. The
        # hive owns everything under its root, so creating the directory is part of declaring it.
        target.mkdir(parents=True, exist_ok=True)
        # The engine's own answer is authoritative for *what was created*; the follow stream is
        # what keeps the list live. `create_workspace` waits for the frame and falls back to the
        # receipt, so nothing here has to re-read a possibly stale list.
        view, _created = await self._connection.create_workspace(str(target))
        if name and view.title != name:
            view = await self._connection.rename_workspace(view.workspaceId, name)
        workspace = _to_workspace(view)
        self._remember(workspace)
        return workspace

    def _owns(self, workspace: Workspace) -> bool:
        """Whether this place is inside the hive's root — the edge of what it may hand out."""
        if not workspace.path:
            return False
        try:
            Path(workspace.path).expanduser().resolve().relative_to(self._root)
        except ValueError:
            self._logger.debug(
                "dsh hive %s: ignoring workspace %s — outside root %s",
                self._name, workspace.path, self._root,
            )
            return False
        return True

    def _remember(self, workspace: Workspace) -> None:
        """Upsert one place into the cached list — the mutation's answer lands here at once."""
        for index, item in enumerate(self._workspaces):
            if item.id == workspace.id:
                self._workspaces[index] = workspace
                return
        self._workspaces.append(workspace)

    # -- the runtime --------------------------------------------------------------- #

    def agents(self) -> Agents:
        return self._agents

    # -- the one async read step ---------------------------------------------------- #

    async def refresh(self, *, force: bool = False, ttl: float = _REFRESH_TTL) -> None:
        """Fill the two cached read faces. Idempotent; safe to call on every meta refresh."""
        if not force and (time.monotonic() - self._refreshed_at) < ttl:
            return
        # A place is this hive's only if it lives inside the hive's root. The engine's list is
        # global — a node's dsh may carry a default workspace somewhere else on the machine — and
        # handing that out would give an agent a boundary this hive never granted.
        workspaces = [
            workspace
            for workspace in (_to_workspace(view) for view in await self._connection.workspaces())
            if self._owns(workspace)
        ]
        roster = await self._surface.presets()
        # Both reads succeeded — only now is the cache fresh. Stamping the time before the awaits
        # would turn one failure into thirty seconds of an empty list that looks authoritative.
        self._refreshed_at = time.monotonic()
        self._workspaces = workspaces
        self._presets = [
            AgentPreset(
                id=row.id,
                name=row.name or "",
                description=row.description or "",
                config_json_schema=_DSH_CONFIG_SCHEMA,
            )
            for row in roster.presets
        ]

    async def refresh_if_stale(self, *, ttl: float = _REFRESH_TTL) -> None:
        """The meta-refresh entry point: never raises, because a notice is not worth a crash."""
        try:
            await self.refresh(ttl=ttl)
        except Exception:  # noqa: BLE001
            self._logger.debug("dsh hive %s: refresh failed", self._name, exc_info=True)


class DshAgents(Agents):
    """The sessions one dsh hive holds, addressed by runtime name."""

    def __init__(self, hive: DshHive) -> None:
        self._hive = hive
        self._agents: dict[SessionName, DshAgentSession] = {}
        self._hooks: list[EventHook] = []
        self._tasks: dict[SessionName, asyncio.Task] = {}
        self._subs: dict[SessionName, Callable[[], None]] = {}
        self._fresh: set[SessionName] = set()
        self._started = False

    async def __aenter__(self) -> "DshAgents":
        """The runtime is entered by the hive's exit stack — it exits with everything else."""
        self._started = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.clear()
        self._started = False

    @property
    def hive(self) -> DshHive:
        """Bound here for the channel and the tests; not part of the blueprint."""
        return self._hive

    # -- discovery ------------------------------------------------------------------ #

    def list_sessions(self) -> list[AgentSession]:
        return list(self._agents.values())

    def get(self, name: SessionName) -> AgentSession:
        agent = self._agents.get(name)
        if agent is None:
            known = ", ".join(self._agents) or "(none)"
            raise NameError(f"dsh agents: no agent named {name!r}; held: {known}")
        return agent

    def on_event(self, callback: EventHook) -> Callable[[], None]:
        self._hooks.append(callback)

        def _dispose() -> None:
            if callback in self._hooks:
                self._hooks.remove(callback)

        return _dispose

    # -- lifecycle -------------------------------------------------------------------- #

    async def open(
        self,
        name: SessionName,
        *,
        workspace: WorkspaceName,
        preset: str = "",
        readonly: bool = False,
    ) -> AgentSession:
        if name in self._agents:
            raise ValueError(f"dsh agents: {name!r} is already held")
        place = self._hive.workspace(workspace)
        # dsh takes `workspaceId` **or** `cwd`, never both — and the workspace is the declared
        # place, so it wins; the cwd the engine derives from it is read back on attach.
        create = SessionCreateParams(agentPreset=preset or None)
        if place.id:
            create.workspaceId = place.id
        else:
            create.cwd = place.path or None
        value = await self._hive.connection.client.session_create(create)
        agent = await self._attach(
            str(value.sessionId),
            name=name,
            preset=value.agentPreset or preset,
            workspace_id=place.id,
            bumped=True,
        )
        if readonly:
            # Creation-time permission. dsh cannot express it in `session/create`, so it lands
            # before this method returns — that is, before any caller can send the first message.
            # The invariant: no turn ever runs under a permission nobody asked for.
            await agent.engine_session.set_policy(sandbox="read-only")
            agent.prime(readonly=True)
        return agent

    async def watch(self, name: SessionName, session_id: SessionId) -> AgentSession:
        if name in self._agents:
            raise ValueError(f"dsh agents: {name!r} is already held")
        if not await self._session_exists(session_id):
            raise KeyError(f"dsh agents: no session {session_id!r} exists to watch")
        agent = await self._attach(session_id, name=name, bumped=True)
        preset_id = await agent.engine_session.agent_preset() or ""
        if preset_id:
            agent.prime(preset=preset_id)
            agent._preset = self._preset_snapshot(preset_id)  # noqa: SLF001 — the snapshot is ours
        return agent

    async def retire(self, name: SessionName, reason: str = "") -> None:
        """End an agent's membership by stopping the runtime that carries it.

        There is no separate "close the session" step here on purpose: the task **is** the runtime,
        so cancelling it stops the session, drops the subscriptions and takes it off the roster —
        all in one place, in one order.
        """
        agent = self._agents.get(name)
        if agent is None:
            raise NameError(f"dsh agents: no agent named {name!r}")
        if reason:
            with contextlib.suppress(Exception):
                await agent.cancel(reason)
        task = self._tasks.get(name)
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        else:
            await agent.close()
            self._discord(name, agent)
        self._hive.logger.info("dsh agents: retired %s (%s)", name, reason or "-")

    async def clear(self) -> None:
        for name in list(self._agents):
            await self.retire(name)

    # -- presentation ------------------------------------------------------------------ #

    def as_channel(self) -> Channel | None:
        from .channel import new_agents_channel

        return new_agents_channel(self)

    # -- internals ----------------------------------------------------------------------- #

    async def _attach(
        self,
        session_id: SessionId,
        *,
        name: SessionName,
        preset: str = "",
        cwd: str = "",
        workspace_id: str = "",
        bumped: bool = False,
    ) -> DshAgentSession:
        session = self._hive.connection.create_session(session_id, self._hive.logger)
        # The session is a lifecycle object: `__aenter__` starts the consume loop that turns
        # follow frames into dispatched events. Without it the stream is alive and silent.
        await session.__aenter__()
        # ...and the constant facts (cwd / preset) arrive as the follow snapshot, not as part of
        # create. Wait for that frame; an RPC for what the stream is already sending is the same
        # mistake as reading the workspace list right after a mutation.
        await session.wait_follow_snapshot(timeout=_SNAPSHOT_WAIT)
        agent = DshAgentSession(
            session,
            hive=self._hive.name,
            engine=f"{self._hive.engine_name}@{self._hive.version}",
            name=name,
            preset=self._preset_snapshot(preset),
            cwd=cwd,
            workspace_id=workspace_id,
            on_event=self._fanout,
        )
        if not cwd:
            try:
                cwd = await session.cwd() or ""
            except Exception:  # noqa: BLE001 — an unknown cwd is not a reason to refuse a session
                cwd = ""
        agent.prime(preset=preset, cwd=cwd, workspace_id=workspace_id)
        self._agents[name] = agent
        if bumped:
            self._fresh.add(name)
        # The subscription belongs to the runtime task's lifetime, not to the caller's: `_discord`
        # drops it, and `_govern` calls `_discord` however the task ends.
        self._subs[name] = agent.on_turn_end(self._turn_end_hook(name))
        # One task per session: the task *is* the agent's lifetime. When the carrier closes, the
        # task wakes and the roster follows — a dead task can never look like a live agent.
        self._tasks[name] = asyncio.create_task(self._govern(name, agent))
        return agent

    async def _govern(self, name: SessionName, agent: DshAgentSession) -> None:
        """The agent's runtime task — one session, owned for its whole life.

        Stopping this task **is** stopping the session: a cancelled task becomes
        ``cancel(reason)`` + ``close()``, so no carrier is ever left running with nobody holding
        it. Whichever way it ends, the subscriptions go with it — after this task returns, nothing
        that session emits can reach a caller.
        """
        try:
            await agent.wait_closed()
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await agent.cancel("the agent's runtime was stopped")
            with contextlib.suppress(Exception):
                await agent.close()
            raise
        finally:
            self._discord(name, agent)

    def _discord(self, name: SessionName, agent: DshAgentSession) -> None:
        """Cut every subscription this agent had, and take it off the roster.

        "Discord" is the whole point of an orderly end: a closed session can still have frames
        queued behind it, and a hook that survives would deliver them to a caller who believes the
        agent is gone. Unsubscribing first is what makes "closed" mean "silent".
        """
        disposer = self._subs.pop(name, None)
        if disposer is not None:
            try:
                disposer()
            except Exception:  # noqa: BLE001 — a broken disposer must not block the teardown
                pass
        if self._agents.get(name) is agent:
            self._agents.pop(name, None)
        self._tasks.pop(name, None)
        self._fresh.discard(name)

    def _preset_snapshot(self, preset_id: str) -> AgentPreset:
        """The declaration as it stands now — the snapshot a session keeps for its lifetime."""
        if not preset_id:
            return AgentPreset(id="")
        try:
            return self._hive.preset(preset_id).model_copy(deep=True)
        except KeyError:
            return AgentPreset(id=preset_id)

    async def _session_exists(self, session_id: SessionId) -> bool:
        """Whether the engine still knows this session. **Fails closed**: unverifiable is refused.

        Taking up a session that is gone leaves a task waiting on a close that never comes, which
        looks exactly like a healthy idle agent. Refusing is the cheaper mistake.
        """
        value = await self._hive.connection.client.call(
            "session/list", SessionListParams(), SessionListValue, args_key="_request"
        )
        return any(item.sessionId == session_id for item in value.items)

    def _fanout(self, event: AgentSessionEvent) -> None:
        for hook in list(self._hooks):
            try:
                hook(event)
            except Exception:  # noqa: BLE001 — one bad subscriber must not stop the others
                continue


    def _turn_end_hook(self, name: SessionName) -> Callable[[str, AgentSessionEvent], None]:
        """The callback for one agent — bound to its name, so the signal can address it back."""

        def _hook(text: str, event: AgentSessionEvent) -> None:
            self._signal_answer(name, text)

        return _hook

    def _signal_answer(self, name: SessionName, text: str) -> None:
        """Deliver one agent's turn end as a callback addressed by the hive's name.

        The container carries only what is needed to walk back to the source: the hive (as the tag
        — and the hive's name is the node's name), the agent's runtime name, and a timestamp.
        Nothing else. Not the session id: that is the engine's internal handle, and a name can
        outlive it, so carrying one only invites a stale one to be trusted.

        ``next=True`` on every one of them — an answer that loses the arbitration is an answer the
        ghost never learns about. The level starts at BACKGROUND; only an agent's first turn after
        ``open``/``watch`` is raised, capped at NOTICE.
        """
        sender = self._hive.send_signal
        if sender is None or not text.strip():
            return
        first = name in self._fresh
        self._fresh.discard(name)
        container = Message.new(
            tag=self._hive.name,
            attributes={"agent": name},
            timestamp=True,
        ).with_content(text)
        signal = new_notify_signal(
            container,
            priority=Priority.NOTICE if first else Priority.BACKGROUND,
            next=True,
            description=f"{self._hive.name} · {name} ended a turn",
        )
        try:
            sender(signal)
        except Exception:  # noqa: BLE001 — a signal that cannot be sent must not kill the stream
            self._hive.logger.debug("dsh agents: turn-end signal dropped", exc_info=True)


# The patch surface below is dsh 0.2.0's own: engine knowledge, kept in the implementation.
_DSH_CONFIG_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "A patch. Absent fields keep their current value; nothing is reset by omission.",
    "properties": {
        "llm": {
            "type": "object",
            "description": "Model route for subsequent turns.",
            "properties": {
                "provider": {"type": "string"},
                "model": {"type": "string"},
                "reasoningEffort": {
                    "type": "string",
                    "description": "`off` disables thinking — it is an instruction, not 'unset'.",
                },
            },
            "required": ["provider", "model"],
        },
        "sandbox": {"type": "string", "enum": ["read-only", "workspace-write"]},
        "approval": {"type": "string", "enum": ["ask", "never"]},
    },
    "additionalProperties": False,
}
