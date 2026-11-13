"""The agents channel — a child of its owner's main channel, one fragment per held agent.

Shape, and why:

- **A child, not a root.** The node is the address; this is an organ of it. A root channel here
  would compete with the node for the same name, and the ghost would have two things called `dsh`.
- **One control plane, for the agents.** Every *agent* verb lives here and takes a name. The place
  itself — workspaces — is declared on the owner's channel, because that is where policy belongs.
  An agent is reached through the collection (`get(name)`), so this channel is an adapter and never
  a second implementation.
- **Named notices are the perception face.** One fragment per held agent showing its session id and
  phase — the only two facts a driver needs every turn. A folded notice cannot express "this agent
  is gone"; a named fragment can, because omission *is* removal.
- **Cold reads stay cold.** `state` is a command, never notice material: a notice that grows with
  every agent is a notice nobody reads.
"""

from __future__ import annotations

import json
from pathlib import Path

from ghoshell_moss.agents.blueprint.agent import (
    AgentPreset,
    Agents,
    AgentSession,
    AgentState,
    MessageMode,
    Workspace,
)
from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel
from ghoshell_moss.message import Message
from ghoshell_moss.message.prompt import message_from_prompt

from .hive import DshAgents

__all__ = ["new_agents_channel"]

CHANNEL_NAME = "agents"

_MODE_VALUES = {mode.value for mode in MessageMode}

_INSTRUCTION = """\
The agents this node holds, addressed by the name each was opened under.

`open(name, workspace, preset, readonly)` creates one inside a declared workspace;
`watch(name, session_id)` takes up one that already exists. `send(name, text)` is fire-and-forget —
it answers `ok` or `failed`, and the reply arrives later as one message tagged with the hive's name —
the turn's final answer, nothing in between. `cancel` stops a turn (its reason reaches the agent),
`retire` ends a membership, `state` and `presets` are cold reads.

Places are declared on the node's own channel (`workspaces` lists them); a path is never an address
here.
"""


def new_agents_channel(agents: DshAgents) -> MutableChannel:
    hive = agents.hive
    chan = new_channel(name=CHANNEL_NAME, description="the agents this node holds, by name.")

    # -- perception ---------------------------------------------------------------- #

    @chan.build.instruction
    def _instruction() -> str:
        return _INSTRUCTION

    @chan.build.notice
    def _notice() -> str:
        sessions = agents.list_sessions()
        running = sum(1 for item in sessions if item.state().phase.value == "running")
        places = ", ".join(space.name or space.id for space in hive.workspaces()) or "(none)"
        return (
            f"{hive.name} · {len(sessions)} agent(s), {running} running · "
            f"workspaces: {places}"
        )

    @chan.build.named_notices
    def _named_notices() -> dict[str, str | None]:
        return {
            item.name: f"{item.id} · {item.state().phase.value}"
            for item in agents.list_sessions()
        }

    @chan.build.refresh_meta
    async def _refresh_meta() -> None:
        # The declared places and templates are cached reads; this is where they are filled.
        await hive.refresh_if_stale()

    # -- lifecycle ------------------------------------------------------------------ #

    @chan.build.command(name="open", blocking=False, always_observe=True)
    async def _open(name: str, workspace: str, preset: str = "", readonly: bool = False) -> str:
        """Create an agent inside a declared workspace, under this name.

        :param name: the runtime name to address it by later. Unknown after a restart.
        :param workspace: a place declared on the owner's channel (`workspaces` lists them), by
            name, id or title. Never a path.
        :param preset: which preset composes it; empty = the deployment's default.
        :param readonly: created read-only, before its first message. Not a later patch.
        :return: `ok: <session id>`, or `failed: <reason>`.
        """
        try:
            agent = await agents.open(name, workspace=workspace, preset=preset, readonly=readonly)
        except Exception as exc:  # noqa: BLE001 — a failed open is a fact, not a crash
            return f"failed: {exc}"
        return f"ok: {agent.id}"

    @chan.build.command(name="watch", blocking=False, always_observe=True)
    async def _watch(name: str, session_id: str) -> str:
        """Take up a session that already exists, under this name.

        :param name: the runtime name to address it by from now on.
        :param session_id: the engine session id to take up.
        :return: `ok: <session id>`, or `failed: <reason>`.
        """
        try:
            agent = await agents.watch(name, session_id)
        except Exception as exc:  # noqa: BLE001
            return f"failed: {exc}"
        return f"ok: {agent.id}"

    # -- driving ---------------------------------------------------------------------- #

    @chan.build.command(name="send", blocking=False)
    async def _send(name: str, text__: str, path: str = "", mode: str = "enqueue") -> str:
        """Send one message to an agent. Answers `ok` or `failed` — the reply is a signal.

        :param name: the agent's runtime name.
        :param text__: the message body.
        :param path: a file under the project home carried along as an `@` reference; a path
            outside it fails the call rather than being dropped.
        :param mode: `enqueue` (after the current turn) · `inject` (held, does not wake) ·
            `steer` (stop what is running and answer now).
        :return: `ok`, or `failed: <reason>`.
        """
        if mode not in _MODE_VALUES:
            return f"failed: unknown mode {mode!r}; use one of {sorted(_MODE_VALUES)}"
        try:
            agent = _agent(agents, name)
            await agent.send(*_prompt_messages(text__, path, hive.project_root), mode=MessageMode(mode))
        except Exception as exc:  # noqa: BLE001 — the two-valued contract
            return f"failed: {exc}"
        return "ok"

    @chan.build.command(name="cancel", blocking=False)
    async def _cancel(name: str, reason: str = "") -> str:
        """Stop the turn in flight on one agent; the queue is kept.

        :param name: the agent's runtime name.
        :param reason: why — travels with the cancellation so a later reader can tell it from a failure.
        """
        try:
            await _agent(agents, name).cancel(reason)
        except Exception as exc:  # noqa: BLE001
            return f"failed: {exc}"
        return "ok"

    @chan.build.command(name="retire", blocking=False, always_observe=True)
    async def _retire(name: str, reason: str = "") -> str:
        """Close one agent and drop it from the roster — the end of its membership.

        :param name: the agent's runtime name.
        :param reason: why it is ending.
        """
        try:
            await agents.retire(name, reason)
        except Exception as exc:  # noqa: BLE001
            return f"failed: {exc}"
        return "ok"

    @chan.build.command(name="configure", blocking=False, always_observe=True)
    async def _configure(name: str, text__: str) -> str:
        """Patch one agent's runtime settings. The body is a JSON object.

        :param name: the agent's runtime name.
        :param text__: a JSON object; an omitted field keeps its current value.
            {
              "llm": {
                "provider": "deepseek-official",
                "model": "deepseek-flash",
                "reasoningEffort": "off" | "low" | "high" | "max"
              },
              "sandbox": "read-only" | "workspace-write",
              "approval": "ask" | "never"
            }
        :return: what is now in force, or `failed: <reason>`.
        """
        try:
            patch = json.loads(text__ or "{}")
        except Exception as exc:  # noqa: BLE001
            return f"failed: the body is not JSON: {exc}"
        if not isinstance(patch, dict):
            return "failed: the body must be a JSON object"
        try:
            return await _agent(agents, name).configure(patch)
        except Exception as exc:  # noqa: BLE001
            return f"failed: {exc}"

    # -- cold reads --------------------------------------------------------------------- #

    @chan.build.command(name="state", blocking=False, always_observe=True)
    async def _state(name: str = "") -> str:
        """Cold read: one agent's state, or every held agent. Never on the notice.

        :param name: the agent's runtime name; empty = list everything held.
        """
        try:
            if not name:
                sessions = agents.list_sessions()
                if not sessions:
                    return "(no agents)"
                return "\n".join(_row(item.state()) for item in sessions)
            return _row(_agent(agents, name).state())
        except Exception as exc:  # noqa: BLE001
            return f"failed: {exc}"

    @chan.build.command(name="presets", blocking=False, always_observe=True)
    async def _presets() -> str:
        """Cold read: the presets an agent may be composed from.

        The composition is the engine's; this only lists what the deployment declared.
        """
        items = hive.presets()
        if not items:
            return "(no presets)"
        return "\n".join(_preset_row(item) for item in items)

    return chan


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _agent(agents: Agents, name: str) -> AgentSession:
    return agents.get(name)


def workspace_row(item: Workspace) -> str:
    """One place, as a ghost reads it: what to call it, its id, and the directory it bounds."""
    name = item.name or item.title or item.id
    return f"{name} · {item.id} · {item.path}"


def _preset_row(item: "AgentPreset") -> str:
    head = f"{item.id} · {item.name}" if item.name else item.id
    return f"{head} — {item.description}" if item.description else head


def _row(item: AgentState) -> str:
    bits = [
        item.name,
        item.id,
        item.phase.value,
        f"preset={item.preset or '-'}",
        item.cwd or "-",
    ]
    sandbox = (item.config or {}).get("sandbox")
    if sandbox:
        # The permission boundary is part of a state read: "what may this agent touch" is not
        # answerable from cwd alone when the session was created read-only.
        bits.append(f"sandbox={sandbox}")
    return " · ".join(bits)


def _prompt_messages(text: str, path: str, project_root: str) -> list[Message]:
    """Turn `text` (+ an optional project-home file) into messages the agent can read.

    The `@path` form is the message protocol's own file reference: it resolves to a message whose
    contents are the file's, so the agent receives the file rather than its name.
    """
    if path:
        target = Path(path)
        if not target.is_absolute():
            root = Path(project_root) if project_root else Path.cwd()
            target = (root / path).resolve()
        else:
            target = target.resolve()
        if project_root:
            try:
                target.relative_to(Path(project_root).resolve())
            except ValueError:
                raise PermissionError(f"path {target} is outside the project home {project_root}") from None
        text = f"@{target}\n{text}"
    base = Path(project_root) if project_root else Path.cwd()
    # A file reference the protocol refuses is a failed send, not a text-only one: silently
    # downgrading would deliver something the caller believes carried a file.
    #
    # Not `Prompt`: that class is a *usage* — a handle bound to the process cwd, and it refuses
    # any base_dir outside it (message/prompt.py). Here the anchor is the project home, which is
    # a fact of the hive, not of the process: pass it explicitly.
    return message_from_prompt(text, base_dir=base)
