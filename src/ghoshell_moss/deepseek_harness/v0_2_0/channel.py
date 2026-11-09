"""The dsh hive's channel — one place to drive every session, one fragment per session.

Shape, and why it is shaped this way:

- **A single control plane.** Every verb lives on this one channel and takes an address. Splitting
  verbs across child channels would make the model pick where a verb lives before it can use it;
  with one flat set of verbs the only thing to know is the address.
- **Virtual children are the sessions themselves.** Each held session appears as a channel named
  after its runtime alias (or a short id), carrying nothing but its own identity and state — the
  addressable half of "who exists". Control stays compressed into the parent.
- **Named notices are the perception face.** One fragment per session showing only the session id
  and whether it is running: the only two facts a driver needs on every turn. Anything more
  detailed is a cold read, pulled when it matters.
- **Cold reads stay cold.** ``state`` and ``events`` are commands, never notice material — a
  notice that grows with every session is a notice nobody reads.
"""

from __future__ import annotations

import time
from pathlib import Path

from ghoshell_moss.agents.blueprint.agent import (
    AgentPhase,
    AgentSessionEvent,
    AgentState,
    MessageMode,
)
from ghoshell_moss.core.blueprint.channel_builder import (
    MutableChannel,
    new_channel,
)
from ghoshell_moss.message.prompt import Prompt

from .hive import DshHive

__all__ = ["new_dsh_hive_channel"]

CHANNEL_NAME = "dsh"

_MODE_VALUES = {mode.value for mode in MessageMode}

_INSTRUCTION = """\
A dsh hive: every session it holds, driven from one place.

Address a session by the runtime **name** you gave it, or by its session id — both work.
Sessions are runtime only: nothing here survives the process, so coming back to one is an explicit
`attach(session_id=...)`.

- `send(to, text)` — fire and forget. It returns only `ok` or `failed`; the answer arrives later
  as a signal (and in `events`).
- `open(...)` — create a session from scratch. `attach(session_id=...)` — take up one that already
  exists. Both take a `name`, which lives only at runtime.
- `state(to="")` / `events(since=0)` — cold reads for debugging and observation. Never on the notice.
"""


def new_dsh_hive_channel(hive: DshHive) -> MutableChannel:
    """Build the channel for one hive. Every closure below closes over the same hive."""
    chan = new_channel(name=CHANNEL_NAME, description="dsh hive — sessions, opened and driven here.")
    started = time.monotonic()

    # -- perception ---------------------------------------------------------------- #

    @chan.build.instruction
    def _instruction() -> str:
        return _INSTRUCTION

    @chan.build.notice
    def _notice() -> str:
        sessions = hive.sessions()
        running = sum(1 for item in sessions if item.phase is AgentPhase.running)
        return (
            f"{hive.engine_name}@{hive.version} · hive {hive.name} · "
            f"{len(sessions)} session(s), {running} running · "
            f"{len(hive.presets())} preset(s) · {len(hive.workspaces())} workspace(s)"
        )

    @chan.build.named_notices
    def _named_notices() -> dict[str, str | None]:
        notices: dict[str, str | None] = {}
        for item in hive.sessions():
            notices[_fragment_name(hive, item)] = f"{item.id} · {item.phase.value}"
        return notices

    @chan.build.refresh_meta
    async def _refresh_meta() -> None:
        # The two cached read faces are filled here, and only when they may have moved: a meta
        # refresh happens often, and two RPCs per turn would be paid for nothing.
        await hive.refresh_if_stale()

    @chan.build.virtual_children
    def _virtual_children() -> dict[str, MutableChannel]:
        children: dict[str, MutableChannel] = {}
        for item in hive.sessions():
            children[_fragment_name(hive, item)] = _session_channel(item)
        return children

    # -- lifecycle ------------------------------------------------------------------ #

    @chan.build.command(name="open", blocking=False, always_observe=True)
    async def _open(name: str = "", preset: str = "", cwd: str = "", title: str = "") -> str:
        """Create a session from scratch with the engine's own default preset.

        :param name: a runtime alias to address it by later. Runtime only — it dies with the process.
        :param preset: which preset composes the agent (see the hive's preset list); empty = the deployment's default.
        :param cwd: the directory it may act in — absolute; empty = the engine's default.
        :param title: a human-facing title, set on the engine side.
        :return: a receipt naming the new session id.
        """
        try:
            agent = await hive.open(name=name, preset=preset, cwd=cwd, title=title)
        except Exception as exc:  # noqa: BLE001 — a failed open is a fact, not a crash
            return f"failed: {exc}"
        return f"opened {agent.id}" + (f" as {name}" if name else "")

    @chan.build.command(name="attach", blocking=False, always_observe=True)
    async def _attach(session_id: str, name: str = "") -> str:
        """Take up a session that already exists — the only way back into one.

        :param session_id: the engine session id, as reported elsewhere.
        :param name: a runtime alias to address it by later.
        :return: a receipt confirming the session is now held.
        """
        try:
            agent = await hive.resume(session_id, name=name)
        except Exception as exc:  # noqa: BLE001
            return f"failed: {exc}"
        return f"attached {agent.id}" + (f" as {name}" if name else "")

    # -- driving ---------------------------------------------------------------------- #

    @chan.build.command(name="send", blocking=False)
    async def _send(to: str, text__: str, path: str = "", mode: str = "enqueue") -> str:
        """Send one message to a session. Returns only `ok` or `failed` — the answer is a signal.

        :param to: the session's runtime name or its session id.
        :param text__: the message body.
        :param path: a file under the project home to carry along, written as an `@` reference.
        :param mode: `enqueue` (run after the current turn) · `inject` (held, does not wake) ·
            `steer` (stop what is running and answer now).
        """
        if mode not in _MODE_VALUES:
            return f"failed: unknown mode {mode!r}; use one of {sorted(_MODE_VALUES)}"
        try:
            messages = _prompt_messages(text__, path, hive.project_root)
            await hive.send(to, *messages, mode=MessageMode(mode))
        except Exception as exc:  # noqa: BLE001 — the two-valued contract
            return f"failed: {exc}"
        return "ok"

    @chan.build.command(name="cancel", blocking=False)
    async def _cancel(to: str, reason: str = "") -> str:
        """Stop the turn in flight on one session; the queue is kept.

        :param to: the session's runtime name or its session id.
        :param reason: why — travels with the cancellation so a later reader can tell it from a failure.
        """
        try:
            await hive.cancel(to, reason)
        except Exception as exc:  # noqa: BLE001
            return f"failed: {exc}"
        return "ok"

    @chan.build.command(name="retire", blocking=False, always_observe=True)
    async def _retire(to: str) -> str:
        """Close one session and drop it: the end of its membership.

        :param to: the session's runtime name or its session id.
        """
        try:
            await hive.retire(to)
        except Exception as exc:  # noqa: BLE001
            return f"failed: {exc}"
        return "ok"

    # -- cold reads --------------------------------------------------------------------- #

    @chan.build.command(name="state", blocking=False, always_observe=True)
    async def _state(to: str = "") -> str:
        """Cold read: one session's state, or every held session. Never on the notice.

        :param to: a name or session id; empty = list every session this hive holds.
        """
        if not to:
            items = hive.sessions()
            if not items:
                return "(no sessions)"
            return "\n".join(_row(item) for item in items)
        item = hive.session(to)
        if item is None:
            return f"failed: no session {to!r}"
        return _row(item.state())

    @chan.build.command(name="events", blocking=False, always_observe=True)
    async def _events(since: int = 0) -> str:
        """Cold read: the event tail across every session, newest last.

        :param since: the cursor returned by the previous call; 0 starts from the oldest event kept.
        :return: one line per event, then the cursor to pass next time.
        """
        items, cursor = hive.events(since)
        if not items:
            return f"(no events) cursor={cursor}"
        lines = [
            f"{item.session_id[:8]} · {item.phase.value} · {item.event_name} · "
            f"{(item.description or '').strip()[:120]}"
            for item in items
        ]
        lines.append(f"cursor={cursor}")
        return "\n".join(lines)

    return chan


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _fragment_name(hive: DshHive, item: AgentState) -> str:
    """A stable, tag-safe key for one session — the alias when it has one, else a short id."""
    alias = hive.alias_of(item.id)
    return _tag_safe(alias) if alias else _tag_safe(item.id[:12])


def _tag_safe(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in value) or "session"


def _row(item: AgentState) -> str:
    return f"{item.id} · {item.phase.value} · preset={item.preset or '-'} · {item.title or '-'}"


def _session_channel(item: AgentState) -> MutableChannel:
    """The addressable half: one child channel per session, carrying identity and state only."""
    child = new_channel(name=_tag_safe(item.id[:12]), description=f"dsh session {item.id}")

    @child.build.notice
    def _notice() -> str:
        return f"{item.id} · {item.phase.value}"

    return child


def _prompt_messages(text: str, path: str, project_root: str):
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
    try:
        return Prompt(text, base_dir=base).messages()
    except PermissionError:
        # The prompt protocol confines itself to cwd; when the project home is outside it, send
        # the text as-is rather than losing the message.
        from ghoshell_moss.message import Message

        return [Message.new().with_content(text)]
