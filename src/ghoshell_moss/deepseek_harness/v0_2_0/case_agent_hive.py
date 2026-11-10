"""The dsh case hive — cases on a memento store, carried by dsh sessions.

This is the first implementation of ``agents/blueprint/case.py`` on top of the dsh engine.
It does **not** reuse ``DshAgents``: a case carrier has a different life (it is driven by a
case loop, writes its own commits, never raises a dsh signal), and sharing the roster would
make one of the two meanings win. It reuses the carrier class ``DshAgentSession`` — that is
the adapter to one engine session, not the collection.

## Layout

A hive owns one directory::

    <cell>/.hive/
        .dsh/          # the engine home (owned by dsh's launcher)
        cases/         # the memento store: ``branches/<id>/`` is one case

The memento store *is* the case collection: the roster is read from it, never kept in
memory. A case directory holds ``CASE.md`` (prompt face), ``case.meta.json`` (control face)
and ``tmp/`` (the durable mirror of the carrier's scratch directory).

## The verb set

``draft`` writes the documents, ``bind`` gives the case a carrier, ``autopilot`` dispatches
the loop. ``pause`` stops it from outside; a violation or an exhausted budget stops it from
inside. Both write a ``run.end`` commit; only ``pause`` leaves the carrier attached.

## Commits

| commit | metatype | metadata |
|---|---|---|
| a finished step | ``""`` | ``operation``, ``session_ref``, ``usage`` |
| a carrier was bound | ``init`` | ``session_ref``, ``preset``, ``workspace``, ``tmp_dir`` |
| a run started | ``run.start`` | ``run`` |
| a run ended | ``run.end`` | ``run`` |
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import frontmatter

from ghoshell_dsh.blueprint.anchor import DshSessionAnchor
from ghoshell_dsh.v0_2_0.types.sessions import SessionCreateParams

from ghoshell_moss.agents.blueprint.agent import (
    AgentPhase,
    AgentPreset,
    AgentSession,
    AgentSessionEvent,
    MessageMode,
)
from ghoshell_moss.agents.blueprint.case import (
    AutopilotRequest,
    BindRequest,
    Case,
    CaseAgent,
    CaseAgentManager,
    CaseBrief,
    CaseCommit,
    CaseDefinition,
    CaseMeta,
    CaseProtocol,
    CaseState,
    CommitCallback,
    Next,
    Operation,
    ProtocolTemplate,
    Run,
    Usage,
)
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.memento import Branch, BranchAddr, CommitView, Memento, new_local_memento
from ghoshell_moss.message import Message

from .agent import DshAgentSession
from .hive import DshHive

__all__ = ["DshCase", "DshCaseAgent", "DshCaseAgentHive", "DshCaseAgentManager", "DshCaseCommit"]

CASE_DOCUMENT = "CASE.md"
CASE_META = "case.meta.json"
CASE_TMP = "tmp"
PROTOCOL_DIR = "protocols"

_CONTINUE = "continue: take the next step."
"""What the loop says to wake the carrier for its next step. The protocol, not the task, is
what makes this sentence stable: a continuation carries no new information."""

_SNAPSHOT_WAIT = 5.0
"""How long ``bind`` waits for the engine's follow snapshot before reading the cwd back."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _addr(branch: Branch) -> BranchAddr:
    """The branch's durable address: identity, display name and ordinal."""
    return BranchAddr(
        branch_id=branch.ref.branch_id,
        branch_name=branch.ref.name,
        branch_index=branch.index,
    )


# --------------------------------------------------------------------------- #
# Documents
# --------------------------------------------------------------------------- #


def _write_brief(path: Path, brief: CaseBrief, owner: str, created: datetime) -> None:
    """Write ``CASE.md``: the prompt face, frontmatter plus the brief as body."""
    metadata = {
        "name": brief.name,
        "description": brief.description,
        "owner": owner,
        "created": created.isoformat(),
        "protocol": brief.protocol.model_dump(mode="json"),
    }
    post = frontmatter.Post(brief.content, **metadata)
    path.write_text(frontmatter.dumps(post), encoding="utf-8")


def _read_brief(path: Path) -> CaseBrief:
    post = frontmatter.loads(path.read_text(encoding="utf-8"))
    protocol = post.metadata.get("protocol")
    return CaseBrief(
        name=str(post.metadata.get("name") or ""),
        description=str(post.metadata.get("description") or ""),
        content=post.content,
        protocol=CaseProtocol.model_validate(protocol or {}),
    )


def _write_meta(path: Path, meta: CaseMeta) -> None:
    path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")


def _read_meta(path: Path) -> CaseMeta:
    return CaseMeta.model_validate_json(path.read_text(encoding="utf-8"))


def _render_priming(brief: CaseBrief, manifest: str) -> str:
    """The text a carrier receives before its first prompt — parts 2–5 of the chain.

    Part 1 (the preset's own prompt) belongs to the engine and is already in the session.
    """
    return "\n".join(
        [
            "# Delegation",
            "",
            "## The case",
            "",
            (brief.content.strip() or brief.description.strip() or "(no brief written)"),
            "",
            "## Protocol",
            "",
            brief.protocol.instruction(),
            "",
            "## Files in your scratch directory",
            "",
            manifest,
        ]
    )


def _manifest(tmp_dir: Path) -> str:
    if not tmp_dir.is_dir():
        return "(empty)"
    names = sorted(
        item.relative_to(tmp_dir).as_posix() for item in tmp_dir.rglob("*") if item.is_file()
    )
    return "\n".join(f"- {name}" for name in names) if names else "(empty)"


def _mirror(source: Path, target: Path) -> None:
    """Copy the files under ``source`` into ``target``, overwriting; never delete."""
    if not source.is_dir():
        return
    target.mkdir(parents=True, exist_ok=True)
    for item in source.rglob("*"):
        relative = item.relative_to(source)
        destination = target / relative
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, destination)


# --------------------------------------------------------------------------- #
# The op line
# --------------------------------------------------------------------------- #

_STEP_LINE = re.compile(
    r"^(?P<op>[A-Za-z][A-Za-z0-9_-]*)(?:\("
    r"(?P<target>[^)]*)\))?\s*[:：]\s*(?P<title>.*)$"
)


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_operation(text: str, protocol: CaseProtocol) -> Operation:
    """Read one reply as one step. A violation is a step that cannot be read — never an exception.

    The word must come from the case's own protocol: an unknown word is a violation exactly
    like an unreadable line, because the loop cannot know what follows it.
    """
    message = _strip_fence(text)
    first, _, _rest = message.partition("\n")
    match = _STEP_LINE.match(first.strip())
    if match is None:
        return Operation(op="", title=first.strip(), next=Next.BREAK, violation=True)
    word = match.group("op")
    target = (match.group("target") or "").strip()
    title = match.group("title").strip()
    operator = protocol.op(word)
    if operator is None:
        return Operation(op=word, target=target, title=title, next=Next.BREAK, violation=True)
    return Operation(op=word, target=target, title=title, next=operator.next, violation=False)


def _no_answer_stop(reason: str) -> str:
    """Why a turn that produced no step ended — the reason, not a guess."""
    if reason in ("error", "aborted", "blocked", "max-tokens", "interrupted"):
        return reason
    return "no_answer"


def _tokens(payload: dict[str, Any]) -> int:
    """One step's token cost, in the engine's own accounting.

    ``TokenUsage`` keeps the counts disjoint (uncached input, output, cache read/write), so the
    standard total is their sum; ``totalTokens`` is used when the adapter reports it.
    """
    message = payload.get("message")
    if not isinstance(message, dict):
        return 0
    usage = message.get("usage")
    if not isinstance(usage, dict):
        return 0
    total = usage.get("totalTokens")
    if isinstance(total, int):
        return total
    parts = ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens")
    return sum(int(usage.get(key) or 0) for key in parts)


# --------------------------------------------------------------------------- #
# The trace
# --------------------------------------------------------------------------- #


class DshCaseCommit(CaseCommit[str]):
    """One commit of a case's branch, read as a step or a system record."""

    def __init__(self, view: CommitView) -> None:
        self._view = view
        self._operation = Operation.from_commit_ref(view.ref)

    @property
    def view(self) -> CommitView:
        return self._view

    @property
    def operation(self) -> Operation | None:
        return self._operation

    @property
    def session_ref(self) -> str | None:
        value = self._view.ref.metadata.get("session_ref")
        return value if isinstance(value, str) and value else None


# --------------------------------------------------------------------------- #
# The durable half
# --------------------------------------------------------------------------- #


class DshCase(Case):
    """A case as its branch directory holds it."""

    def __init__(self, branch: Branch) -> None:
        self._branch = branch

    @property
    def name(self) -> str:
        """The branch's name — the address callers use for this case."""
        return self._branch.ref.name

    @property
    def directory(self) -> Path:
        return self._branch.path

    @property
    def document(self) -> Path:
        return self._branch.path / CASE_DOCUMENT

    @property
    def tmp(self) -> Path:
        return self._branch.path / CASE_TMP

    @property
    def brief(self) -> CaseBrief:
        return _read_brief(self.document)

    @property
    def meta(self) -> CaseMeta:
        return _read_meta(self._branch.path / CASE_META)

    @property
    def branch(self) -> Branch:
        return self._branch

    def commits(self, limit: int = 50) -> list[CaseCommit]:
        view = self._branch.view(n=limit)
        return [DshCaseCommit(item) for item in view.latest]


# --------------------------------------------------------------------------- #
# The living half
# --------------------------------------------------------------------------- #


class _Turn:
    """One finished turn, as the loop receives it.

    ``reason`` is the engine's own turn-end reason (``completed`` / ``error`` / ``aborted`` …).
    It is kept because "the turn ended without an answer" and "the turn failed" look identical
    from the text alone, and only one of them is a model output problem.
    """

    __slots__ = ("text", "tokens", "reason")

    def __init__(self, text: str, tokens: int, reason: str = "") -> None:
        self.text = text
        self.tokens = tokens
        self.reason = reason


class DshCaseAgent(CaseAgent[str]):
    """One case's carrier and the observation that feeds its loop."""

    def __init__(
        self, name: str, case: DshCase, carrier: DshAgentSession, scratch: Path
    ) -> None:
        self._name = name
        self._case = case
        self._carrier = carrier
        self._scratch = scratch
        self._hooks: list[CommitCallback] = []
        self._queue: asyncio.Queue[_Turn] = asyncio.Queue()
        self._pending_text = ""
        self._pending_tokens = 0
        self._running = False
        self._run: Run | None = None
        self._task: asyncio.Task[None] | None = None
        self._disposers = [
            carrier.on_turn_end(self._on_turn_end),
            carrier.on_session_event(self._on_event),
        ]

    # -- the observation ----------------------------------------------------------------- #

    def _on_turn_end(self, text: str, event: AgentSessionEvent) -> None:
        self._pending_text = text

    def _on_event(self, event: AgentSessionEvent) -> None:
        if event.event_name == "assistant/message":
            self._pending_tokens += _tokens(event.payload)
            return
        if event.phase is not AgentPhase.idle:
            return
        # A turn closed. Whether it had an answer or not, the loop must hear about it: a turn
        # that produced no answer is still a turn that is over, and waiting for one that never
        # comes is how a loop hangs. The idle event carries the turn's reason in its description.
        reason = event.description if event.event_name == "turn/end" else ""
        turn = _Turn(self._pending_text, self._pending_tokens, reason)
        self._pending_text = ""
        self._pending_tokens = 0
        self._queue.put_nowait(turn)

    # -- CaseAgent ------------------------------------------------------------------------- #

    @property
    def name(self) -> str:
        return self._name

    @property
    def case(self) -> Case:
        return self._case

    @property
    def session(self) -> AgentSession | None:
        return self._carrier

    @property
    def carrier(self) -> DshAgentSession:
        return self._carrier

    def is_running(self) -> bool:
        return self._running

    def on_commit(self, callback: CommitCallback) -> Callable[[], None]:
        self._hooks.append(callback)

        def _dispose() -> None:
            with contextlib.suppress(ValueError):
                self._hooks.remove(callback)

        return _dispose

    # -- internals -------------------------------------------------------------------------- #

    def _emit(self, commit: DshCaseCommit) -> None:
        for hook in list(self._hooks):
            try:
                result = hook(commit)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception:  # noqa: BLE001 — a consumer's bug never touches the trace
                pass

    def _drain(self) -> None:
        while not self._queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()

    async def _close(self) -> None:
        self._running = False
        for dispose in self._disposers:
            with contextlib.suppress(Exception):
                dispose()
        self._disposers.clear()
        self._hooks.clear()
        with contextlib.suppress(Exception):
            await self._carrier.close()


# --------------------------------------------------------------------------- #
# The collection
# --------------------------------------------------------------------------- #


class DshCaseAgentManager(CaseAgentManager[str]):
    """Cases on one memento store, carried by dsh sessions."""

    def __init__(self, hive: DshHive, store: Path) -> None:
        self._hive = hive
        self._store = store
        self._memento: Memento = new_local_memento(store)
        self._cases: dict[str, DshCase] = {}
        self._agents: dict[str, DshCaseAgent] = {}

    # -- the store -------------------------------------------------------------------------- #

    @property
    def memento(self) -> Memento:
        return self._memento

    def _load(self, name: str) -> DshCase:
        case = self._cases.get(name)
        if case is not None:
            return case
        branch = self._memento.get_branch(name)
        if branch is None:
            raise NameError(f"case {name!r} does not exist")
        case = DshCase(branch)
        self._cases[name] = case
        return case

    def list_cases(self, state: CaseState | None = None) -> list[Case]:
        cases: list[Case] = []
        for ref in self._memento.list_branches():
            case = self._load(ref.name)
            if state is not None and case.meta.state is not state:
                continue
            cases.append(case)
        return cases

    def get(self, case_name: str) -> Case:
        return self._load(case_name)

    # -- protocols -------------------------------------------------------------------------- #

    def _protocol_dir(self) -> Path:
        return self._store / PROTOCOL_DIR

    async def protocols(self) -> list[ProtocolTemplate]:
        directory = self._protocol_dir()
        if not directory.is_dir():
            return []
        return [
            ProtocolTemplate.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(directory.glob("*.json"))
        ]

    async def save_protocol(self, template: ProtocolTemplate) -> None:
        directory = self._protocol_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{template.name}.json").write_text(
            template.model_dump_json(indent=2), encoding="utf-8"
        )

    def _resolve_protocol(self, value: CaseProtocol | str) -> CaseProtocol:
        if isinstance(value, CaseProtocol):
            return value
        path = self._protocol_dir() / f"{value}.json"
        if not path.is_file():
            raise KeyError(f"protocol {value!r} is not stored in this hive")
        return ProtocolTemplate.model_validate_json(path.read_text(encoding="utf-8"))

    # -- documents -------------------------------------------------------------------------- #

    def _set_state(self, case: DshCase, state: CaseState, reason: str = "") -> CaseMeta:
        meta = case.meta
        meta = meta.model_copy(update={"state": state, "reason": reason, "updated": _now()})
        _write_meta(case.directory / CASE_META, meta)
        return meta

    async def draft(self, definition: CaseDefinition) -> Case:
        if self._memento.get_branch(definition.name) is not None:
            raise NameError(f"case {definition.name!r} already exists")
        protocol = self._resolve_protocol(definition.protocol)
        created = _now()
        branch = self._memento.create_branch(definition.name, definition.description, metatype="case")
        brief = CaseBrief(
            name=definition.name,
            description=definition.description,
            content=definition.content,
            protocol=protocol,
        )
        _write_brief(branch.path / CASE_DOCUMENT, brief, definition.owner, created)
        meta = CaseMeta(
            state=CaseState.draft,
            owner=definition.owner,
            address=_addr(branch),
            created=created,
            updated=created,
        )
        _write_meta(branch.path / CASE_META, meta)
        (branch.path / CASE_TMP).mkdir(exist_ok=True)
        case = DshCase(branch)
        self._cases[definition.name] = case
        return case

    # -- carriers ---------------------------------------------------------------------------- #

    def _preset_snapshot(self, preset_id: str) -> AgentPreset:
        if not preset_id:
            return AgentPreset(id="")
        try:
            return self._hive.preset(preset_id).model_copy(deep=True)
        except KeyError:
            return AgentPreset(id=preset_id)

    def _last_init(self, case: DshCase) -> dict[str, Any]:
        for ref in reversed(case.branch.commits()):
            if ref.metatype == "init" and isinstance(ref.metadata, dict):
                return ref.metadata
        return {}

    async def bind(self, request: BindRequest) -> CaseAgent[str]:
        case = self._load(request.case_name)
        previous = self._agents.pop(request.case_name, None)
        if previous is not None:
            await self._retire(previous, reason="rebind")

        init = self._last_init(case)
        workspace = request.workspace or str(init.get("workspace") or "")
        if not workspace:
            raise ValueError(
                f"case {case.name!r} was never bound: pass `workspace` to bind it the first time"
            )
        preset = request.preset or str(init.get("preset") or "")
        config = request.config if request.config is not None else init.get("config")
        tmp_dir = request.tmp_dir or str(init.get("tmp_dir") or "")

        place = self._hive.workspace(workspace)
        create = SessionCreateParams(agentPreset=preset or None)
        if place.id:
            create.workspaceId = place.id
        else:
            create.cwd = place.path or None
        value = await self._hive.connection.client.session_create(create)
        session_id = str(value.sessionId)

        session = self._hive.connection.create_session(session_id, self._hive.logger)
        await session.__aenter__()
        await session.wait_follow_snapshot(timeout=_SNAPSHOT_WAIT)
        cwd = await session.cwd() or place.path
        carrier = DshAgentSession(
            session,
            hive=self._hive.name,
            engine=f"{self._hive.engine_name}@{self._hive.version}",
            name=case.name,
            preset=self._preset_snapshot(preset),
            cwd=cwd,
            workspace_id=place.id,
        )
        carrier.prime(preset=preset, cwd=cwd, workspace_id=place.id, readonly=request.readonly)
        if request.readonly:
            await session.set_policy(sandbox="read-only")
        if isinstance(config, dict) and config:
            await carrier.configure(config)

        scratch = Path(tmp_dir) if tmp_dir else Path(cwd) / ".case-tmp" / case.name
        scratch.mkdir(parents=True, exist_ok=True)
        _mirror(case.tmp, scratch)

        brief = case.brief
        priming = _render_priming(brief, _manifest(scratch))
        await carrier.send(
            Message.new(tag="delegation").with_content(priming), mode=MessageMode.inject
        )

        case.branch.commit(
            message=f"init: {preset or 'default'} · {workspace}",
            metatype="init",
            metadata={
                "session_ref": session_id,
                "preset": preset,
                "workspace": workspace,
                "tmp_dir": str(scratch),
                "config": config if isinstance(config, dict) else None,
            },
        )
        agent = DshCaseAgent(case.name, case, carrier, scratch)
        self._agents[case.name] = agent
        self._set_state(case, CaseState.open, "")
        return agent

    async def _retire(self, agent: DshCaseAgent, reason: str = "") -> None:
        await self._stop_loop(agent, reason or "closed")
        with contextlib.suppress(Exception):
            await agent.carrier.cancel(reason)
        await agent._close()
        self._agents.pop(agent.name, None)

    async def _stop_loop(self, agent: DshCaseAgent, stop: str) -> None:
        """Stop the driving task and let its own ``finally`` close the run record."""
        task = agent._task
        if task is None or task.done():
            agent._running = False
            return
        if agent._run is not None and not agent._run.stop:
            agent._run.stop = stop
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    # -- the loop ---------------------------------------------------------------------------- #

    async def _write_step(self, agent: DshCaseAgent, turn: _Turn) -> DshCaseCommit:
        case = agent.case
        operation = parse_operation(turn.text, case.brief.protocol)
        # The engine's own coordinate at the turn's end: the anchor a later `chat` asks from.
        # Without it, "ask this step" would silently mean "ask the session's current end".
        seq: int | None = None
        with contextlib.suppress(Exception):
            seq = (await agent.carrier.engine_session.anchor()).seq
        ref = case.branch.commit(
            message=turn.text,
            metadata={
                "operation": operation.model_dump(mode="json"),
                "session_ref": agent.carrier.session_id,
                "session_seq": seq,
                "usage": Usage(tokens=turn.tokens).model_dump(mode="json"),
            },
        )
        view = case.branch.get_commit(ref.seq)
        if view is None:  # pragma: no cover — a commit that just landed must be readable
            raise RuntimeError(f"case {case.name!r}: commit {ref.seq} is not readable")
        return DshCaseCommit(view)

    async def _loop(self, agent: DshCaseAgent, run: Run, kickoff: str) -> None:
        case = agent.case
        deadline = (
            time.monotonic() + run.budget.seconds if run.budget.seconds is not None else None
        )
        used = 0
        try:
            await agent.carrier.send(
                Message.new(tag="kickoff").with_content(kickoff), mode=MessageMode.enqueue
            )
            while True:
                timeout: float | None = None
                if deadline is not None:
                    timeout = deadline - time.monotonic()
                    if timeout <= 0:
                        run.stop = "exhausted"
                        break
                try:
                    turn = await asyncio.wait_for(agent._queue.get(), timeout=timeout)
                except asyncio.TimeoutError:
                    run.stop = "exhausted"
                    break

                if not turn.text.strip():
                    # A turn that ended without an answer is not a step: nothing is written,
                    # and the loop stops rather than waiting for a turn that already happened.
                    run.stop = _no_answer_stop(turn.reason)
                    break

                commit = await self._write_step(agent, turn)
                used += turn.tokens
                _mirror(agent._scratch, case.tmp)
                agent._emit(commit)

                operation = commit.operation
                if operation is None or operation.violation:
                    run.stop = "violation"
                    break
                if operation.next is Next.BREAK:
                    run.stop = operation.op or "break"
                    break
                if run.budget.tokens is not None and used >= run.budget.tokens:
                    run.stop = "exhausted"
                    break
                await agent.carrier.send(
                    Message.new(tag="continue").with_content(_CONTINUE), mode=MessageMode.enqueue
                )
        except asyncio.CancelledError:
            if not run.stop:
                run.stop = "cancelled"
            raise
        except Exception as exc:  # noqa: BLE001 — the loop must close its run and say why
            run.stop = f"error: {exc}"
        finally:
            run.ended = _now()
            with contextlib.suppress(Exception):
                case.branch.commit(
                    message=f"run end: {run.stop or 'ended'}",
                    metatype="run.end",
                    metadata={"run": run.model_dump(mode="json")},
                )
            agent._running = False
            agent._run = None
            agent._task = None

    async def autopilot(self, request: AutopilotRequest) -> CaseAgent[str]:
        agent = self._agents.get(request.case_name)
        if agent is None:
            raise RuntimeError(
                f"case {request.case_name!r} has no carrier — call `bind` first"
            )
        if agent.is_running():
            raise RuntimeError(f"case {request.case_name!r} is already running")
        kickoff = f"kickoff: {agent.case.brief.name} — begin; one step per reply."
        run = Run(started=_now(), budget=request.budget)
        agent.case.branch.commit(
            message=kickoff, metatype="run.start", metadata={"run": run.model_dump(mode="json")}
        )
        agent._drain()
        agent._run = run
        agent._running = True
        agent._task = asyncio.create_task(self._loop(agent, run, kickoff))
        return agent

    # -- stops ------------------------------------------------------------------------------- #

    async def pause(self, case_name: str, reason: str = "") -> None:
        agent = self._agents.get(case_name)
        if agent is None:
            raise NameError(f"case {case_name!r} has no carrier to pause")
        await self._stop_loop(agent, "paused")
        with contextlib.suppress(Exception):
            await agent.carrier.cancel(reason)

    async def resolve(self, case_name: str, reason: str = "") -> None:
        case = self._load(case_name)
        agent = self._agents.get(case_name)
        if agent is not None:
            await self._retire(agent, reason=reason or "resolved")
        self._set_state(case, CaseState.resolved, reason)

    async def close(self, case_name: str, reason: str = "") -> None:
        case = self._load(case_name)
        agent = self._agents.get(case_name)
        if agent is not None:
            await self._retire(agent, reason=reason or "closed")
        self._set_state(case, CaseState.cancelled, reason)
        with contextlib.suppress(Exception):
            self._memento.archive_branch(case_name)
        self._cases.pop(case_name, None)

    # -- reading ----------------------------------------------------------------------------- #

    def read(self, case_name: str, commit_coord: str = "", limit: int = 10) -> list[CaseCommit]:
        case = self._load(case_name)
        if commit_coord:
            view = self._memento.resolve_commit(commit_coord)
            return [DshCaseCommit(view)] if view is not None else []
        return [DshCaseCommit(item) for item in case.branch.view(n=limit).latest]

    async def chat(self, case_name: str, commit_coord: str, question: str) -> str:
        """Ask one step a question with the context that produced it.

        The commit carries the session that produced it, so the question is answered on that
        session's own context — same model, same history. A live carrier is asked directly; a
        carrier that is gone needs the plugin's resume-and-slice path, and says so instead of
        answering from the wrong place.
        """
        view = self._memento.resolve_commit(commit_coord)
        if view is None:
            raise NameError(f"no commit at {commit_coord!r}")
        session_ref = view.ref.metadata.get("session_ref")
        if not isinstance(session_ref, str) or not session_ref:
            raise ValueError(f"commit {commit_coord!r} was not produced by a carrier")
        seq = view.ref.metadata.get("session_seq")
        anchor = DshSessionAnchor(session_id=session_ref, seq=seq if isinstance(seq, int) else None)
        agent = self._agents.get(case_name)
        if agent is not None and agent.carrier.session_id == session_ref:
            reply = await agent.carrier.engine_session.chat_off_record(question, anchor=anchor)
            return reply.text
        # No live carrier: the plugin slices the stored log at the anchored seq and asks there,
        # so a cold case is as answerable as a live one.
        session = self._hive.connection.create_session(session_ref, self._hive.logger)
        try:
            reply = await session.chat_off_record(question, anchor=anchor)
        finally:
            with contextlib.suppress(Exception):
                await session.close()
        return reply.text

    # -- presentation ------------------------------------------------------------------------ #

    def as_channel(self) -> Channel | None:
        from .case_channel import new_cases_channel

        return new_cases_channel(self)


# --------------------------------------------------------------------------- #
# The place
# --------------------------------------------------------------------------- #


class DshCaseAgentHive(DshHive):
    """A dsh hive whose cases live in ``<root>/.hive/cases``.

    The engine home and the case store are one directory: ``.hive``. The store *is* the case
    collection — the roster is read from memento, never kept in memory.
    """

    def __init__(self, *args: Any, cases_root: str | Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cases_root = Path(cases_root)
        self._case_manager = DshCaseAgentManager(self, self._cases_root)

    @property
    def memento(self) -> Memento:
        return self._case_manager.memento

    @property
    def cases(self) -> DshCaseAgentManager:
        return self._case_manager
