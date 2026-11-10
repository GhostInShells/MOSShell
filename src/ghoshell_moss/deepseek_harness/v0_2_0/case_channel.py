"""The cases channel — the issuer's control surface over a case store.

A **sustain child** of the node's own channel: cases are the thing the hive exists for, so
this channel is mounted from the start, unlike ``agents``/``ui`` which stay closed until the
ghost opens them.

Three things decide the shape:

- **Commands take one payload.** ``draft`` / ``bind`` / ``autopilot`` each take a JSON body
  whose schema is the model's own pydantic schema, rendered into the docstring at build time.
  That is the same text the ghost reads, so there is one description, not a hand-written copy
  that drifts from the code.
- **Named notices are the perception face.** One fragment per case: state, whether a loop is
  running, the last coordinate, and — the part that decides the next action — what the last
  step said (`op→next`) and how the last run ended. A fragment can announce its own
  disappearance. The per-case detail is a cold read (``read``), never notice material.
- **The reading faces are commands.** ``read`` and ``chat`` are explicit acts — the trace is
  the interface, but only when someone looks.
"""

from __future__ import annotations

import json

from ghoshell_moss.agents.blueprint.case import (
    AutopilotRequest,
    BindRequest,
    CaseDefinition,
    Operation,
    ProtocolTemplate,
)
from ghoshell_moss.core.blueprint.channel_builder import MutableChannel, new_channel

from .case_agent_hive import DshCaseAgentManager

__all__ = ["new_cases_channel"]

CHANNEL_NAME = "cases"

_INSTRUCTION = """\
The cases this hive holds. A case is one stateful task with its own trajectory; it survives
every carrier that runs it.

- `draft(definition)` creates one: it writes the case documents and returns the case's path.
  Authoring a case means editing `CASE.md` there — the body is the brief.
- `bind(request)` defines a **new** carrier for a case: it opens a fresh session and teaches it
  the case. It is the explicit redefinition — use it when the session is gone or a different
  composition is wanted, not as the way back after a restart.
- `autopilot(request)` continues the case and returns at once. With no live carrier it reattaches
  to the session the last commit names (a reboot: the engine resumes it), then dispatches the
  loop. It fails only when that session is archived or gone — then `bind` defines a new one.
  Every finished step arrives as a commit — read them with `read`, or ask one a question with
  `chat`.
- `pause` stops a running case from the outside; `resolve` declares it finished; `close`
  abandons it and archives its branch.

A case whose loop stopped is still open: `autopilot` again is how it continues.
"""


def _failed(exc: Exception) -> str:
    """One failure shape: the type is part of the reason (an empty message is not a reason)."""
    detail = str(exc).strip()
    return f"failed: {type(exc).__name__}: {detail}" if detail else f"failed: {type(exc).__name__}"


def _schema(model: type) -> str:
    """The model's JSON schema, as the body the command expects."""
    return json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2)


def new_cases_channel(manager: DshCaseAgentManager) -> MutableChannel:
    chan = new_channel(name=CHANNEL_NAME, description="the cases this hive holds, by name.")

    # -- perception ------------------------------------------------------------------------ #

    @chan.build.instruction
    def _instruction() -> str:
        return _INSTRUCTION

    @chan.build.notice
    def _notice() -> str:
        cases = manager.list_cases()
        running = sum(1 for case in cases if _running(manager, case.branch.ref.name))
        return f"{manager.memento.root} · {len(cases)} case(s), {running} running"

    @chan.build.named_notices
    def _named_notices() -> dict[str, str | None]:
        fragments: dict[str, str | None] = {}
        for case in manager.list_cases():
            name = case.branch.ref.name
            refs = case.branch.commits()
            last = f"{case.branch.index}-{refs[-1].seq}" if refs else "empty"
            state = case.meta.state.value
            running = "running" if _running(manager, name) else "idle"
            description = case.brief.description.strip()
            if len(description) > 80:
                description = description[:80] + "…"
            line = f"{state} · {running} · {last}"
            step = _last_step_fragment(refs)
            if step:
                line += f" · {step}"
            stop = _last_run_fragment(refs)
            if stop:
                line += f" · {stop}"
            if description:
                line += f" · {description}"
            fragments[name] = line
        return fragments

    @chan.build.refresh_meta
    async def _refresh_meta() -> None:
        await manager._hive.refresh_if_stale()  # noqa: SLF001 — the hive is this channel's owner

    # -- lifecycle -------------------------------------------------------------------------- #

    @chan.build.command(name="draft", blocking=False, always_observe=True, doc=_draft_doc())
    async def _draft(text__: str) -> str:
        try:
            definition = CaseDefinition.model_validate_json(text__)
            case = await manager.draft(definition)
        except Exception as exc:  # noqa: BLE001 — a refused draft is a fact, not a crash
            return _failed(exc)
        return f"ok: {case.branch.ref.name} · {case.document}"

    @chan.build.command(name="bind", blocking=False, always_observe=True, doc=_bind_doc())
    async def _bind(text__: str) -> str:
        try:
            request = BindRequest.model_validate_json(text__)
            agent = await manager.bind(request)
        except Exception as exc:  # noqa: BLE001
            return _failed(exc)
        return f"ok: bound {agent.name} ({agent.carrier.session_id})"

    @chan.build.command(name="autopilot", blocking=False, always_observe=True, doc=_autopilot_doc())
    async def _autopilot(text__: str) -> str:
        try:
            request = AutopilotRequest.model_validate_json(text__)
            agent = await manager.autopilot(request)
        except Exception as exc:  # noqa: BLE001
            return _failed(exc)
        return f"ok: running {agent.name}"

    @chan.build.command(name="pause", blocking=False, always_observe=True)
    async def _pause(case: str, reason: str = "") -> str:
        """Stop a running case from the outside; the case stays open and the carrier stays bound.

        :param case: the case name.
        :param reason: travels to the carrier as a system reminder, so the agent knows why.
        """
        try:
            await manager.pause(case, reason)
        except Exception as exc:  # noqa: BLE001
            return _failed(exc)
        return "ok"

    @chan.build.command(name="resolve", blocking=False, always_observe=True)
    async def _resolve(case: str, reason: str = "") -> str:
        """Declare a case finished. The trajectory stays readable; the name keeps resolving."""
        try:
            await manager.resolve(case, reason)
        except Exception as exc:  # noqa: BLE001
            return _failed(exc)
        return "ok"

    @chan.build.command(name="close", blocking=False, always_observe=True)
    async def _close(case: str, reason: str = "") -> str:
        """Abandon a case: stop its carrier, archive its branch, drop the name."""
        try:
            await manager.close(case, reason)
        except Exception as exc:  # noqa: BLE001
            return _failed(exc)
        return "ok"

    # -- reading ---------------------------------------------------------------------------- #

    @chan.build.command(name="read", blocking=False, always_observe=True)
    async def _read(case: str, coord: str = "", limit: int = 10) -> str:
        """Read a case's steps as steps: the latest ones, or the one at a coordinate.

        :param case: the case name.
        :param coord: ``{branch}-{seq}``; empty = the latest ``limit`` commits.
        :param limit: how many commits to read, from the newest.
        """
        try:
            commits = manager.read(case, coord, limit)
        except Exception as exc:  # noqa: BLE001
            return _failed(exc)
        if not commits:
            return "(no commits)"
        lines = []
        for commit in commits:
            head = commit.headline or "-"
            detail = commit.message.replace("\n", " · ")
            if len(detail) > 220:
                detail = detail[:220] + "…"
            lines.append(f"{commit.coord} [{commit.metatype or 'step'}] {head}")
            if commit.operation is not None and commit.operation.violation:
                lines.append("  (violation: no op line)")
            lines.append(f"  {detail}")
        return "\n".join(lines)

    @chan.build.command(name="chat", blocking=False, always_observe=True)
    async def _chat(case: str, coord: str, question: str) -> str:
        """Ask one step a question; answers come back as a signal, not as this command's return.

        The receipt names the exchange: the signal carries `case`, `coord` and `chat=<id>`, so a
        result can be matched to the question that produced it. The exchange is not written into
        the case.

        :param case: the case name.
        :param coord: the step to ask about, e.g. ``1-42``.
        :param question: what you want to know.
        :return: `ok: <chat id>`, or `failed: <Type>: <reason>`.
        """
        try:
            chat_id = await manager.chat(case, coord, question)
        except Exception as exc:  # noqa: BLE001
            return _failed(exc)
        return f"ok: {chat_id}"

    # -- protocol library --------------------------------------------------------------------- #

    @chan.build.command(name="protocols", blocking=False, always_observe=True)
    async def _protocols() -> str:
        """Cold read: the protocol templates this hive can compose a case from."""
        items = await manager.protocols()
        if not items:
            return "(no protocols)"
        return "\n".join(
            f"{item.name} · {len(item.ops)} ops — {item.description}" for item in items
        )

    @chan.build.command(name="save_protocol", blocking=False, always_observe=True)
    async def _save_protocol(text__: str) -> str:
        """Store a protocol template, replacing any template of the same name.

        :param text__: one ``ProtocolTemplate`` as JSON:
            {"name": "...", "description": "...",
             "introduction": "...",
             "ops": [{"name": "do", "instruction": "...", "next": "continue", "level": "info"}]}
        """
        try:
            template = ProtocolTemplate.model_validate_json(text__)
            await manager.save_protocol(template)
        except Exception as exc:  # noqa: BLE001
            return _failed(exc)
        return f"ok: {template.name}"

    return chan


def _running(manager: DshCaseAgentManager, name: str) -> bool:
    agent = manager._agents.get(name)  # noqa: SLF001 — the roster of live carriers is runtime
    return agent is not None and agent.is_running()


def _last_step_fragment(refs: list) -> str:
    """``op→next`` of the newest step — the verdict a roster scan needs.

    A case whose last word was `done→break` is waiting for ``resolve``; one whose word was
    `research→continue` is waiting for another run. Leaving this out forced one ``read`` per
    idle case just to decide what to do with it.
    """
    for ref in reversed(refs):
        operation = Operation.from_commit_ref(ref)
        if operation is not None:
            return f"{operation.op or 'violation'}→{operation.next.value}"
    return ""


def _last_run_fragment(refs: list) -> str:
    """How the last run ended — `exhausted` wants a new grant, `done` wants resolve."""
    for ref in reversed(refs):
        if ref.metatype != "run.end" or not isinstance(ref.metadata, dict):
            continue
        run = ref.metadata.get("run")
        if isinstance(run, dict):
            return str(run.get("stop") or "")
        return ""
    return ""


def _draft_doc() -> str:
    return (
        "Create a case: the branch, CASE.md and its control document; returns the document path.\n"
        "Authoring the case is editing that file afterwards — the body is the brief.\n\n"
        ":param text__: one CaseDefinition as JSON:\n" + _schema(CaseDefinition)
    )


def _bind_doc() -> str:
    return (
        "Define a new carrier for a case: read the documents, prime the new session, record an\n"
        "init commit. Every field but `case_name` is optional; an empty bind reuses the recorded\n"
        "form but still opens a new session — this is the explicit redefinition, not the way back\n"
        "after a restart (use `autopilot` for that).\n\n"
        ":param text__: one BindRequest as JSON:\n" + _schema(BindRequest)
    )


def _autopilot_doc() -> str:
    return (
        "Continue the case: reattach a carrier if needed, then dispatch the loop; returns at\n"
        "once. With no live carrier it resumes the session the last commit names; it fails only\n"
        "when that session is archived or gone (then `bind` defines a new carrier).\n"
        "The loop stops on a violation, an exhausted budget, a break word, or `pause`.\n\n"
        ":param text__: one AutopilotRequest as JSON:\n" + _schema(AutopilotRequest)
    )
