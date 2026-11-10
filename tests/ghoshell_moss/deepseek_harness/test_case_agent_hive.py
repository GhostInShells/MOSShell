"""Case store and op-line behavior — no engine, no carrier.

The parts tested here are the ones a carrier cannot make true: what a draft writes, what a
case reads back, how a reply is read as a step, and what an unknown word means.
"""

from __future__ import annotations

import logging

import pytest

from ghoshell_moss.agents.blueprint.case import (
    CaseDefinition,
    CaseProtocol,
    CaseState,
    Level,
    Next,
    Operator,
    ProtocolTemplate,
)
from ghoshell_moss.deepseek_harness.v0_2_0.case_agent_hive import (
    DshCaseAgentManager,
    parse_operation,
)


def _protocol() -> CaseProtocol:
    return CaseProtocol(
        introduction="Every reply is one step: <op>(<target>): <title> then a body.",
        ops=[
            Operator(name="plan", instruction="write the plan", next=Next.CONTINUE, level=Level.NOTICE),
            Operator(name="do", instruction="finish one step", next=Next.CONTINUE),
            Operator(name="done", instruction="finished", next=Next.BREAK, level=Level.NOTICE),
        ],
    )


@pytest.fixture()
def manager(tmp_path):
    # The hive is only consulted by `bind`; these verbs are the store's.
    return DshCaseAgentManager(None, tmp_path / "cases")  # type: ignore[arg-type]


def _definition(name: str = "case-one", protocol=None) -> CaseDefinition:
    return CaseDefinition(
        name=name,
        description="a probe",
        content="do the thing",
        protocol=protocol or _protocol(),
        owner="ghost/moss",
    )


@pytest.mark.asyncio
async def test_draft_writes_the_documents(manager):
    case = await manager.draft(_definition())
    assert case.document.is_file()
    assert "do the thing" in case.document.read_text(encoding="utf-8")
    brief = case.brief
    assert brief.name == "case-one"
    assert brief.protocol.words() == ["plan", "do", "done"]
    assert case.meta.state is CaseState.draft
    assert case.meta.owner == "ghost/moss"
    assert case.tmp.is_dir()
    assert [item.branch.ref.name for item in manager.list_cases()] == ["case-one"]


@pytest.mark.asyncio
async def test_draft_refuses_a_taken_name(manager):
    await manager.draft(_definition())
    with pytest.raises(NameError):
        await manager.draft(_definition())


@pytest.mark.asyncio
async def test_draft_resolves_a_protocol_template(manager):
    await manager.save_protocol(
        ProtocolTemplate(
            name="goal-task",
            description="goal driven",
            introduction="template intro",
            ops=_protocol().ops,
        )
    )
    case = await manager.draft(_definition(protocol="goal-task"))
    assert case.brief.protocol.introduction == "template intro"


@pytest.mark.asyncio
async def test_read_returns_steps_with_their_session_and_usage(manager):
    case = await manager.draft(_definition())
    operation = parse_operation("do(s1): 写完了\n细节", case.brief.protocol)
    case.branch.commit(
        message="do(s1): 写完了\n细节",
        metadata={
            "operation": operation.model_dump(mode="json"),
            "session_ref": "session-1",
            "usage": {"tokens": 42},
        },
    )
    commits = manager.read("case-one")
    assert len(commits) == 1
    commit = commits[0]
    assert commit.headline == "do(s1): 写完了"
    assert commit.session_ref == "session-1"
    assert commit.usage.tokens == 42
    assert commit.metatype == ""


def test_parse_operation_reads_a_step():
    operation = parse_operation("```\nplan(rev1): 三步\nbody\n```", _protocol())
    assert (operation.op, operation.target, operation.title) == ("plan", "rev1", "三步")
    assert operation.next is Next.CONTINUE
    assert not operation.violation


def test_parse_operation_breaks_on_prose_and_unknown_words():
    prose = parse_operation("我打算先看看", _protocol())
    assert prose.violation and prose.next is Next.BREAK

    unknown = parse_operation("sing: 唱歌", _protocol())
    assert unknown.violation and unknown.op == "sing"
    assert unknown.next is Next.BREAK


def test_history_folds_steps_for_a_rebuilt_carrier(tmp_path):
    import asyncio

    from ghoshell_moss.deepseek_harness.v0_2_0.case_agent_hive import DshCase, _render_history

    async def run():
        manager = DshCaseAgentManager(None, tmp_path / "cases")  # type: ignore[arg-type]
        case = await manager.draft(
            CaseDefinition(
                name="history-case", protocol=_protocol(), owner="ghost/moss"
            )
        )
        assert "first run" in _render_history(case)
        operation = parse_operation("do(s1): 写完了", case.brief.protocol)
        case.branch.commit(
            message="do(s1): 写完了\n细节一\n细节二",
            metadata={"operation": operation.model_dump(mode="json")},
        )
        text = _render_history(DshCase(case.branch))
        assert "1-1 do(s1): 写完了" in text
        assert "细节一" in text

    asyncio.run(run())


def test_mirror_is_a_mirror_not_a_copy(tmp_path):
    from ghoshell_moss.deepseek_harness.v0_2_0.case_agent_hive import _mirror

    source = tmp_path / "scratch"
    target = tmp_path / "durable"
    (source / "sub").mkdir(parents=True)
    (source / "a.txt").write_text("a", encoding="utf-8")
    (source / "sub" / "b.txt").write_text("b", encoding="utf-8")
    target.mkdir()
    (target / "stale.txt").write_text("gone", encoding="utf-8")
    (target / "olddir").mkdir()

    _mirror(source, target)
    assert (target / "a.txt").read_text(encoding="utf-8") == "a"
    assert (target / "sub" / "b.txt").read_text(encoding="utf-8") == "b"
    assert not (target / "stale.txt").exists()
    assert not (target / "olddir").exists()

    (source / "a.txt").unlink()
    _mirror(source, target)
    assert not (target / "a.txt").exists()


def test_failed_says_which_failure_it_was():
    from ghoshell_moss.deepseek_harness.v0_2_0.case_channel import _failed

    class Boom(Exception):
        pass

    assert _failed(Boom("nope")) == "failed: Boom: nope"
    assert _failed(Boom("")) == "failed: Boom"


# -- the default start: reattach, do not rebuild ------------------------------------------- #


class _FakeConnection:
    """Only the two archive facts the case layer asks of a connection."""

    def __init__(self) -> None:
        self.archived: set[str] = set()
        self.archived_calls: list[str] = []

    async def session_archived(self, session_id: str) -> bool:
        return session_id in self.archived

    async def archive_session(self, session_id: str, *, stop: bool = True) -> None:
        self.archived_calls.append(session_id)
        self.archived.add(session_id)


class _FakeHive:
    name = "dsh"
    engine_name = "dsh"
    version = "0.2.0"
    logger = logging.getLogger("test.case")

    def __init__(self) -> None:
        self.connection = _FakeConnection()


def _commit_init(case, session_ref: str = "session-1") -> None:
    case.branch.commit(
        message="init: default · repo",
        metatype="init",
        metadata={"session_ref": session_ref, "preset": "", "workspace": "repo"},
    )


@pytest.mark.asyncio
async def test_autopilot_refuses_a_case_that_was_never_bound(tmp_path):
    from ghoshell_moss.agents.blueprint.case import AutopilotRequest

    manager = DshCaseAgentManager(_FakeHive(), tmp_path / "cases")  # type: ignore[arg-type]
    await manager.draft(_definition())
    with pytest.raises(RuntimeError, match="never bound"):
        await manager.autopilot(AutopilotRequest(case_name="case-one"))


@pytest.mark.asyncio
async def test_autopilot_fails_fast_when_the_recorded_session_is_archived(tmp_path):
    from ghoshell_moss.agents.blueprint.case import AutopilotRequest

    hive = _FakeHive()
    manager = DshCaseAgentManager(hive, tmp_path / "cases")  # type: ignore[arg-type]
    case = await manager.draft(_definition())
    _commit_init(case)
    hive.connection.archived.add("session-1")
    with pytest.raises(RuntimeError, match="archived"):
        await manager.autopilot(AutopilotRequest(case_name="case-one"))


@pytest.mark.asyncio
async def test_resolve_archives_an_idle_cases_session_and_closes_its_run(tmp_path):
    from datetime import datetime, timezone

    from ghoshell_moss.agents.blueprint.case import Budget, Run

    hive = _FakeHive()
    manager = DshCaseAgentManager(hive, tmp_path / "cases")  # type: ignore[arg-type]
    case = await manager.draft(_definition())
    _commit_init(case)
    run = Run(started=datetime.now(timezone.utc), budget=Budget(seconds=10))
    case.branch.commit(
        message="run start", metatype="run.start", metadata={"run": run.model_dump(mode="json")}
    )

    await manager.resolve("case-one")

    assert hive.connection.archived_calls == ["session-1"]
    refs = case.branch.commits()
    ends = [ref for ref in refs if ref.metatype == "run.end"]
    assert len(ends) == 1, "a hard-killed run must be closed, not left open forever"
    assert ends[0].metadata["run"]["stop"] == "interrupted"


@pytest.mark.asyncio
async def test_reboot_reminder_marks_the_boundary_and_keeps_the_real_history(tmp_path):
    from ghoshell_moss.deepseek_harness.v0_2_0.case_agent_hive import _render_reboot

    manager = DshCaseAgentManager(None, tmp_path / "cases")  # type: ignore[arg-type]
    case = await manager.draft(_definition())
    operation = parse_operation("done(dsh-node): 报告写完了", case.brief.protocol)
    case.branch.commit(
        message="done(dsh-node): 报告写完了\n正文",
        metadata={"operation": operation.model_dump(mode="json")},
    )
    step = manager._last_step(case)  # noqa: SLF001 — the reminder's read face is what is tested
    assert step is not None

    text = _render_reboot(
        name="case-one",
        last_step=step,
        last_run={"stop": "interrupted"},
        scratch=tmp_path / "scratch",
        trigger="continue: case-one — pick up where you stopped; one step per reply.",
    )

    assert text.startswith("<system_reminder>")
    assert "rebuilt after a restart" in text
    assert "The conversation above is your real history" in text
    assert f"{step.coord} — done(dsh-node): 报告写完了" in text
    assert "Last run: interrupted" in text
    assert str(tmp_path / "scratch") in text
    assert text.rstrip().endswith("one step per reply.")
