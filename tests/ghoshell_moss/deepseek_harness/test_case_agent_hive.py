"""Case store and op-line behavior — no engine, no carrier.

The parts tested here are the ones a carrier cannot make true: what a draft writes, what a
case reads back, how a reply is read as a step, and what an unknown word means.
"""

from __future__ import annotations

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
