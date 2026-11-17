"""emanation task — the request container, the parse, and the retry.

Each test proves one promise of the mechanism: the container carries the schema, a rejected
attempt says why and is asked again, and nothing invalid ever comes back as a value.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from ghoshell_moss.deepseek_harness.v0_2_0 import DshSessionAnchor
from ghoshell_moss.ghosts.dolores.mindset.emanation_task import (
    TOOL_NAME,
    EmanationChatError,
    EmanationTaskError,
    EmanationTasks,
)


class Verdict(BaseModel):
    ok: bool
    reason: str


def _anchor() -> DshSessionAnchor:
    return DshSessionAnchor(session_id="s-1", seq=7)


def _call(args: dict, name: str = TOOL_NAME) -> SimpleNamespace:
    return SimpleNamespace(name=name, arguments=json.dumps(args))


def _reply(text: str = "", calls: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(text=text, tool_calls=list(calls or []), usage=None)


class _Session:
    def __init__(self, replies: list[SimpleNamespace]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple] = []

    async def chat_off_record(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return self.replies.pop(0)


class _Connection:
    def __init__(self, session: _Session) -> None:
        self.session = session

    def attach_session(self, session_id, logger=None, *, follow=True):
        assert follow is False, "an emanation task must not open a follow stream"
        assert session_id == "s-1"
        return self.session


class _Prompts:
    """A prompt registry with one visible convention: the key is the instruction."""

    def render(self, key: str) -> str:
        return f"PROMPT:{key}"


def _body(call) -> str:
    messages, _ = call
    return messages[0][0].text


@pytest.mark.asyncio
async def test_returns_the_validated_answer_and_carries_the_schema():
    session = _Session([_reply(calls=[_call({"id": "et-1", "resp": {"ok": True, "reason": "fine"}})])])
    tasks = EmanationTasks(_Connection(session), retries=0)

    value = await tasks.request("judge it", Verdict, anchor=_anchor())

    assert value == Verdict(ok=True, reason="fine")
    body = _body(session.calls[0])
    assert '<emanation-task-request id="et-1">' in body
    assert "<response-schema>" in body
    assert '"ok"' in body and '"type": "boolean"' in body
    assert TOOL_NAME in body


@pytest.mark.asyncio
async def test_a_string_resp_is_rejected_and_asked_again():
    session = _Session([
        _reply(calls=[_call({"id": "et-1", "resp": "ok"})]),
        _reply(calls=[_call({"id": "et-1", "resp": {"ok": False, "reason": "no"}})]),
    ])
    tasks = EmanationTasks(_Connection(session), retries=1)

    value = await tasks.request("judge it", Verdict, anchor=_anchor())

    assert value == Verdict(ok=False, reason="no")
    assert len(session.calls) == 2
    messages, _ = session.calls[1]
    retry = messages[1][0].text
    assert "<emanation-task-retry" in retry
    assert "resp must be a JSON object" in retry


@pytest.mark.asyncio
async def test_prose_answer_is_rejected_and_eventually_raises():
    session = _Session([_reply("I think it is fine."), _reply("Really.")])
    tasks = EmanationTasks(_Connection(session), retries=1)

    with pytest.raises(EmanationTaskError, match="no valid answer"):
        await tasks.request("judge it", Verdict, anchor=_anchor())

    assert len(session.calls) == 2


@pytest.mark.asyncio
async def test_a_call_for_another_request_id_is_rejected():
    session = _Session([
        _reply(calls=[_call({"id": "et-99", "resp": {"ok": True, "reason": "fine"}})]),
        _reply(calls=[_call({"id": "et-1", "resp": {"ok": True, "reason": "fine"}})]),
    ])
    tasks = EmanationTasks(_Connection(session), retries=1)

    value = await tasks.request("judge it", Verdict, anchor=_anchor())

    assert value.ok is True
    messages, _ = session.calls[1]
    assert "wrong id" in messages[1][0].text


@pytest.mark.asyncio
async def test_chat_rides_the_emanation_container_and_returns_the_text():
    session = _Session([_reply("because the release was due")])
    tasks = EmanationTasks(_Connection(session))

    text = await tasks.chat(
        "why?",
        anchor=_anchor(),
        task_key="scheduler.chat",
        position=' at="2026-11-18 00:00:00+8" planned_turn="2"',
        seed="SEED",
    )

    assert text == "because the release was due"
    body = _body(session.calls[0])
    assert '<emanation-reminder task="scheduler.chat"' in body
    assert ' at="2026-11-18 00:00:00+8" planned_turn="2"' in body
    assert "<instruction>SEED</instruction>" in body
    assert "<question>why?</question>" in body


@pytest.mark.asyncio
async def test_chat_renders_the_prompt_asset_when_a_manager_is_bound():
    session = _Session([_reply("ok")])
    tasks = EmanationTasks(_Connection(session), prompts=_Prompts())

    await tasks.chat(
        "why?",
        anchor=_anchor(),
        task_key="memento.chat",
        position=' at="2026-11-18 00:00:00+8"',
        seed="SEED",
    )

    assert "<instruction>PROMPT:memento.chat</instruction>" in _body(session.calls[0])


@pytest.mark.asyncio
async def test_chat_rejects_a_tool_call_answer():
    session = _Session([_reply(calls=[_call({}, name="bash")])])
    tasks = EmanationTasks(_Connection(session))

    with pytest.raises(EmanationChatError, match="tool call"):
        await tasks.chat(
            "why?",
            anchor=_anchor(),
            task_key="scheduler.chat",
            position="",
            seed="SEED",
        )


@pytest.mark.asyncio
async def test_chat_rejects_an_answer_with_no_text():
    session = _Session([_reply("")])
    tasks = EmanationTasks(_Connection(session))

    with pytest.raises(EmanationChatError, match="returned no text"):
        await tasks.chat(
            "why?",
            anchor=_anchor(),
            task_key="scheduler.chat",
            position="",
            seed="SEED",
        )
