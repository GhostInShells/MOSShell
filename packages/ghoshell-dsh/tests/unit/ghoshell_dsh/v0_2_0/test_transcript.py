"""TranscriptView — the reading of a conversation, and the rules that make it one.

What these tests hold down is the *exclusions*, because they are the reason this view exists at
all: tool results, reasoning, and synthetic user messages must not appear in a reading, however
convenient it would be to render them.
"""

from __future__ import annotations

import pytest
from ghoshell_dsh.blueprint.models.session_events import SessionEvent, SessionEventMeta
from ghoshell_dsh.blueprint.transcript import TranscriptView

TS, UM, AM, TC, TR, TE = (
    "turn/start", "user/message", "assistant/message", "tool/call", "tool/result", "turn/end",
)


def _log(*events) -> list[SessionEvent]:
    return [
        SessionEvent(meta=SessionEventMeta(type=kind, seq=index, time=0), data=data)
        for index, (kind, data) in enumerate(events)
    ]


def _user(text: str, kind: str = "user") -> dict:
    return {"content": [{"type": "text", "text": text}], "source": {"kind": kind}}


def _assistant(text: str) -> dict:
    return {"message": {"content": [{"type": "text", "text": text}]}}


def three_turns() -> list[SessionEvent]:
    events = []
    for turn in range(3):
        events += [
            (TS, {"turn": turn}),
            (UM, _user(f"question {turn}")),
            (AM, _assistant(f"answer {turn}")),
            (TC, {"name": "search", "arguments": '{"q": "x"}'}),
            (TR, {"message": {"content": [{"type": "text", "text": "HUGE RESULT"}]}}),
            (TE, {"turn": turn}),
        ]
    return _log(*events)


class _PagingSession:
    """Pages backwards over a fixed log, the way a real session does."""

    def __init__(self, events: list[SessionEvent]):
        self._events = events
        self.pages = 0

    async def history(self, *, before_seq=None, max_messages=None):
        self.pages += 1
        end = len(self._events) if before_seq is None else before_seq
        size = max_messages or 10
        return self._events[max(0, end - size) : end]


@pytest.mark.asyncio
async def test_read_renders_the_whole_conversation_and_counts_it():
    session = _PagingSession(three_turns())
    view = await TranscriptView.read(session)
    assert view.turns == 3
    assert view.messages == 6
    assert view.tool_calls == 3
    assert view.text.splitlines()[0] == "  > question 0"
    assert "HUGE RESULT" not in view.text


@pytest.mark.asyncio
async def test_read_keeps_only_the_newest_turns_that_were_asked_for():
    session = _PagingSession(three_turns())
    view = await TranscriptView.read(session, limit_turns=1)
    assert view.turns == 1
    assert "question 0" not in view.text
    assert "question 2" in view.text


@pytest.mark.asyncio
async def test_read_can_drop_tool_calls_entirely():
    session = _PagingSession(three_turns())
    view = await TranscriptView.read(session, include_tool_calls=False)
    assert view.tool_calls == 0
    assert "@" not in view.text


@pytest.mark.asyncio
async def test_reading_excludes_synthetic_context_and_results():
    events = _log(
        (TS, {"turn": 0}),
        (UM, _user("real words")),
        (UM, _user("file changed: a.py", kind="plugin")),
        (AM, {"message": {"content": [
            {"type": "reasoning", "text": "private scratch"},
            {"type": "text", "text": "public answer"},
        ]}}),
        (TC, {"name": "read_file", "arguments": '{"path": "a.py"}'}),
        (TR, {"message": {"content": [{"type": "text", "text": "file body"}]}}),
        (TE, {"turn": 0}),
    )
    view = await TranscriptView.read(_PagingSession(events))
    assert view.text == (
        "  > real words\n"
        "  ~ public answer\n"
        '  @ read_file({"path": "a.py"})'
    )
    assert "private scratch" not in view.text
    assert "file body" not in view.text
    assert "file changed" not in view.text


@pytest.mark.asyncio
async def test_arguments_are_capped_but_newlines_are_kept():
    events = _log((TS, {"turn": 0}), (TC, {"name": "edit", "arguments": "a\nb\nc"}), (TE, {"turn": 0}))
    view = await TranscriptView.read(_PagingSession(events), tool_arguments_limit=3)
    # "a\nb\nc" capped at 3 chars is "a\n…": the newline survives, so the body folds
    # under the marker instead of being flattened into one line.
    assert view.text.splitlines() == ["  @ edit(a", "    …)"]
