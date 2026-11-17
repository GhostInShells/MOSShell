"""DshSchedulerRecaller — the dolores binding between a plan's anchor and a live session.

What is under test:

- ``stamp`` writes the live (session, turn) pair, and stays honest before a session exists;
- ``describe`` renders a pointer the model can read, and stays silent on a broken anchor;
- ``chat`` reopens the session at that turn and returns the answer — one round, no record;
- an anchor without a session is refused with a reason, not a stack trace.
"""

from types import SimpleNamespace

import pytest

from ghoshell_moss.core.mindflow.scheduler import decode_anchor, encode_anchor
from ghoshell_moss.ghosts.dolores._ego import EgoState
from ghoshell_moss.ghosts.dolores.scheduler_recall import METATYPE, DshSchedulerRecaller


class FakeSession:
    def __init__(self, text: str = "answer", tool_calls=()) -> None:
        self.text = text
        self.tool_calls = list(tool_calls)
        self.usage = SimpleNamespace(
            inputTokens=1, cacheReadTokens=0, cacheWriteTokens=0,
            outputTokens=1, reasoningTokens=None,
        )
        self.attached_anchor = None
        self.closed = False
        self.message = ""

    async def chat_off_record(self, messages, *, anchor, llm):
        self.attached_anchor = anchor
        self.message = messages[0][0].text
        return self

    async def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, session: FakeSession) -> None:
        self.session = session
        self.asked_session = ""

    def attach_session(self, session_id: str, logger, *, follow: bool = True):
        self.asked_session = session_id
        assert follow is False, "a recall chat must not join the live session's follow stream"
        return self.session


def make_state(**kwargs) -> EgoState:
    state = EgoState()
    for key, value in kwargs.items():
        setattr(state, key, value)
    return state


def test_stamp_points_at_the_live_turn():
    state = make_state(session_id="session-1", turn=812)
    anchor = DshSchedulerRecaller(None, state).stamp()
    assert anchor is not None
    assert anchor.metatype == METATYPE
    assert decode_anchor(anchor) == {"session": "session-1", "turn": 812}


def test_stamp_is_honest_before_a_session_exists():
    assert DshSchedulerRecaller(None, make_state()).stamp() is None


def test_describe_names_the_turn_and_session():
    recaller = DshSchedulerRecaller(None, make_state())
    line = recaller.describe(encode_anchor(METATYPE, session="session-1", turn=812))
    assert "turn 812" in line
    assert "session-" in line


def test_describe_is_silent_on_a_foreign_anchor():
    recaller = DshSchedulerRecaller(None, make_state())
    assert recaller.describe(encode_anchor("someone-else", foo=1)) == ""


@pytest.mark.asyncio
async def test_chat_reopens_the_turn_and_returns_the_answer():
    session = FakeSession(text="because the release was due")
    connection = FakeConnection(session)
    recaller = DshSchedulerRecaller(connection, make_state())

    anchor = encode_anchor(METATYPE, session="session-1", turn=812)
    answer = await recaller.chat(anchor, "why did you plan this?")

    assert answer == "because the release was due"
    assert connection.asked_session == "session-1"
    assert session.attached_anchor.session_id == "session-1"
    assert session.attached_anchor.turn == 812
    assert session.closed is True
    assert "why did you plan this?" in session.message


@pytest.mark.asyncio
async def test_chat_refuses_an_anchor_with_no_session():
    recaller = DshSchedulerRecaller(FakeConnection(FakeSession()), make_state())
    with pytest.raises(ValueError, match="no session anchor"):
        await recaller.chat(encode_anchor(METATYPE), "why")
