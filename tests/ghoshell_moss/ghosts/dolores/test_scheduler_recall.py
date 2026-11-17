"""DshSchedulerRecaller — the dolores binding between a plan's anchor and a live session.

What is under test:

- ``stamp`` prefers the in-flight turn, falls back to the last completed one, and stays
  honest when no turn exists yet;
- ``describe`` renders a pointer the model can read, and stays silent on a broken anchor;
- ``chat`` asks the shared caller at the recorded turn, with the scheduler prompt asset;
- an anchor without a session is refused with a reason, not a stack trace.
"""

from types import SimpleNamespace

import pytest

from ghoshell_moss.core.mindflow.scheduler import decode_anchor, encode_anchor
from ghoshell_moss.ghosts.dolores._ego import EgoState
from ghoshell_moss.ghosts.dolores.mindset.emanation_task import EmanationTasks
from ghoshell_moss.ghosts.dolores.mindset.prompter import DefaultPrompterKeys
from ghoshell_moss.ghosts.dolores.mindset.seeds import SCHEDULER_CHAT_SEED
from ghoshell_moss.ghosts.dolores.scheduler_recall import (
    METATYPE,
    DshSchedulerRecaller,
    SchedulerRecallerProvider,
)


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


class FakeTasks:
    """The shared caller, reduced to what this recaller must hand it."""

    def __init__(self, text: str = "answer") -> None:
        self.text = text
        self.calls: list[tuple[str, dict]] = []

    async def chat(self, prompt: str, **kwargs) -> str:
        self.calls.append((prompt, kwargs))
        return self.text


def make_state(**kwargs) -> EgoState:
    state = EgoState()
    for key, value in kwargs.items():
        setattr(state, key, value)
    return state


def test_stamp_points_at_the_in_flight_turn():
    state = make_state(session_id="session-1", turn=811, turn_in_flight=812)
    anchor = DshSchedulerRecaller(None, state).stamp()
    assert anchor is not None
    assert anchor.metatype == METATYPE
    assert decode_anchor(anchor) == {"session": "session-1", "turn": 812}


def test_stamp_falls_back_to_the_last_completed_turn():
    state = make_state(session_id="session-1", turn=811)
    anchor = DshSchedulerRecaller(None, state).stamp()
    assert anchor is not None
    assert decode_anchor(anchor) == {"session": "session-1", "turn": 811}


def test_stamp_refuses_without_a_turn():
    state = make_state(session_id="session-1")
    assert DshSchedulerRecaller(None, state).stamp() is None


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
async def test_chat_asks_the_shared_caller_at_the_recorded_turn():
    tasks = FakeTasks("because the release was due")
    recaller = DshSchedulerRecaller(tasks, make_state())

    anchor = encode_anchor(METATYPE, session="session-1", turn=812)
    answer = await recaller.chat(anchor, "why did you plan this?")

    assert answer == "because the release was due"
    prompt, kwargs = tasks.calls[0]
    assert prompt == "why did you plan this?"
    assert kwargs["anchor"].session_id == "session-1"
    assert kwargs["anchor"].turn == 812
    assert kwargs["task_key"] == DefaultPrompterKeys.SCHEDULER_CHAT
    assert 'planned_turn="812"' in kwargs["position"]
    assert kwargs["seed"] == SCHEDULER_CHAT_SEED
    assert kwargs["kind"] == "scheduler.chat"
    assert kwargs["subject"] == "plan turn=812"


@pytest.mark.asyncio
async def test_chat_rides_the_emanation_container():
    session = FakeSession(text="because the release was due")
    connection = FakeConnection(session)
    recaller = DshSchedulerRecaller(EmanationTasks(connection), make_state())

    anchor = encode_anchor(METATYPE, session="session-1", turn=812)
    answer = await recaller.chat(anchor, "why did you plan this?")

    assert answer == "because the release was due"
    assert connection.asked_session == "session-1"
    assert session.attached_anchor.session_id == "session-1"
    assert session.attached_anchor.turn == 812
    assert session.closed is True
    assert '<emanation-reminder task="scheduler.chat"' in session.message
    assert 'planned_turn="812"' in session.message
    assert "concurrent Emanation of yourself above" in session.message
    assert "<question>why did you plan this?</question>" in session.message


@pytest.mark.asyncio
async def test_chat_refuses_an_anchor_with_no_session():
    recaller = DshSchedulerRecaller(None, make_state())
    with pytest.raises(ValueError, match="no session anchor"):
        await recaller.chat(encode_anchor(METATYPE), "why")


def test_provider_builds_from_the_bound_contracts():
    """The runtime binds EgoState; the provider is what knows the concrete recaller."""
    from ghoshell_container import Container

    con = Container(name="scheduler-recaller-provider-test")
    con.set(EgoState, make_state(session_id="session-1", turn_in_flight=812))
    con.set(EmanationTasks, EmanationTasks(FakeConnection(FakeSession())))

    recaller = SchedulerRecallerProvider().factory(con)

    assert isinstance(recaller, DshSchedulerRecaller)
    assert decode_anchor(recaller.stamp()) == {"session": "session-1", "turn": 812}
