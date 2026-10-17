"""What the ghost puts on the bus — and what it deliberately keeps off it.

The session output is the full record: every forwarded dsh event, in order, for
whoever reads that session. The bus is a different surface with a different budget:
a status light (thinking / idle) and the actions (tool calls). Anything else would
be traffic nobody reads, and traffic is what drowns a signal.
"""

from ghoshell_moss.host.ghost_runtime import _PUBLISHED_EVENT_NAMES as PUBLISHED


def test_the_state_light_is_on_the_bus():
    assert "ghost/lifecycle/thinking" in PUBLISHED
    assert "ghost/lifecycle/idle" in PUBLISHED


def test_the_finished_answer_is_one_event():
    """A surface needs what the turn said, whole — one event, not a delta stream."""
    assert "ghost/output" in PUBLISHED


def test_actions_are_on_the_bus():
    assert "dsh/tool/call" in PUBLISHED


def test_the_rest_of_the_session_stream_is_not():
    """turn/*, assistant/message and tool/result stay in the session output only."""
    for name in ("dsh/turn/start", "dsh/turn/end", "dsh/assistant/message", "dsh/tool/result"):
        assert name not in PUBLISHED, name
