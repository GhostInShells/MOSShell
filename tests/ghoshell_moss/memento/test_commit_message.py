"""Contract tests for ``CommitMessage`` — the commit message's protocol value.

The class is storage-independent: these cases exercise parse / render / validation only,
plus the ``CommitView`` delegation that makes the value reachable from a read view.
"""

import pytest

from ghoshell_moss.memento.abcd import CommitMessage, CommitRef, CommitView, Note


def test_parse_op_line_with_target_and_body():
    message = CommitMessage.parse("propose(dev): v1\n\nbody line\nsecond")

    assert (message.op, message.target, message.content) == ("propose", "dev", "v1")
    assert message.body == "body line\nsecond"
    assert message.headline == "propose(dev): v1"
    assert message.to_str() == "propose(dev): v1\n\nbody line\nsecond"


def test_parse_prose_is_tolerated():
    message = CommitMessage.parse("D11 15:28–18:06：他一句「先提交」\nbody")

    # `D11` is followed by a space, not a colon, so this is not an op line at all.
    assert message.op == ""
    assert message.headline == "D11 15:28–18:06：他一句「先提交」"
    assert message.body == "body"


def test_op_is_a_candidate_reading_not_a_guarantee():
    # Full-width colon + no space: the grammar reads `D11` as an op. Consumers validate
    # the word against their own vocabulary; the value object only reads syntax.
    message = CommitMessage.parse("D11：随手记")

    assert (message.op, message.content) == ("D11", "随手记")


def test_parse_empty_message():
    message = CommitMessage.parse("")

    assert (message.op, message.target, message.content, message.body) == ("", "", "", "")
    assert message.headline == ""
    assert message.to_str() == ""


def test_strict_requires_an_op_line():
    with pytest.raises(ValueError):
        CommitMessage.parse("just prose", strict=True)


def test_body_separator_is_normalized():
    message = CommitMessage.parse("title\n\n\nbody\n\n")

    assert message.body == "body"
    assert message.to_str() == "title\n\nbody"


@pytest.mark.parametrize("kwargs", [
    {"op": "1bad"},
    {"op": "has space"},
    {"target": "x"},
    {"op": "exit", "target": "a(b"},
    {"op": "exit", "content": "two\nlines"},
])
def test_construction_rejects_invalid_protocol(kwargs):
    with pytest.raises(ValueError):
        CommitMessage(**kwargs)


def test_str_is_the_write_path():
    message = CommitMessage(op="exit", content="graceful shutdown", body="last final answer")

    assert str(message) == "exit: graceful shutdown\n\nlast final answer"


def test_commit_view_delegates_to_the_message_protocol():
    view = CommitView(
        branch_index=1,
        ref=CommitRef(seq=7),
        note=Note(commit_id="c1", message="exit: graceful shutdown\n\nlast final answer"),
    )

    message = view.message_view()
    assert (message.op, message.target, message.content, message.body) == (
        "exit", "", "graceful shutdown", "last final answer",
    )
    assert view.title == "exit: graceful shutdown"
    assert view.body == "last final answer"


def test_commit_view_message_view_is_pure():
    view = CommitView(branch_index=1, ref=CommitRef(seq=1), note=Note(commit_id="c", message="title"))

    assert view.message_view() is not view.message_view()
    assert view.message_view().content == "title"
