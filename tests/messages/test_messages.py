import pytest

from ghoshell_moss.message import Message, Text


def test_message_baseline():
    msg = Message.new(role="user")
    assert msg.role == "user"
    assert msg.seq == "completed"

    incomplete = msg.as_incomplete([Text.new("hello").to_content()])
    assert incomplete.seq == "incomplete"

    head = incomplete.as_head()
    # 测试互相不污染.
    assert head.seq == "head"
    assert msg.seq == "completed"
    assert incomplete.seq == "incomplete"

    with pytest.raises(ValueError):
        incomplete.as_completed(Text.new("hello"))
