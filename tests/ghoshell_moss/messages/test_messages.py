import pytest

from ghoshell_moss.message import Message, Text, MessageMeta


def test_message_baseline():
    msg = Message.new(role="user")
    assert msg.role == "user"

    msg.with_content(*[Text.new("hello").to_content()])
    assert len(msg.contents) == 1


def test_message_meta_attributes_str():
    meta = MessageMeta()
    assert meta.gen_attributes_str() == ""
