"""MOSS Message → dsh UserMessage 单向映射的行为证据."""

import pytest

from ghoshell_moss.message import Base64Image, Message
from ghoshell_moss.deepseek_harness.message_mapper import to_content_block, to_user_message


def test_to_user_message_role_user_and_source():
    msg = Message.new(tag="").with_content("hello world")
    result = to_user_message(msg)
    assert result.role == "user"
    assert result.source.kind == "user"
    assert [b.model_dump(exclude_none=True) for b in result.content] == [
        {"type": "text", "text": "hello world"}
    ]


def test_to_user_message_joins_adjacent_text():
    msg = Message.new(tag="").with_content("hello ", "world")
    result = to_user_message(msg)
    assert len(result.content) == 1
    assert result.content[0].text == "hello world"


def test_to_user_message_tag_wrapped_by_default():
    msg = Message.new(tag="channel", attributes={"foo": "bar"}).with_content("ping")
    result = to_user_message(msg)
    text = result.content[0].text
    assert "<channel" in text
    assert 'foo="bar"' in text
    assert "</channel>" in text


def test_to_user_message_without_meta_keeps_plain_text():
    msg = Message.new(tag="channel").with_content("ping")
    result = to_user_message(msg, with_meta=False)
    assert result.content[0].text == "ping"


def test_to_content_block_text_direct():
    block = to_content_block({"type": "text", "text": "hi"})
    assert block.type == "text"
    assert block.text == "hi"


def test_to_user_message_image_raises_not_implemented():
    msg = Message.new(tag="").with_content(
        Base64Image.from_base64(media_type="image/png", data="AAAA")
    )
    with pytest.raises(NotImplementedError):
        to_user_message(msg)
