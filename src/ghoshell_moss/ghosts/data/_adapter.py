"""MOSS Message / Moment to pydantic-ai message adapters for Data."""

from collections.abc import Iterable

from pydantic_ai import ImageUrl, TextContent, UserContent
from pydantic_ai.messages import ModelRequest

from ghoshell_moss.core.blueprint.mindflow import Moment
from ghoshell_moss.message import Base64Image, Message, Text

__all__ = ["messages_to_parts", "moment_to_request"]


def messages_to_parts(messages: Iterable[Message]) -> list[UserContent]:
    parts: list[UserContent] = []
    for message in messages:
        for content in message.as_contents(with_meta=True):
            if text := Text.from_content(content):
                parts.append(TextContent(content=text.text))
            elif image := Base64Image.from_content(content):
                parts.append(ImageUrl(url=image.data_url))
    return parts


def moment_to_request(moment: Moment) -> ModelRequest:
    return ModelRequest(parts=messages_to_parts(moment.as_request_messages()))
