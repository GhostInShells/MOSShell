"""moss Message → pydantic-ai parts — the message conversion protocol.

Formalizes the ghosts/atom/_adapter.py pattern into the pydantic-ai engine
of LLMFuncs: general (any moss Message), pydantic-ai imports are lazy
(no module-level drag), unknown content degrades to text so nothing is
dropped. Each model engine owns a moss→engine conversion; this is the
pydantic-ai one. ``as_contents`` renders the message's meta as XML context
when the message carries a tag.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from pydantic_ai import UserContent

    from ghoshell_moss.message import Message

__all__ = ["message_to_parts", "messages_to_parts"]


def message_to_parts(message: Message, *, with_meta: bool = False) -> list[UserContent]:
    """Convert one moss Message into pydantic-ai UserContent parts.

    Text → TextContent; Base64Image → BinaryContent(decoded bytes); any other
    content type degrades to its text representation (Message.content_as_string)
    — the conversion never drops content, it only lowers fidelity.

    ``BinaryContent`` (not ``ImageUrl``) is the protocol-neutral image carrier:
    pydantic-ai serializes it as a base64 ``source`` block on the anthropic
    protocol and as ``image_url`` on the openai protocol. ``ImageUrl(data_url)``
    would instead send ``source:{type:'url', url:'data:...'}`` on the anthropic
    path, which DeepSeek's anthropic-compatible endpoint rejects — the two
    providers' image protocols are not the same.

    Content order is preserved (``join_text=False``) — multimodal order
    matters ("look at @img, now describe it"). ``with_meta=False`` sends the
    raw contents, no MOSS meta XML; pass True to render the message's meta
    as XML context.
    """
    from pydantic_ai import BinaryContent, TextContent

    from ghoshell_moss.message import Base64Image, Text

    parts: list[UserContent] = []
    for content in message.as_contents(with_meta=with_meta, join_text=False):
        if text := Text.from_content(content):
            parts.append(TextContent(content=text.text))
        elif image := Base64Image.from_content(content):
            source = image.source or {}
            data = source.get("data")
            if data:
                parts.append(BinaryContent(
                    data=base64.b64decode(data),
                    media_type=source.get("media_type") or "image/png",
                ))
            else:
                parts.append(TextContent(content=message.content_as_string(content)))
        else:
            parts.append(TextContent(content=message.content_as_string(content)))
    return parts


def messages_to_parts(
        messages: Iterable[Message],
        *,
        with_meta: bool = False,
) -> list[UserContent]:
    """Convert multiple moss Messages into one flat pydantic-ai part list."""
    parts: list[UserContent] = []
    for message in messages:
        parts.extend(message_to_parts(message, with_meta=with_meta))
    return parts
