"""MOSS Message → pydantic AI 消息适配.

Atom 原型专用，仅处理 text + base64 image 两种 content type.
后续抽象为独立 MessageAdapter 时路径清晰.
"""

from typing import Iterable

from ghoshell_moss.message import Message, Text, Base64Image
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai import UserContent, TextContent, ImageUrl

__all__ = ["messages_to_parts", "moment_to_request", "moment_to_user_text"]


def messages_to_parts(messages: Iterable[Message]) -> list[UserContent]:
    """将 MOSS Message 转为 pydantic AI UserContent 列表."""
    parts: list[UserContent] = []
    for msg in messages:
        for content in msg.as_contents(with_meta=True):
            if text := Text.from_content(content):
                parts.append(TextContent(content=text.text))
            elif base64_image := Base64Image.from_content(content):
                parts.append(ImageUrl(url=base64_image.data_url))
    return parts


def moment_to_user_text(moment) -> str:
    """将 Moment 的所有请求消息合并为单个纯文本字符串.

    避免 pydantic_ai OpenAIModel streaming 与包含 XML 的多个 TextContent
    user_prompt 不兼容 (AssertionError: Expected code to be unreachable).
    perspectives (mindflow 状态) 与用户输入合并为一段文本传给模型.
    """
    chunks: list[str] = []
    for msg in moment.as_request_messages():
        for content in msg.as_contents(with_meta=True):
            if text := Text.from_content(content):
                if text.text and text.text.strip():
                    chunks.append(text.text)
    return "\n\n".join(chunks)


def moment_to_request(moment) -> ModelRequest:
    """将 Moment 转为 pydantic AI ModelRequest."""
    from ghoshell_moss.core.blueprint.mindflow import Moment as _Moment
    parts = messages_to_parts(moment.as_request_messages())
    return ModelRequest(parts=parts)
