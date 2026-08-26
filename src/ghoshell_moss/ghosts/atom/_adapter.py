"""MOSS Message → pydantic AI 消息适配.

Atom 原型专用，仅处理 text + base64 image 两种 content type.
后续抽象为独立 MessageAdapter 时路径清晰.
"""

from typing import Iterable

from ghoshell_moss.depends import depend_ghost

depend_ghost()
from ghoshell_moss.message import Message, Text, Base64Image
from pydantic_ai.messages import ModelRequest, ModelResponse, ModelMessage, TextPart
from pydantic_ai import UserContent, TextContent, ImageUrl

__all__ = ["messages_to_parts", "moment_to_request", "moments_to_history"]


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


def moment_to_request(moment) -> ModelRequest:
    """将 Moment 转为 pydantic AI ModelRequest."""
    from ghoshell_moss.core.blueprint.mindflow import Moment as _Moment
    parts = messages_to_parts(moment.full_observation_messages())
    return ModelRequest(parts=parts)


def moments_to_history(moments) -> list[ModelMessage]:
    """将 Moments 轨迹重建为 pydantic AI message_history.

    只保留已完成回合 (带 logos), 跳过当前未完成帧. 动态消息由
    ``as_history_messages`` 天然丢弃, 不进历史 — 动态消息只在当前帧最新一份.
    """
    from ghoshell_moss.core.blueprint.moment import Moment

    history: list[ModelMessage] = []
    for inputs, logos in Moment.to_history_turns(moments):
        if logos is None:
            continue
        history.append(ModelRequest(parts=messages_to_parts(inputs)))
        history.append(ModelResponse(parts=[TextPart(content=logos)]))
    return history
