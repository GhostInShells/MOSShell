from collections.abc import Iterable

from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
    ImageURL,
)
from openai.types.chat.chat_completion_content_part_text_param import ChatCompletionContentPartTextParam
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)

from ghoshell_moss.message import contents
from ghoshell_moss.message.abcd import Message

__all__ = ["parse_message_to_chat_completion_param", "parse_messages_to_params"]


def parse_messages_to_params(messages: Iterable[Message]) -> list[dict]:
    result = []
    for message in messages:
        got = parse_message_to_chat_completion_param(message)
        if len(got) > 0:
            result.extend(got)
    return result


def parse_message_to_chat_completion_param(
    message: Message,
    system_user_name: str = "__moss_system__",
) -> list[dict]:
    message = message.as_completed()
    if len(message.contents) == 0:
        return []

    content_parts = []
    has_media = False
    for content in message.contents:
        if text := contents.Text.from_content(content):
            content_parts.append(
                ChatCompletionContentPartTextParam(
                    text=text.text,
                    type="text",
                )
            )
        elif image_url := contents.ImageUrl.from_content(content):
            has_media = True
            content_parts.append(
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=ImageURL(
                        url=image_url.url,
                        detail="auto",
                    ),
                )
            )
        elif base64_image := contents.Base64Image.from_content(content):
            has_media = True
            content_parts.append(
                ChatCompletionContentPartImageParam(
                    type="image_url",
                    image_url=ImageURL(
                        url=base64_image.data_url,
                        detail="auto",
                    ),
                )
            )
    if len(content_parts) == 0:
        return []

    if message.role == "assistant":
        item = ChatCompletionAssistantMessageParam(
            role="assistant",
            content=content_parts,
        )
    elif message.role == "user":
        item = ChatCompletionUserMessageParam(
            role="user",
            content=content_parts,
        )
    elif not has_media:
        item = ChatCompletionSystemMessageParam(
            role="system",
            content=content_parts,
        )
    else:
        item = ChatCompletionUserMessageParam(
            role="user",
            name=system_user_name,
            content=content_parts,
        )

    if message.meta.name:
        item["name"] = message.meta.name

    return [item]
