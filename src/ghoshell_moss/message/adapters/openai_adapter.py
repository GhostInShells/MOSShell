from typing import Iterable, Any
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
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
from ghoshell_moss.message.abcd import Message, MessageAdapter, MessageMeta

__all__ = ["parse_message_to_chat_completion_param", "parse_messages_to_params"]


class OpenAIParamsAdapter(MessageAdapter[ChatCompletionMessageParam]):
    """
    OpenAI params 协议转换.
    """

    @classmethod
    def protocol(cls) -> str:
        return 'openai'

    def raw_to_message(self, raw: ChatCompletionMessageParam) -> Message:
        return Message.from_raw(
            meta=MessageMeta(
                role=raw['role'],
                name=raw['name'],
            ),
            raw_data=raw,
            type='',
            protocol=self.protocol(),
        )

    def message_to_raw(self, message: Message) -> ChatCompletionMessageParam | None:
        if message.protocol == "":
            got = parse_message_to_chat_completion_param(message)
            if len(got) > 0:
                return got[0]
            return None
        elif message.protocol == self.protocol():
            return message.raw
        return None


def parse_messages_to_params(messages: Iterable[Message | Any]) -> list[ChatCompletionMessageParam]:
    result = []
    for message in messages:
        if isinstance(message, Message):
            got = parse_message_to_chat_completion_param(message)
            if len(got) > 0:
                result.extend(got)
        else:
            result.append(message)
    return result


def parse_message_to_chat_completion_param(
        message: Message,
        system_user_name: str = "__moss_system__",
) -> list[dict]:
    message = message
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
