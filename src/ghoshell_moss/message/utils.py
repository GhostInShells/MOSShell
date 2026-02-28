from .abcd import Message, MessageMeta, Role
from .contents import Text

__all__ = [
    "new_text_message",
]


def new_text_message(content: str, *, role: str | Role = "") -> Message:
    """
    创建一个系统消息.
    """
    meta = MessageMeta(role=str(role))
    obj = Text(text=content)
    # Ensure the message is not already marked as `completed`, otherwise
    # `Message.as_completed()` may no-op depending on implementation.
    return Message(meta=meta, seq="head").as_completed([obj.to_content()])
