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
    return Message(meta=meta).as_completed([obj.to_content()])
