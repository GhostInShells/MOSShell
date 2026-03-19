from ghoshell_moss.message.abcd import Message, MessageMeta
from ghoshell_moss.message.contents import Text

__all__ = [
    "new_text_message",
]


def new_text_message(content: str, *, role: str = "") -> Message:
    """
    创建一个系统消息. 由于经过很多改造, 暂时没啥用. 先为了单测保留.
    """
    meta = MessageMeta(role=str(role))
    obj = Text(text=content)
    return Message(meta=meta).with_content(obj)
