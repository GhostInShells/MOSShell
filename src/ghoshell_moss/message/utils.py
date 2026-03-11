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


def merge_done_messages(messages: list[Message]) -> list[Message]:
    """
    简单过滤, 并且合并相同类型消息体, 只保留完成后的尾包.
    不知道这样做是否有任何收益.
    """
    last_message = None
    result = []
    for message in messages:
        if not message.is_done():
            # 丢弃非尾包.
            continue
        elif last_message is None:
            # 设置 last.
            last_message = message.get_copy()
            continue
        elif last_message.meta.id == message.meta.id:
            # 是同一个消息体, 采取替换逻辑.
            # 按时序, 先来后到.
            last_message = message.get_copy()
            continue
        elif len(last_message.contents) == 0:
            # 空消息跳过.
            last_message = message.get_copy()
            continue
        # 相同类型的消息. 我们认为可以合并.
        elif last_message.name == message.name and last_message.role == message.role:
            # 增加 contents, 叠在一起.
            last_message.contents.extend(message.contents)
        else:
            result.append(last_message)
            last_message = message.get_copy()
    if last_message is not None:
        result.append(last_message)
    return result
