"""
moss.message → dsh 消息体单向映射 (MOSS 上行内容容器 → dsh UserMessage).

ghoshell_moss.message 的 Message 本质是 anthropic content block 的 xml 容器: 上行给模型的内容,
`as_contents()` 把它展开成 anthropic 风格 Content 块 (text/image)。本模块把 MOSS Message 映射成
dsh 的 UserMessage (role 钉死 "user"), content 来自 as_contents() 逐块转 ContentBlock。

设计边界 (沿 ghoshell_moss.message "放弃全量协议支持" 的思路, 映射器保持薄):
- 返回 dsh 现有 Message 模型, role 传 "user", 不新建 UserMessage 类型。
- image 块: MOSS 内联 base64, dsh ImageBlock 需要 ImageAttachmentRef (attachment service 的
  存储引用), 纯映射器没有 attachmentId 造不出 ref —— 抛明确错误, 不静默吞。后续接 attachment
  注册回调或走 session.prompt (host 提升 base64) 时再落地。
"""

from __future__ import annotations

from ghoshell_moss.deepseek_harness.types.session_events import ContentBlock, Message, MessageSource
from ghoshell_moss.message import Message as MossMessage
from ghoshell_moss.message.contents.abcd import Content

__all__ = ["to_user_message", "to_content_block"]


def to_content_block(content: Content) -> ContentBlock:
    """把 MOSS anthropic 风格 Content 块映射成 dsh ContentBlock."""
    ctype = content.get("type")
    if ctype == "text":
        return ContentBlock(type="text", text=content.get("text"))
    if ctype == "image":
        raise NotImplementedError(
            "MOSS image content 是内联 base64 (source), dsh ContentBlock.image 需要 "
            "ImageAttachmentRef; 先注册到 attachment service 拿 ref, 或走 session.prompt "
            "让 host 提升 base64."
        )
    # 未知块透传: dsh ContentBlock merge-extensible (extra="allow").
    return ContentBlock.model_validate(content)


def to_user_message(
    message: MossMessage,
    *,
    with_meta: bool = True,
    timestamp: bool = True,
    join_text: bool = True,
) -> Message:
    """把 MOSS Message (上行内容容器) 映射成 dsh UserMessage (role=user)."""
    blocks = [
        to_content_block(c)
        for c in message.as_contents(with_meta=with_meta, timestamp=timestamp, join_text=join_text)
    ]
    return Message(role="user", content=blocks, source=MessageSource(kind="user"))
