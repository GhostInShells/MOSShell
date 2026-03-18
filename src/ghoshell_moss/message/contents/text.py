from typing_extensions import Self

from pydantic import Field

from ghoshell_moss.message.abcd import ContentModel

__all__ = ["Text"]

"""
自带的常用多模态消息体类型. 
"""


class Text(ContentModel):
    """
    最基础的文本类型.
    """

    text: str = Field(
        default="",
        description="Text of the message",
    )

    @classmethod
    def new(cls, text: str) -> Self:
        return cls(text=text)

    @classmethod
    def content_type(cls) -> str:
        return 'text'

    def marshal(self) -> str:
        return self.text

    @classmethod
    def unmarshal(cls, content: str) -> dict:
        return {'text': content}
