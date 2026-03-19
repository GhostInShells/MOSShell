from typing_extensions import Self

from pydantic import Field

from ghoshell_moss.message.abcd import ContentModel, Content

__all__ = ["Text"]

"""
自带的常用多模态消息体类型. 
"""


class Text(ContentModel):
    """
    最基础的文本类型. 经过多轮改造, 保留用于兼容一些历史单测.
    """

    text: str = Field(
        default="",
        description="Text of the message",
    )

    @classmethod
    def new(cls, text: str) -> Self:
        return cls(text=text)

    def to_content(self) -> Content:
        return self.text
