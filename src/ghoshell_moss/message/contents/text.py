from typing_extensions import Self

from pydantic import Field

from ghoshell_moss.message.contents.abcd import ContentModel

__all__ = ["Text"]

"""
自带的常用多模态消息体类型. 
"""


class Text(ContentModel):
    """
    text model for text block
    """
    text: str = Field(
        description="the text value"
    )

    @classmethod
    def new(cls, text: str) -> Self:
        return cls(text=text)

    @classmethod
    def content_type(cls) -> str:
        return 'text'
