from typing_extensions import Self

from pydantic import Field

from ghoshell_moss.message.contents.abcd import ContentModel, Content

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
        if isinstance(text, str):
            return cls.model_construct(text=text)
        else:
            return cls(text=text)

    @classmethod
    def new_content(cls, text: str) -> Content:
        return Content(text=text, type=cls.content_type())

    @classmethod
    def content_type(cls) -> str:
        return 'text'
