from typing_extensions import Self

from pydantic import Field

from ghoshell_moss.message.abcd import ContentModel, DeltaModel, Delta

__all__ = ["Text", "TextDelta"]

"""
自带的常用多模态消息体类型. 
"""


class Text(ContentModel):
    """
    最基础的文本类型.
    """

    CONTENT_TYPE = "text"
    text: str = Field(
        default="",
        description="Text of the message",
    )

    @classmethod
    def new(cls, text: str) -> "Text":
        return cls(text=text)

    @classmethod
    def from_delta(cls, delta: Delta | DeltaModel) -> Self | None:
        if isinstance(delta, Delta):
            model = TextDelta.from_delta(delta)
        else:
            model = delta
        return cls(text=model.text)

    def buffer_delta(self, delta: Delta | DeltaModel) -> bool:
        if isinstance(delta, Delta):
            model = TextDelta.from_delta(delta)
        else:
            model = delta
        if model and isinstance(model, TextDelta):
            self.text += model.text
            return True
        return False


class TextDelta(DeltaModel):
    DELTA_TYPE = "text"

    content: str = Field(
        default="",
        description="The text of the delta",
    )
