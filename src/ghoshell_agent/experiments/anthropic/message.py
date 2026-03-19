from pydantic import BaseModel, Field
from anthropic.types import ContentBlock, TextBlock, ThinkingBlock, TextBlockParam
from pydantic_ai import ModelMessage, TextPart, ModelRequest


class Foo(BaseModel):
    contents: list[ContentBlock] = Field(
        default_factory=list,
    )
    text: TextPart | None = Field(
        default=None
    )


if __name__ == "__main__":
    foo = Foo(
        contents=[
            TextBlock(text='Hello World', type='text'),
            ThinkingBlock(thinking='Hello World', signature="hello", type='thinking'),
        ],
        text=TextPart(content="hello"),
    )

    print(foo)
