import os
from anthropic import Anthropic
from anthropic.types import ContentBlock, ContentBlockParam, TextBlockParam

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),  # This is the default and can be omitted
)

if __name__ == '__main__':
    print(type(TextBlockParam), type(TextBlockParam(text="hello world")))
    # message = client.messages.create(
    #     max_tokens=1024,
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": "Hello, Claude",
    #         }
    #     ],
    #     model="claude-opus-4-6",
    # )
    # print(message.content)
