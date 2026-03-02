from collections.abc import AsyncIterable
from typing import Any

from ghoshell_moss.core.llm.shell_driver import ChatTextStreamer
from ghoshell_moss.message import Message
from ghoshell_moss.message.adapters.openai_adapter import parse_messages_to_params
from ghoshell_moss_contrib.agent.depends import check_agent

__all__ = [
    "litellm_text_streamer",
]


def litellm_text_streamer(
    *,
    acompletion: Any | None = None,
) -> ChatTextStreamer:
    """Create a LiteLLM-backed text streamer.

    It adapts ghoshell `Message` objects into OpenAI-compatible message dicts,
    then yields `delta.content` strings from LiteLLM's streaming response.
    """

    litellm_acompletion = None
    if acompletion is None:
        check_agent()
        import litellm

        litellm_acompletion = litellm.acompletion

    def _streamer(messages: list[Message], /, **kwargs: Any) -> AsyncIterable[str]:
        async def _iter() -> AsyncIterable[str]:
            _acompletion = acompletion or litellm_acompletion
            params = dict(kwargs)
            params["messages"] = parse_messages_to_params(messages)
            params["stream"] = True
            response_stream = await _acompletion(**params)
            async for chunk in response_stream:
                delta = chunk.choices[0].delta
                content = getattr(delta, "content", None)
                if content:
                    yield content

        return _iter()

    return _streamer
