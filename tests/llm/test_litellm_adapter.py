import pytest

from ghoshell_moss import Message
from ghoshell_moss_contrib.agent.shell_litellm import litellm_text_streamer


class _FakeDelta:
    def __init__(self, content: str | None):
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None):
        self.delta = _FakeDelta(content)


class _FakeChunk:
    def __init__(self, content: str | None):
        self.choices = [_FakeChoice(content)]


@pytest.mark.asyncio
async def test_litellm_text_streamer_yields_content_deltas():
    async def fake_acompletion(**kwargs):
        assert kwargs.get("stream") is True
        assert isinstance(kwargs.get("messages"), list)

        async def _stream():
            yield _FakeChunk("hello")
            yield _FakeChunk(" ")
            yield _FakeChunk("world")

        return _stream()

    streamer = litellm_text_streamer(acompletion=fake_acompletion)
    messages = [Message.new(role="user").with_content("hi")]

    got = []
    async for delta in streamer(messages):
        got.append(delta)

    assert "".join(got) == "hello world"
