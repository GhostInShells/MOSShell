import pytest

from ghoshell_moss import Message, new_chan, new_ctml_shell
from ghoshell_moss.core.llm.shell_driver import ShellTurnRequest, shell_turn, with_shell


@pytest.mark.asyncio
async def test_shell_turn_decorator_accepts_message_list():
    shell = new_ctml_shell()
    chan = new_chan("a")
    shell.main_channel.import_channels(chan)

    @chan.build.command()
    async def ping() -> int:
        return 1

    def fake_llm(messages: list[Message], /, **kwargs):
        async def _stream():
            yield "<a:ping />"

        return _stream()

    @shell_turn(shell, llm=fake_llm)
    async def respond(text: str):
        return [Message.new(role="user").with_content(text)]

    async with shell:
        result = await respond("hi")
        assert "a:ping" in result.executed_tokens


@pytest.mark.asyncio
async def test_shell_turn_decorator_accepts_request_object():
    shell = new_ctml_shell()
    chan = new_chan("a")
    shell.main_channel.import_channels(chan)

    @chan.build.command()
    async def add(a: int, b: int) -> int:
        return a + b

    def fake_llm(messages: list[Message], /, **kwargs):
        async def _stream():
            yield '<a:add a="1" b="2" />'

        return _stream()

    @shell_turn(shell, llm=fake_llm)
    def respond_with_history() -> ShellTurnRequest:
        history = [Message.new(role="assistant").with_content("previous")]
        inputs = [Message.new(role="user").with_content("now")]
        return ShellTurnRequest(inputs=inputs, history=history)

    async with shell:
        result = await respond_with_history()
        assert "a:add" in result.executed_tokens
        assert any(
            m.name in {"a:add", "__command_result__"} and "3" in m.to_json() for m in result.interpretation.messages
        )


@pytest.mark.asyncio
async def test_with_shell_runner_is_directly_callable():
    shell = new_ctml_shell()
    chan = new_chan("a")
    shell.main_channel.import_channels(chan)

    @chan.build.command()
    async def ping() -> int:
        return 1

    def fake_llm(messages: list[Message], /, **kwargs):
        async def _stream():
            yield "<a:ping />"

        return _stream()

    runner = with_shell(shell, llm=fake_llm)

    async with shell:
        result = await runner([Message.new(role="user").with_content("hi")])

        assert "a:ping" in result.executed_tokens
