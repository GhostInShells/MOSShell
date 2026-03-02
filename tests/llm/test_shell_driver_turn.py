import pytest

from ghoshell_moss import Message, new_chan, new_ctml_shell
from ghoshell_moss.core.llm.shell_driver import run_shell_turn


@pytest.mark.asyncio
async def test_run_shell_turn_executes_commands_and_collects_messages():
    shell = new_ctml_shell()
    chan = new_chan("a")
    shell.main_channel.import_channels(chan)

    @chan.build.command()
    async def add(a: int, b: int) -> int:
        return a + b

    def fake_llm(messages: list[Message], /, **kwargs):
        async def _stream():
            # One normal text chunk and one command chunk.
            yield "hello "
            yield '<a:add a="2" b="3" />'

        return _stream()

    async with shell:
        result = await run_shell_turn(
            shell,
            llm=fake_llm,
            inputs=[Message.new(role="user").with_content("hi").as_completed()],
            history=[],
        )

        assert "hello" in result.assistant_text
        assert "a:add" in result.executed_tokens
        # command result should be observable in interpretation.messages
        assert any(
            m.name == "a:add" and "5" in m.to_json() for m in result.interpretation.messages
        )
