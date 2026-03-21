from ghoshell_moss.core import new_ctml_shell
import pytest


@pytest.mark.asyncio
async def test_shell_parse_token_baseline():
    shell = new_ctml_shell()
    async with shell:
        tokens = []

        async def generate():
            _content = "<wait><a:foo/>hello</wait><wait><a:foo/>world</wait>"
            for c in _content:
                yield c

        async for token in shell.parse_text_to_command_tokens(generate()):
            tokens.append(token)
        assert tokens[0].seq == "start"
        assert tokens[0].name == "ctml"
        assert tokens[-1].seq == "end"
        assert tokens[-1].name == "ctml"
        assert tokens[0].command_id() == tokens[-1].command_id()

        content = "hello world"
        tasks = []
        async for task in shell.parse_text_to_tasks(content):
            tasks.append(task)
        assert len(tasks) == 1

        async with await shell.interpreter() as interpreter:
            interpreter.feed(content)
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            assert len(tasks) == 1
            assert list(tasks.values())[0].tokens == content
