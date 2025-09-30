from ghoshell_moss.shell.shell_impl import ShellImpl
import pytest


@pytest.mark.asyncio
async def test_shell_parse_tokens_baseline():
    shell = ShellImpl()
    async with shell:
        tokens = []
        async for token in shell.parse_tokens("<foo />"):
            tokens.append(token)
        assert len(tokens) == 4


@pytest.mark.asyncio
async def test_shell_parse_tasks_baseline():
    shell = ShellImpl()
    async with shell:
        tasks = []
        async for token in shell.parse_tasks("<foo>hello</bar>"):
            tasks.append(token)
        assert len(tasks) == 3
