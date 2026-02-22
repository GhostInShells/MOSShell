import pytest

from ghoshell_moss.core.shell.shell_impl import CTMLShell


@pytest.mark.asyncio
async def test_shell_parse_tokens_baseline():
    shell = CTMLShell()
    async with shell:
        assert shell.is_running()
        tokens = []
        async for token in shell.parse_text_to_command_tokens("<foo />"):
            tokens.append(token)
        assert len(tokens) == 4


@pytest.mark.asyncio
async def test_shell_parse_tasks_baseline():
    shell = CTMLShell()
    async with shell:
        tasks = []
        async for token in shell.parse_text_to_tasks("<foo>hello</foo><bar/>"):
            tasks.append(token)
        # 只生成了 1 个, 因为 foo 和 bar 函数都不存在.
        assert len(tasks) == 1


@pytest.mark.asyncio
async def test_shell_parse_tokens_to_tasks():
    shell = CTMLShell()

    @shell.main_channel.build.command()
    async def foo():
        return 123

    async with shell:
        assert shell.is_running()
        got = []
        tokens = shell.parse_text_to_command_tokens("<foo/>hello<foo/>")
        tasks = shell.parse_tokens_to_command_tasks(tokens)
        async for t in tasks:
            got.append(t)
        assert len(got) == 3
