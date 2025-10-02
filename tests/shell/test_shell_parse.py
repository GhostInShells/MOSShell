from ghoshell_moss.shell.impl import DefaultShell
import pytest


@pytest.mark.asyncio
async def test_shell_parse_tokens_baseline():
    shell = DefaultShell()
    async with shell:
        assert shell.is_running()
        tokens = []
        async for token in shell.parse_text_to_tokens("<foo />"):
            tokens.append(token)
        assert len(tokens) == 4


@pytest.mark.asyncio
async def test_shell_parse_tasks_baseline():
    shell = DefaultShell()
    async with shell:
        tasks = []
        async for token in shell.parse_text_to_tasks("<foo>hello</foo><bar/>"):
            tasks.append(token)
        # 只生成了 1 个, 因为 foo 和 bar 函数都不存在.
        assert len(tasks) == 1
