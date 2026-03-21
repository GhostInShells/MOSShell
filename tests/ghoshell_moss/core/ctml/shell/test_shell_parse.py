import pytest

from ghoshell_moss.core.ctml.shell.ctml_shell import CTMLShell
from ghoshell_moss.core.concepts.errors import InterpretError


@pytest.mark.asyncio
async def test_shell_parse_tokens_baseline():
    shell = CTMLShell()

    async def foo():
        pass

    shell.main_channel.build.command()(foo)
    async with shell:
        assert shell.is_running()
        tokens = []
        async for token in shell.parse_text_to_command_tokens("<foo />"):
            tokens.append(token)
        assert len(tokens) == 4

        tasks = []
        with pytest.raises(InterpretError):
            async for task in shell.parse_text_to_tasks("<bar />"):
                tasks.append(task)


@pytest.mark.asyncio
async def test_shell_parse_tasks_baseline():
    shell = CTMLShell()
    async with shell:
        tasks = []
        async for token in shell.parse_text_to_tasks("<foo>hello</foo><bar/>", ignore_wrong_command=True):
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


@pytest.mark.asyncio
async def test_shell_attrs_parsing():
    shell = CTMLShell()

    @shell.main_channel.build.command()
    async def foo(f: float | None, i: int, b: bool, c: list, d: dict):
        return f, i, b, c, d

    async with shell:
        assert shell.is_running()
        async with await shell.interpreter() as interpreter:
            interpreter.feed("<foo f='0.2' i='1' b='False' c='[1, 2]' d='{}'/>")
            interpreter.commit()
            tasks = await interpreter.wait_tasks()
            interpreter.raise_exception()
            assert len(tasks) == 1
            task = list(tasks.values())[0]
            assert await task == (0.2, 1, False, [1, 2], {})
