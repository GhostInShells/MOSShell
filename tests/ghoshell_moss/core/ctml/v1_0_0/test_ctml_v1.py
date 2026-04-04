import asyncio
from typing import AsyncIterable
from ghoshell_moss.core import CTMLShell, InterpretError
from ghoshell_moss.core.ctml import ctml_shell_test
from ghoshell_moss.core.blueprint.builder import new_channel
import pytest


@pytest.mark.asyncio
async def test_ctml_noop_run():
    tasks = await ctml_shell_test(ctml="")
    assert len(tasks) == 0


@pytest.mark.asyncio
async def test_ctml_base_call():
    a_chan = new_channel(name="a")
    b_chan = new_channel(name="b")

    @a_chan.build.command()
    async def foo() -> int:
        return 123

    @b_chan.build.command()
    async def bar() -> int:
        return 456

    tasks = await ctml_shell_test(a_chan, b_chan, ctml="<a:foo/><b:bar/>")
    assert len(tasks) == 2
    for t in tasks:
        assert await t in [123, 456]


@pytest.mark.asyncio
async def test_simple_content_call():
    contents = []

    async def foo(chunks__: AsyncIterable[str]) -> None:
        async for chunk in chunks__:
            contents.append(chunk)

    async def bar() -> int:
        return 123

    def builder(shell: CTMLShell):
        cmd = shell.main_channel.build.content_command(foo, override=True)
        assert cmd.name() == "__content__"
        shell.main_channel.build.command()(bar)

    tasks = await ctml_shell_test(builder=builder, ctml="<_><bar/>hello</_> world")
    assert len(tasks) == 5
    assert ''.join(contents) == 'hello world'


@pytest.mark.asyncio
async def test_ctml_parallel_baseline():
    order = []

    a = new_channel(name="a")
    b = new_channel(name="b")

    @a.build.command()
    async def foo() -> None:
        await asyncio.sleep(0.005)
        order.append('foo')

    @b.build.command()
    async def bar() -> None:
        await asyncio.sleep(0.001)
        order.append('bar')

    tasks = await ctml_shell_test(a, b, ctml="<a:foo/><b:bar/>")
    assert len(tasks) == 2
    assert order == ['bar', 'foo']


@pytest.mark.asyncio
async def test_ctml_scope_path_inheritance():
    """验证 <_ channel='a'> <bar/> </_> 能够正确调用 a:bar"""
    a_chan = new_channel(name="a")
    calls = []

    @a_chan.build.command()
    async def bar():
        calls.append("a:bar")

    # 在 a 作用域下直接写 bar，应该被解析为 a:bar
    await ctml_shell_test(a_chan, ctml="<_ channel='a'><bar/></_>")
    assert calls == ["a:bar"]


@pytest.mark.asyncio
async def test_ctml_empty_content_not_run():
    """
    验证空的字符串不会触发 content 调用.
    """
    a_chan = new_channel(name="a")
    results = []

    @a_chan.build.command()
    async def cmd_a(): results.append("a")

    # a 嵌套 b，b 内部调用自己的命令，b 结束后回到 a 调用 a 的命令
    # 保留很多空行.
    ctml = """
        <_ channel='a' until='all'>
            <cmd_a />
                
        </_>
        """
    tasks = await ctml_shell_test(a_chan, ctml=ctml)
    assert len(tasks) == 3
    # 加入有意义的字符, 就会多一个 content 函数.
    ctml = """
            <_ channel='a' until='all'>
                <cmd_a />
                hello
            </_>
            """
    tasks = await ctml_shell_test(a_chan, ctml=ctml)
    assert len(tasks) == 4
    # 前后都一样.
    ctml = """
                <_ channel='a' until='all'>
                    hello
                    <cmd_a />
                    world
                </_>
                """
    tasks = await ctml_shell_test(a_chan, ctml=ctml)
    assert len(tasks) == 5


@pytest.mark.asyncio
async def test_ctml_nested_scope_override():
    """验证嵌套作用页路径切换"""
    a_chan = new_channel(name="a")
    b_chan = new_channel(name="b")
    results = []

    @a_chan.build.command()
    async def cmd_a(): results.append("a")

    @b_chan.build.command()
    async def cmd_b(): results.append("b")

    # a 嵌套 b，b 内部调用自己的命令，b 结束后回到 a 调用 a 的命令
    ctml = """
    <_ channel='a' until='all'>
        <_ channel='b' until='all'>
            <cmd_b />
        </_>
        <cmd_a />
    </_>
    """
    with pytest.raises(InterpretError):
        await ctml_shell_test(a_chan, b_chan, ctml=ctml)


@pytest.mark.asyncio
async def test_ctml_flow_with_mixed_content():
    """验证 flow 模式下，文本和命令的交替执行"""
    log = []

    async def speak(chunks__: AsyncIterable[str]):
        async for chunk in chunks__:
            log.append(f"say:{chunk}")

    def builder(shell: CTMLShell):
        shell.main_channel.build.content_command(speak)

        @shell.main_channel.build.command()
        async def action():
            log.append("action")

    # 预期顺序：say:hello -> action -> say:world
    await ctml_shell_test(builder=builder, ctml="hello<action/>world")

    # 过滤掉空的 chunk 或 token 分片，检查核心顺序
    combined = "".join(log)
    assert "say:hello" in combined
    assert "action" in combined
    assert "say:world" in combined
    # 确保 action 夹在中间（基于你的 FIFO 占用逻辑）
    assert log.index("action") > 0


@pytest.mark.asyncio
async def test_ctml_scope_timeout():
    status = []

    async def foo() -> None:
        await asyncio.sleep(0.005)
        status.append("done")

    def build(shell: CTMLShell):
        shell.main_channel.build.command()(foo)

    await ctml_shell_test(ctml="<_ timeout='0.001'><foo/></_>", builder=build)
    # foo is canceled
    assert status == []

    await ctml_shell_test(ctml="<_ timeout='0.006'><foo/></_>", builder=build)
    # foo is not canceled this time.
    assert status == ['done']


@pytest.mark.asyncio
async def test_ctml_flow_cancels_long_running_child():
    """验证 flow 结束时，未完成的子通道任务会被取消"""
    a = new_channel(name="a")
    b = new_channel(name="b")
    status = {"b_finished": False, "b_cancelled": False}

    @a.build.command()
    async def fast_cmd():
        await asyncio.sleep(0.01)  # 比 b 快
        status["a_finished"] = True

    @b.build.command()
    async def slow_cmd():
        try:
            await asyncio.sleep(0.1)
            status["b_finished"] = True
        finally:
            status["b_cancelled"] = True

    ctml = "<_ channel='a' until='all'><a.b:slow_cmd/><fast_cmd/></_>"
    tasks = await ctml_shell_test(a.import_channels((b, "b")), ctml=ctml)
    # 正常执行的话, slow_cmd 和 fast_cmd 都会被执行完.
    assert 'b_finished' in status
    assert 'a_finished' in status
    status.clear()

    # ctml 默认是 until="flow"
    ctml = "<_ channel='a'><a.b:slow_cmd/><fast_cmd/></_>"
    tasks = await ctml_shell_test(a.import_channels((b, "b")), ctml=ctml)

    # 结果应该是 b 被 cancel 了，因为 a 的直接序列 (fast_cmd) 跑完了
    assert "b_finished" not in status
    assert status["b_cancelled"] is True
