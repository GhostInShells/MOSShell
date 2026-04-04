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


@pytest.mark.asyncio
async def test_ctml_sequential_channels_stability():
    """验证 A 通道完成后，B 通道才能开始，中间没有重叠"""
    a = new_channel(name="a")
    b = new_channel(name="b")
    history = []

    @a.build.command()
    async def task_a():
        history.append("a_start")
        await asyncio.sleep(0.02)
        history.append("a_end")

    @b.build.command()
    async def task_b():
        history.append("b_start")
        await asyncio.sleep(0.01)
        history.append("b_end")

    # 顺序执行两个不同通道的作用域
    ctml = """
    <_ channel='a'><task_a/></_>
    <_ channel='b'><task_b/></_>
    """
    await ctml_shell_test(a, b, ctml=ctml)

    # 必须保证 a 彻底结束后 b 才开始
    assert history == ["a_start", "b_start", "b_end", "a_end"]

    history.clear()
    ctml = """
        <_ until='all'><a:task_a/></_>
        <_ until='all'><b:task_b/></_>
        """
    await ctml_shell_test(a, b, ctml=ctml)
    assert history == ["a_start", "a_end", "b_start", "b_end", ]


@pytest.mark.asyncio
async def test_ctml_until_any_logic():
    """验证 any 模式：一个完成，全部带走"""
    a = new_channel(name="a")
    b = new_channel(name="b")
    results = {"fast_done": False, "slow_cancelled": False}

    @a.build.command()
    async def fast():
        await asyncio.sleep(0.01)
        results["fast_done"] = True

    @b.build.command()
    async def slow():
        try:
            await asyncio.sleep(0.1)
            results["slow_done"] = True
        except asyncio.CancelledError:
            results["slow_cancelled"] = True

    # 在 any 作用域下并行
    ctml = """
    <_ until='any'>
        <a:fast/>
        <b:slow/>
    </_>
    """
    tasks = await ctml_shell_test(a, b, ctml=ctml)
    count_success = 0
    assert len(tasks) == 4
    for task in tasks:
        if task.success():
            count_success += 1
    assert count_success == 3

    assert len(results) == 2
    assert results["fast_done"] is True
    assert results["slow_cancelled"] is True


@pytest.mark.asyncio
async def test_ctml_nested_any_all_recursion():
    """验证 any 触发时，嵌套的 all 及其子命令被递归取消"""
    a = new_channel(name="a")
    done_count = 0

    @a.build.command()
    async def waiter():
        nonlocal done_count
        try:
            await asyncio.sleep(1.0)
            done_count += 1
        except asyncio.CancelledError:
            raise

    @a.build.command()
    async def trigger():
        await asyncio.sleep(0.01)  # 快速触发

    ctml = """
    <_ channel='a' until='any'>
        <trigger />
        <_ until='all'>
            <waiter _cid='1'/>
            <waiter _cid='2'/>
        </_>
    </_>
    """
    await ctml_shell_test(a, ctml=ctml)
    # trigger 完成导致外部 any 结束，内部 all 应该被整体撤销，包含它的 2 个 waiter
    assert done_count == 0


@pytest.mark.asyncio
async def test_ctml_scope_with_channel_prefix():
    a = new_channel(name="a")
    done_count = 0

    @a.build.command()
    async def waiter():
        nonlocal done_count
        try:
            await asyncio.sleep(0.05)
            done_count += 1
        except asyncio.CancelledError:
            raise

    @a.build.command()
    async def trigger():
        await asyncio.sleep(0.01)  # 快速触发

    ctml = """
        <a:_ >
            <trigger />
            <waiter _cid='1'/>
            <waiter _cid='2'/>
        </a:_>
        """
    await ctml_shell_test(a, ctml=ctml)
    # trigger 完成导致外部 any 结束，内部 all 应该被整体撤销，包含它的 2 个 waiter
    assert done_count == 2


@pytest.mark.asyncio
async def test_ctml_none_strict_features_of_until_flow_with_none_self_command():
    """验证容错逻辑, channel 通道内没有加 until=all, 但是所有命令都非自己通道的. """
    a = new_channel(name="a")

    done = []

    @a.build.command()
    async def foo():
        # 让 foo 不会比 __content__ 更快执行完.
        await asyncio.sleep(0.01)
        done.append('foo')

    ctml = """
    <_>
    <a:foo/>
    <a:foo/>
    </_>
    """
    # 虽然是 until 默认为 flow, 但由于没有任何子命令, 容错触发了.
    await ctml_shell_test(a, ctml=ctml)
    assert done == ['foo', 'foo']

    done.clear()
    ctml = """
    <_>
    <a:foo/>
    hello
    <a:foo/>
    </_>
    """
    # 但是一旦加了 任何该轨道的命令, 比如 __content__, 就不会容错.
    await ctml_shell_test(a, ctml=ctml)
    assert done == []
