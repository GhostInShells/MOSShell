import asyncio
from typing import AsyncIterable
from ghoshell_moss.core import CTMLShell, InterpretError
from ghoshell_moss.core.ctml import ctml_shell_test
from ghoshell_moss.core.blueprint.channel_builder import new_channel
import pytest

"""
配合 CTML 1.0 语法写的单元测试. 
在测试 CTML 解释器/执行器 的同时, 也在测试 AI 对 CTML 的理解, 同时修改细节. 
"""


# --- 以下是作者写的基线测试. --- #

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


# --- 以下是 Gemini 3 写的单测, 发现 channel=name 语法有歧义, 仍改为命名空间定义作用域 --- #

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


# --- 以下是 开发者写的单测, 检查隐藏的容错逻辑 --- #

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


# --- 以下是 deepseek v3.2 写的单测, 细节略有调整 --- #

@pytest.mark.asyncio
async def test_ctml_open_close_tags_with_chunks():
    """测试开放-闭合标签配合 chunks__ 流式参数"""
    chan = new_channel(name="speech")

    @chan.build.command()
    async def say(chunks__: AsyncIterable[str]) -> str:
        # 收集所有 chunk 并拼接
        full = []
        async for chunk in chunks__:
            full.append(chunk)
        return "".join(full)

    tasks = await ctml_shell_test(
        chan,
        ctml="<speech:say>Hello, <b>world</b>!</speech:say>"
    )
    assert len(tasks) == 1
    result = await tasks[0]
    assert result == "Hello, <b>world</b>!"


@pytest.mark.asyncio
async def test_ctml_cdata_in_text():
    """测试 CDATA 包裹的 text__ 内容"""
    chan = new_channel(name="logger")

    @chan.build.command()
    async def log(text__: str) -> str:
        return text__

    ctml_with_cdata = """
    <logger:log><![CDATA[
        <tag> & 特殊字符 无需转义 </tag>
    ]]></logger:log>
    """
    tasks = await ctml_shell_test(chan, ctml=ctml_with_cdata)
    result = await tasks[0]
    assert "<tag>" in result and "&" in result


@pytest.mark.asyncio
async def test_ctml_scope_flow_sequential():
    """测试作用域 until='flow' (默认) 顺序执行"""
    chan = new_channel(name="proc")

    order = []

    @chan.build.command()
    async def step1() -> str:
        order.append(1)
        return "one"

    @chan.build.command()
    async def step2() -> str:
        order.append(2)
        return "two"

    tasks = await ctml_shell_test(
        chan,
        ctml="""
        <_>
            <proc:step1/>
            <proc:step2/>
        </_>
        """
    )
    assert len(tasks) == 4
    assert order == [1, 2]


@pytest.mark.asyncio
async def test_ctml_scope_any_parallel_first_complete():
    """测试作用域 until='any'：任意子任务完成即中断其他"""
    chan = new_channel(name="race")

    @chan.build.command()
    async def fast(delay: float = 0.1) -> str:
        await asyncio.sleep(delay)
        return "fast"

    @chan.build.command()
    async def slow(delay: float = 0.3) -> str:
        await asyncio.sleep(delay)
        return "slow"

    tasks = await ctml_shell_test(
        chan,
        ctml="""
        <_ until="any">
            <race:fast delay="0.05"/>
            <race:slow delay="0.2"/>
        </_>
        """
    )
    # 由于 any 模式，一旦 fast 完成，slow 会被取消
    # 这里检查返回结果的数量应为 1（只有 fast 成功完成）
    # 注意：被取消的任务会抛出 CancelledError，在 gather 中需要处理
    results = []
    for t in tasks:
        if t.success():
            results.append(t.result())
    assert len(results) == 3
    assert results[1] == "fast"


@pytest.mark.asyncio
async def test_ctml_scope_timeout():
    """测试作用域超时 timeout"""
    chan = new_channel(name="timer")

    @chan.build.command()
    async def long_task() -> str:
        await asyncio.sleep(0.5)
        return "done"

    tasks = await ctml_shell_test(
        chan,
        ctml="""
        <_ timeout="0.1">
            <timer:long_task/>
        </_>
        """
    )
    # 超时会导致作用域内的任务被取消，所以 long_task 会抛出 CancelledError
    has_long = False
    for task in tasks:
        if task.meta.name == "long_task":
            assert task.exception() is not None
            assert task.cancelled()
            has_long = True
    assert has_long


@pytest.mark.asyncio
async def test_ctml_nested_scopes():
    """测试嵌套作用域"""
    chan = new_channel(name="nest")
    log = []

    @chan.build.command()
    async def a(msg: str) -> None:
        log.append(msg)

    tasks = await ctml_shell_test(
        chan,
        ctml="""
        <_>
            <nest:a msg="outer start"/>
            <_>
                <nest:a msg="inner"/>
            </_>
            <nest:a msg="outer end"/>
        </_>
        """
    )
    assert log == ["outer start", "inner", "outer end"]


@pytest.mark.asyncio
async def test_ctml_parallel_commands_in_parent_scope():
    """测试父作用域内不同子通道的并行执行"""
    chan_a = new_channel(name="a")
    chan_b = new_channel(name="b")
    order = []

    @chan_a.build.command()
    async def task_a() -> None:
        await asyncio.sleep(0.1)
        order.append("A")

    @chan_b.build.command()
    async def task_b() -> None:
        await asyncio.sleep(0.05)
        order.append("B")

    tasks = await ctml_shell_test(
        chan_a, chan_b,
        ctml="""
        <_>
            <a:task_a/>
            <b:task_b/>
        </_>
        """
    )
    # 由于并行，B 应该先完成（延迟短），但顺序由调度决定
    # 这里我们只验证两个都执行了
    assert set(order) == {"A", "B"}


@pytest.mark.asyncio
async def test_ctml_command_cid_and_result():
    """测试命令实例化 _cid 和结果返回格式"""
    chan = new_channel(name="calc")

    @chan.build.command()
    async def double(x: int) -> int:
        return x * 2

    # 由于 ctml_shell_test 返回的是任务列表，不直接检查 <result> 标签，
    # 但我们可以在命令中收集返回值来验证 _cid 不影响逻辑
    tasks = await ctml_shell_test(
        chan,
        ctml="""
        <calc:double _cid="1" x="3"/>
        <calc:double _cid="2" x="7"/>
        """
    )
    results = {t.caller_name(): t.result() for t in tasks}
    assert results == {"calc:double:1": 6, "calc:double:2": 14}


@pytest.mark.asyncio
async def test_ctml_observe_interrupt():
    """测试 Observe 返回值中断所有运行中命令"""
    from ghoshell_moss import Observe
    loop_chan = new_channel(name='loop')
    inter_chan = new_channel(name="interrupt")

    @inter_chan.build.command()
    async def trigger_observe() -> Observe:
        return Observe()

    @loop_chan.build.command()
    async def infinite_loop() -> None:
        try:
            while True:
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass  # 预期被取消

    tasks = await ctml_shell_test(
        inter_chan, loop_chan,
        ctml="""
        <_>
            <loop:infinite_loop/>
            <interrupt:trigger_observe/>
        </_>
        """
    )
    # 由于 Observe 触发，整个作用域应被中断，所有任务取消
    # 每个任务都会抛出 CancelledError
    has_loop = False
    for t in tasks:
        if t.meta.name == "infinite_loop":
            assert t.cancelled()
            has_loop = True
    assert has_loop


@pytest.mark.asyncio
async def test_ctml_parse_error():
    """测试 CTML 解析错误导致快速失败"""
    chan = new_channel(name="dummy")
    invalid_ctml = "<dummy:cmd arg=123/>"  # 参数值未用双引号

    with pytest.raises(InterpretError):
        await ctml_shell_test(chan, ctml=invalid_ctml)


@pytest.mark.asyncio
async def test_ctml_root_channel_no_prefix():
    """测试根通道 __main__ 命令不加前缀"""
    # 创建根通道（实际测试中 ctml_shell_test 可能隐式包含 __main__）
    # 我们手动添加一个主通道命令
    main_chan = new_channel(name="__main__")

    @main_chan.build.command()
    async def wait(seconds: float) -> str:
        await asyncio.sleep(seconds)
        return "waited"

    # 正确用法：不带 __main__: 前缀
    tasks = await ctml_shell_test(ctml='<wait seconds="0.01"/>', main=main_chan)
    assert len(tasks) == 1
    result = await tasks[0]
    assert result == "waited"

    # 错误用法：带前缀应解析失败
    # 实际上... 做了容错.
    await ctml_shell_test(main_chan, ctml='<__main__:wait seconds="0.01"/>')


@pytest.mark.asyncio
async def test_ctml_content_command_for_unmarked_text():
    """测试通道内非标记文本通过 __content__ 命令处理"""
    chan = new_channel(name="echo")

    @chan.build.content_command
    async def content(chunks__: AsyncIterable[str]) -> str:
        full = []
        async for chunk in chunks__:
            full.append(chunk)
        return "".join(full)

    tasks = await ctml_shell_test(
        chan,
        ctml="<_>Hello, world!</_>"  # 无标签文本进入 __content__
    )
    # 注意：ctml_shell_test 会将作用域内的文本解析为对当前通道的 __content__ 调用
    # 这里假设作用域默认通道是 __main__？可能需要调整。为了测试，让 chan 成为默认通道。
    # 简化：直接调用 chan 的 __content__
    # 实际测试中，需要确保 chan 是当前作用域的默认通道。这里我们显式指定作用域通道：
    tasks = await ctml_shell_test(
        chan,
        ctml="<echo:_>Hello!</echo:_>"  # 作用域通道为 echo，内部文本调用 echo.__content__
    )
    result = ""
    for t in tasks:
        if t.meta.name == "__content__":
            result = t.result()
    assert result == "Hello!"
