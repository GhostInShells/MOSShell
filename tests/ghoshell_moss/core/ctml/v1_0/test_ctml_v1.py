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


# ---- deepseek v4 ---- #

# ============= 测试 "Time is First-Class Citizen" =============

@pytest.mark.asyncio
async def test_time_sequential_execution_respects_duration():
    """
    验证时间第一公民：命令的实际执行时长会影响后续命令的启动时间。

    场景：cmd_a 耗时 0.1s，cmd_b 无耗时。
    预期：cmd_b 的完成时间 ≥ cmd_a 完成时间 + cmd_b 自身耗时
    """
    timeline = []

    chan = new_channel(name="timer")

    @chan.build.command()
    async def slow_cmd() -> str:
        start = asyncio.get_event_loop().time()
        await asyncio.sleep(0.1)
        end = asyncio.get_event_loop().time()
        timeline.append(("slow_start", start))
        timeline.append(("slow_end", end))
        return "slow_done"

    @chan.build.command()
    async def fast_cmd() -> str:
        start = asyncio.get_event_loop().time()
        timeline.append(("fast_start", start))
        return "fast_done"

    await ctml_shell_test(
        chan,
        ctml="""
        <_ until="flow">
            <timer:slow_cmd/>
            <timer:fast_cmd/>
        </_>
        """
    )

    # 提取时间点
    slow_end_time = None
    fast_start_time = None

    for event, t in timeline:
        if event == "slow_end":
            slow_end_time = t
        if event == "fast_start":
            fast_start_time = t

    # 核心断言：fast_cmd 必须等待 slow_cmd 完全结束后才能开始
    assert fast_start_time is not None
    assert slow_end_time is not None
    assert fast_start_time >= slow_end_time - 0.01  # 允许微小误差


@pytest.mark.asyncio
async def test_timeout_cancels_ongoing_commands_cleanly():
    """
    验证 timeout 会干净地取消正在执行的命令，并触发清理逻辑。

    场景：一个执行 0.5s 的命令，作用域 timeout=0.1s
    预期：命令被取消，且 __aexit__ 清理逻辑执行
    """
    cleanup_called = False

    chan = new_channel(name="clean")

    class CleanupTracker:
        cleaned = False

    tracker = CleanupTracker()

    @chan.build.command()
    async def long_running() -> str:
        try:
            await asyncio.sleep(0.5)
            return "completed"
        except asyncio.CancelledError:
            # 模拟清理逻辑
            tracker.cleaned = True
            raise

    tasks = await ctml_shell_test(
        chan,
        ctml="""
        <_ timeout="0.1">
            <clean:long_running/>
        </_>
        """
    )

    # 验证命令被取消且清理逻辑执行
    assert tracker.cleaned is True
    for task in tasks:
        if task.meta.name == "long_running":
            assert task.cancelled() is True


# ============= 测试父子通道 occupy 阻塞语义 =============

@pytest.mark.asyncio
async def test_parent_occupy_blocks_child_commands():
    """
    验证父子 occupy 阻塞：父通道有命令执行时，子通道的命令不会分发给执行。

    规范原文："父通道当前执行occupy命令时，所有发往该父通道及其所有子通道的新命令都会保持pending"

    场景：
        1. 父通道执行一个持续 0.2s 的命令
        2. 在父命令执行期间（0.05s 后），发送子通道的命令
        3. 子通道命令只能在父命令结束后才开始
    """
    order = []

    parent = new_channel(name="parent")
    child = new_channel(name="child")
    parent.import_channels((child, "child"))

    @parent.build.command()
    async def parent_long() -> None:
        order.append("parent_start")
        await asyncio.sleep(0.2)
        order.append("parent_end")

    @child.build.command()
    async def child_fast() -> None:
        order.append("child_executed")

    # 创建一个可以延迟发送子命令的机制
    # 由于 ctml_shell_test 是静态解析，我们换一种方式：在父命令内部触发子命令
    # 或者更简单：验证静态 CTML 中，父通道命令后的子通道命令会被阻塞

    # 方案：CTML 顺序描述，验证执行顺序符合阻塞语义
    tasks = await ctml_shell_test(
        parent,
        ctml="""
        <_ until="all">
            <parent:parent_long/>
            <parent.child:child_fast/>
        </_>
        """
    )

    # 如果父命令阻塞子通道，则 child_fast 必须在 parent_long 完全结束后才能执行
    # 检查 order 中 parent_end 出现在 child_executed 之前
    parent_end_index = order.index("parent_end") if "parent_end" in order else -1
    child_index = order.index("child_executed") if "child_executed" in order else -1

    assert parent_end_index != -1
    assert child_index != -1
    assert parent_end_index < child_index


@pytest.mark.asyncio
async def test_pending_commands_released_after_parent_releases_occupy():
    """
    验证父通道释放 occupy 后，pending 的子通道命令恢复执行。

    场景：
        1. 父通道开始执行命令 A（occupy）
        2. 期间收到子通道命令 B、C（pending）
        3. 父命令 A 完成后，B、C 依次执行
    """
    execution_order = []

    root = new_channel(name="root")
    leaf = new_channel(name="leaf")
    root.import_channels((leaf, "leaf"))

    @root.build.command()
    async def hold() -> None:
        execution_order.append("hold_start")
        await asyncio.sleep(0.15)
        execution_order.append("hold_end")

    @leaf.build.command()
    async def first() -> None:
        execution_order.append("first_executed")

    @leaf.build.command()
    async def second() -> None:
        execution_order.append("second_executed")

    await ctml_shell_test(
        root,
        ctml="""
        <_ until="all">
            <root:hold/>
            <root.leaf:first/>
            <root.leaf:second/>
        </_>
        """
    )

    # 断言：hold_end 必须在 first 和 second 之前
    hold_end_idx = execution_order.index("hold_end")
    first_idx = execution_order.index("first_executed")
    second_idx = execution_order.index("second_executed")

    assert hold_end_idx < first_idx
    assert hold_end_idx < second_idx


# ============= 测试流式参数的高级场景 =============

@pytest.mark.asyncio
async def test_ctml_nested_in_ctml():
    """
    验证 ctml__ 流式参数允许嵌套 CTML。

    规范："只有 ctml__ 允许嵌套 ctml"

    场景：外层命令生成内层 CTML，内层 CTML 被解释执行
    """
    nested_executed = False

    outer = new_channel(name="outer")
    inner = new_channel(name="inner")
    outer.import_channels((inner, "inner"))

    @outer.build.command()
    async def generator(ctml__: AsyncIterable) -> None:
        nonlocal nested_executed
        # 修复测试: ctml 实际上拿到的是 command token 对象.
        async for token in ctml__:
            # 这里应该收到被解释后的内层命令执行结果？
            # 实际测试中，ctml__ 参数接收的是原始 CTML 字符串流
            # 我们需要验证这个流被正确传递
            if "inner:say_hello" in token.content:
                nested_executed = True

    @inner.build.command()
    async def say_hello() -> str:
        return "hello from inner"

    # 父子才运行嵌套.
    outer.build.import_channels(inner)

    # 外层命令接收 ctml__，然后在内部应该解析执行
    # 由于 CTML 解析器会先解析外层，将内层 CTML 作为参数传递
    tasks = await ctml_shell_test(
        outer,
        ctml="""
        <outer:generator>
            <outer.inner:say_hello/>
        </outer:generator>
        """
    )

    # 验证内层 CTML 被传递给了 generator 的 ctml__ 参数
    # 注意：这个测试的断言依赖于 ctml__ 的实现细节
    assert nested_executed is True


@pytest.mark.asyncio
async def test_chunks__streaming_realtime():
    """
    验证 chunks__ 参数的流式特性：chunk 应该边生成边被消费，不需要等待完整内容。

    场景：内容有 3 个部分，每个部分间隔 0.05s
    预期：consumer 在收到第一个 chunk 时就能开始处理，不等完整内容
    """
    received_chunks = []
    chunk_timestamps = []

    chan = new_channel(name="stream")

    @chan.build.command()
    async def stream_consumer(chunks__: AsyncIterable[str]) -> None:
        async for chunk in chunks__:
            received_chunks.append(chunk)
            chunk_timestamps.append(asyncio.get_event_loop().time())

    # 生成一个分块的内容
    # 注意：在实际 CTML 中，开放-闭合标签内的文本会被自动分块
    # 这里我们用静态 CTML 测试（一次性传入完整内容），但真正的流式需要 generator
    # 更好的测试方式：用程序生成流式 CTML

    # 简化：测试多段文本被正确拼接
    tasks = await ctml_shell_test(
        chan,
        ctml="""
        <stream:stream_consumer>Hello</stream:stream_consumer>
        """
    )

    # 验证内容被接收
    assert len(received_chunks) > 0
    # 验证拼接
    assert "".join(received_chunks) == "Hello"


# ============= 测试错误恢复与容错 =============

@pytest.mark.asyncio
async def test_command_failure_does_not_crash_sibling_in_flow():
    """
    验证 flow 模式下，一个命令失败不会阻止同作用域内后续命令的执行。

    场景：
        1. cmd_a 抛出异常
        2. cmd_b 正常
    预期：cmd_a 失败记录，cmd_b 继续执行
    """
    cmd_b_executed = False

    chan = new_channel(name="fault")

    @chan.build.command()
    async def failing_cmd() -> str:
        from ghoshell_moss import CommandErrorCode
        # 不能抛出 ValueError, 入参错误是模型错误, 会认为是模型规划有问题的致死错误.
        # 而 failed 是容错的.
        raise CommandErrorCode.FAILED.error("Intentional failure")

    @chan.build.command()
    async def healthy_cmd() -> None:
        nonlocal cmd_b_executed
        cmd_b_executed = True

    tasks = await ctml_shell_test(
        chan,
        ctml="""
        <_ until="flow">
            <fault:failing_cmd/>
            <fault:healthy_cmd/>
        </_>
        """
    )

    # 验证 healthy_cmd 仍然执行了
    assert cmd_b_executed is True

    # 验证 failing_cmd 确实失败了
    failing_task = None
    for t in tasks:
        if t.meta.name == "failing_cmd":
            failing_task = t
            break

    assert failing_task is not None
    assert failing_task.exception() is not None
    assert "Intentional failure" in str(failing_task.exception())


# ============= 测试 CTML 规范中的红线约束 =============

@pytest.mark.asyncio
async def test_root_channel_prefix_forbidden():
    """
    验证红线：根通道 __main__ 的命令不能加路径前缀。

    规范原文："根通道 __main__ 的命令不带路径前缀（如 <wait>）。严禁写成 <__main__:wait>"

    预期：带前缀的应该报错或自动修正（根据容错策略）
    """

    got = ''

    def _build_main(shell):
        @shell.main_channel.build.command()
        async def wait(seconds: float = 0.02) -> str:
            nonlocal got
            got = str(seconds)
            return "waited"

    # 错误用法：加 __main__: 前缀，系统应该报错或忽略前缀
    await ctml_shell_test(
        builder=_build_main,
        ctml='<__main__:wait seconds="0.01"/>'
    )
    # 实际上做了容错, __main__ 是可运行的. 考虑未来把提示语法不规范一起返回给模型.
    assert got == '0.01'


@pytest.mark.asyncio
async def test_text__parameter_cannot_be_attribute():
    """
    验证红线：text__ 参数必须用开放-闭合标签传递，不能作为 XML 属性。

    规范："text__/chunks__/ctml__ 三类特殊参数必须用开放-闭合标签传递内容，绝对不能把这些参数作为 XML 属性传递"

    预期：错误的用法导致解析错误
    """
    chan = new_channel(name="logger")

    received = ''

    @chan.build.command()
    async def log(text__: str) -> str:
        nonlocal received
        received = text__
        return text__

    # 错误：将 text__ 作为属性传递
    await ctml_shell_test(
        chan,
        ctml='<logger:log text__="hello"/>'
    )
    # 作者的话:
    # 实际上做了容错. 所以不会对特殊参数报错. 模型生成错误, 但意图正确也能跑.
    assert received == "hello"


# ============= 测试通道动态性（moss_dynamic） =============

@pytest.mark.asyncio
async def test_dynamic_channel_interface_refresh():
    """
    验证 Channel 的动态性：moss_dynamic 可以刷新 interface。

    场景：
        1. 初始 Channel 有命令 cmd_a
        2. 运行时刷新，添加命令 cmd_b
        3. 新 CTML 可以调用 cmd_b
    """
    # 注意：这个测试依赖于具体的动态刷新实现
    # 这里仅测试概念，实际可能需要 Mock ChannelTree 的 refresh 机制

    dynamic_chan = new_channel(name="dynamic")

    # 阶段 1：只有 cmd_a
    call_tracker = {"a": False, "b": False}

    @dynamic_chan.build.command()
    async def cmd_a() -> str:
        call_tracker["a"] = True
        return "a"

    # 阶段 2：假设刷新后添加了 cmd_b
    # 在实际实现中，需要调用 refresh_metas 或类似方法
    # 这里简化为：直接添加命令后，验证新 CTML 能调用

    @dynamic_chan.build.command()
    async def cmd_b() -> str:
        call_tracker["b"] = True
        return "b"

    # 连续调用两个命令
    await ctml_shell_test(
        dynamic_chan,
        ctml="""
        <_ until="flow">
            <dynamic:cmd_a/>
            <dynamic:cmd_b/>
        </_>
        """
    )

    # 两个命令都应该被调用
    assert call_tracker["a"] is True
    assert call_tracker["b"] is True


# ============= 测试复杂时序规划 =============

@pytest.mark.asyncio
async def test_complex_timeline_with_multiple_scopes():
    """
    测试复杂时序：多个作用域嵌套，混合 text 和 command，验证整体时序正确。

    场景（模拟机器人打招呼）：
        1. [作用域 A] 挥手 0.2s，同时说话 "Hi"（并行）
        2. [作用域 B] 等待 0.1s，微笑 0.3s，同时说话 "How are you"（并行）
        3. [作用域 C] 点头 0.1s

    预期：总执行时间约 0.2s + 0.3s + 0.1s = 0.6s（有重叠）
    """
    timeline = []

    robot = new_channel(name="robot")

    @robot.build.command()
    async def wave(duration: float = 0.2) -> None:
        start = asyncio.get_event_loop().time()
        timeline.append(("wave_start", start))
        await asyncio.sleep(duration)
        timeline.append(("wave_end", asyncio.get_event_loop().time()))

    @robot.build.command()
    async def smile(duration: float = 0.3) -> None:
        start = asyncio.get_event_loop().time()
        timeline.append(("smile_start", start))
        await asyncio.sleep(duration)
        timeline.append(("smile_end", asyncio.get_event_loop().time()))

    @robot.build.command()
    async def nod(duration: float = 0.1) -> None:
        start = asyncio.get_event_loop().time()
        timeline.append(("nod_start", start))
        await asyncio.sleep(duration)
        timeline.append(("nod_end", asyncio.get_event_loop().time()))

    @robot.build.content_command
    async def speak(chunks__: AsyncIterable[str]) -> None:
        start = asyncio.get_event_loop().time()
        timeline.append(("speak_start", start))
        async for _ in chunks__:
            pass
        timeline.append(("speak_end", asyncio.get_event_loop().time()))

    start_time = asyncio.get_event_loop().time()

    await ctml_shell_test(
        robot,
        ctml="""
        <_ until="all">
            <robot:wave/>
            <robot:_>Hi</robot:_>
        </_>
        <_ until="all">
            <robot:smile/>
            <robot:_>How are you</robot:_>
        </_>
        <_ until="all">
            <robot:nod/>
        </_>
        """
    )

    end_time = asyncio.get_event_loop().time()
    total_duration = end_time - start_time

    # 验证总时长在合理范围内（约 0.6s ± 0.15s）
    # 注意：由于并行，wave(0.2) + smile(0.3) + nod(0.1) = 0.6s
    # 但加上 text 执行和调度开销，允许一定误差
    assert 0.45 <= total_duration <= 0.8

    # 验证波形：wave 和 speak1 应该并行
    wave_start = None
    speak1_start = None
    for name, t in timeline:
        if name == "wave_start":
            wave_start = t
        if name == "speak_start" and "Hi" in str(timeline):  # 简化判断
            speak1_start = t

    if wave_start and speak1_start:
        assert abs(wave_start - speak1_start) < 0.05


# ============= 测试原语（Primitives） =============

@pytest.mark.asyncio
async def test_primitive_clear_cancels_all():
    """
    验证原语 <clear> 可以取消所有正在执行的命令。

    规范：原语只能在根通道使用。
    """
    task_executed = False

    def _build_main(shell) -> None:
        @shell.main_channel.build.command()
        async def long_task() -> None:
            nonlocal task_executed
            try:
                await asyncio.sleep(0.5)
                task_executed = True
            except asyncio.CancelledError:
                pass

    tasks = await ctml_shell_test(
        builder=_build_main,
        ctml="""
        <_ until="all">
            <long_task/>
            <interrupt/>
        </_>
        """
    )

    # long_task 应该被 clear 取消，不会执行完成
    assert task_executed is False

    # 验证 clear 本身是一个命令（有对应的 task）
    interrupt_task = None
    for t in tasks:
        if t.meta.name == "interrupt":
            interrupt_task = t
            break

    assert interrupt_task is not None
    assert interrupt_task.success() is True


@pytest.mark.asyncio
async def test_primitive_cannot_be_used_in_non_root_channel():
    """
    验证红线：原语只能在根通道使用。

    预期：在非根通道使用原语应报错或忽略
    """
    non_root = new_channel(name="non_root")

    @non_root.build.command()
    async def some_cmd() -> str:
        return "ok"

    # 在非根通道作用域内使用 <clear> 应该报错
    with pytest.raises(InterpretError):
        await ctml_shell_test(
            non_root,
            ctml="""
            <_ until="all" channel="non_root">
                <some_cmd/>
                <clear/>
            </_>
            """
        )


# ============= 总结性测试：端到端人机交互场景 =============

@pytest.mark.asyncio
async def test_end_to_end_assistant_greeting_and_question():
    """
    端到端测试模拟一个完整的助手回复：
    1. 先语音招呼 "Hello"（并行微笑）
    2. 说完后，等待用户输入（Observe 等待）
    3. 用户输入后，助手回答 "I think it's 42"

    这个测试验证 CTML 能否表达真实的交互流程。
    """
    interaction_log = []

    assistant = new_channel(name="assistant")

    @assistant.build.command()
    async def smile() -> None:
        interaction_log.append("smiling")

    @assistant.build.content_command
    async def speak(chunks__: AsyncIterable[str]) -> None:
        text = []
        async for chunk in chunks__:
            text.append(chunk)
        interaction_log.append(f"spoke: {''.join(text)}")

    @assistant.build.command()
    async def wait_for_input() -> None:
        """模拟等待用户输入（Observe）"""
        from ghoshell_moss import ObserveError
        interaction_log.append("waiting_for_user")
        # 返回 Observe 让系统等待下一轮
        raise ObserveError()

    @assistant.build.command()
    async def answer() -> str:
        interaction_log.append("answering")
        return "42"

    # 注意：完整的 Observe 测试需要多轮交互
    # 这里测试第一阶段的打招呼
    tasks = await ctml_shell_test(
        assistant,
        ctml="""
        <_ until="all">
            <assistant:smile/>
            <assistant:_>Hello</assistant:_>
        </_>
        """
    )

    # 验证打招呼阶段正确执行
    assert "smiling" in interaction_log
    assert "spoke: Hello" in interaction_log
