import asyncio

import pytest

from ghoshell_moss.core.concepts.channel import ChannelCtx
from ghoshell_moss.core.concepts.command import CommandTask, PyCommand
from ghoshell_moss.core.concepts.errors import CommandError, CommandErrorCode
from ghoshell_moss.core.py_channel import PyChannel, PyChannelBuilder
from ghoshell_moss.message import Message, Text

chan = PyChannel(name="test")


@chan.build.command()
def add(a: int, b: int) -> int:
    """测试一个同步函数是否能正确被调用."""
    return a + b


@chan.build.command()
async def foo() -> int:
    return 9527


@chan.build.command()
async def bar(text: str) -> str:
    return text


@chan.build.command(name="help")
async def some_command_name_will_be_changed_helplessly() -> str:
    return "help"


class Available:
    def __init__(self):
        self.available = True

    def get(self) -> bool:
        return self.available


available_mutator = Available()


@chan.build.command(available=available_mutator.get)
async def available_test_fn() -> int:
    return 123


@pytest.mark.asyncio
async def test_py_channel_baseline() -> None:
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert chan.name() == "test"
        assert runtime.is_connected()
        assert runtime.is_running()
        assert runtime.is_connected()

        # commands 存在.
        commands = list(runtime.own_commands().values())
        assert len(commands) > 0

        # 不用全名来获取函数.
        foo_cmd = runtime.get_command("foo")
        assert foo_cmd is not None
        assert await foo_cmd() == 9527

        # 测试名称有效.
        help_cmd = runtime.get_command("help")
        assert help_cmd is not None
        assert await help_cmd() == "help"

        # 测试乱取拿不到东西
        none_cmd = runtime.get_command("never_exists_command")
        assert none_cmd is None
        # full name 不正确也拿不到.
        help_cmd = runtime.get_command("help")
        assert help_cmd is not None

        # available 测试.
        available_test_cmd = runtime.get_command("available_test_fn")
        assert available_test_cmd is not None
        # 当为 True 的时候.
        assert available_mutator.available
        assert available_test_cmd.is_available() == available_mutator.available
        # 当为 False 的时候, 应该都不能用.
        available_mutator.available = False
        assert available_test_cmd.is_available() == available_mutator.available


@pytest.mark.asyncio
async def test_py_channel_children() -> None:
    assert len(chan.children()) == 0
    a_chan = chan.new_child("a")
    assert len(chan.children()) == 1
    assert isinstance(a_chan, PyChannel)
    assert chan.children()["a"] is a_chan

    async def zoo():
        return 123

    zoo_cmd = a_chan.build.command(return_command=True)(zoo)
    assert isinstance(zoo_cmd, PyCommand)

    assert len(chan.children()) == 1
    async with a_chan.bootstrap() as runtime:
        meta = runtime.self_meta()
        assert meta.name == "a"
        assert len(meta.commands) == 1
        command = runtime.get_command("zoo")
        # 实际执行的是 zoo.
        assert await command() == 123

    assert len(chan.children()) == 1
    async with chan.bootstrap() as runtime:
        assert len(runtime.sub_channels()) == 1
        metas = runtime.metas()
        assert len(metas) == 2
        # meta = runtime.self_meta()
        # assert meta.children == ["a"]


@pytest.mark.asyncio
async def test_py_channel_with_children() -> None:
    main = PyChannel(name="main")
    a_chan = PyChannel(name="a")
    b_chan = PyChannel(name="b")
    main.import_channels(a_chan, b_chan)
    c = PyChannel(name="c")
    d = PyChannel(name="d")
    c.import_channels(d)
    main.import_channels(c)

    async with main.bootstrap() as runtime:
        metas = runtime.metas()
        assert len(metas) == 5
        assert "" in metas
        assert metas["c"].channel_id == c.id()
        assert metas["c.d"].channel_id == c.children()["d"].id()


@pytest.mark.asyncio
async def test_py_channel_execute_task() -> None:
    main = PyChannel(name="main")

    async def foo() -> int:
        _t = ChannelCtx.task()
        _chan = ChannelCtx.channel()
        assert _t is not None
        assert _chan is not None
        return 123

    main.build.command()(foo)
    async with main.bootstrap() as runtime:
        task = runtime.create_command_task("foo")
        runtime.push_task(task)
        result = await task
        assert result == 123


@pytest.mark.asyncio
async def test_py_channel_desc_and_doc_with_ctx() -> None:
    main = PyChannel(name="main")

    def foo_doc() -> str:
        _chan = ChannelCtx.channel()
        return _chan.name()

    async def foo() -> int:
        _t = ChannelCtx.task()
        _chan = ChannelCtx.channel()
        assert _t is None
        assert _chan is not None
        return 123

    main.build.command(doc=foo_doc)(foo)
    async with main.bootstrap() as runtime:
        _foo = runtime.get_own_command("foo")
        r = await _foo()
        assert r == 123
        assert await _foo() == 123
        assert await _foo() == 123
        assert await _foo() == 123
        assert "main" in _foo.meta().interface


@pytest.mark.asyncio
async def test_py_channel_bind():
    class Foo:
        def __init__(self, val: int):
            self.val = val

    main = PyChannel(name="main")
    main.build.with_binding(Foo, Foo(123))

    @main.build.command()
    async def foo() -> int:
        _foo = ChannelCtx.get_contract(Foo)
        return _foo.val

    async with main.bootstrap() as runtime:
        _foo = runtime.get_command("foo")
        assert await _foo() == 123


@pytest.mark.asyncio
async def test_py_channel_context() -> None:
    main = PyChannel(name="main")

    messages = [Message.new().with_content("hello")]

    def foo() -> list[Message]:
        return messages

    # 添加 context message 函数.
    main.build.context_messages(foo)

    async with main.bootstrap() as runtime:
        # 启动时 meta 中包含了生成的 messages.
        meta = runtime.self_meta()
        assert len(meta.context) == 1
        messages.append(Message.new().with_content("world"))

        # 更新后, messages 也变更了.
        await runtime.refresh_metas()
        assert len(runtime.self_meta().context) > 0


@pytest.mark.asyncio
async def test_py_channel_exec_tasks() -> None:
    import asyncio

    main = PyChannel(name="main")

    _sleep = 0.0

    @main.build.command()
    async def foo() -> bool:
        await asyncio.sleep(_sleep)
        t = ChannelCtx.task()
        return t is not None

    async with main.bootstrap() as runtime:
        task = runtime.create_command_task("foo")
        await runtime.execute_task(task)
        assert await task
        task = runtime.create_command_task("foo")
        await runtime.execute_task(task)
        assert await task
        task = runtime.create_command_task("foo")
        await runtime.execute_task(task)
        assert await task

    async with main.bootstrap() as runtime:
        _sleep = 2.0
        task1 = runtime.create_command_task("foo")
        runtime.push_task(task1)
        assert not task1.done()
        await runtime.clear()
        # cleared
        assert task1.done()
        assert task1.exception() is not None
        with pytest.raises(CommandError):
            await task1


@pytest.mark.asyncio
async def test_py_channel_idle() -> None:
    import asyncio

    main = PyChannel(name="main")

    idled = []

    @main.build.command()
    async def foo() -> bool:
        return True

    @main.build.idle
    async def idle() -> None:
        br = ChannelCtx.runtime()
        if br:
            idled.append(1)
        else:
            idled.append(2)

    async with main.bootstrap() as runtime:
        assert len(idled) == 1
        task = runtime.create_command_task("foo")
        runtime.push_task(task)
        await task
        await asyncio.sleep(0.1)
        task = runtime.create_command_task("foo")
        runtime.push_task(task)
        assert len(idled) == 2
        await task
        await asyncio.sleep(0.1)
    assert len(idled) == 3
    assert idled == [1, 1, 1]


@pytest.mark.asyncio
async def test_py_channel_startup_and_close() -> None:
    main = PyChannel(name="main")

    @main.build.command()
    async def foo() -> bool:
        return True

    done = []

    @main.build.startup
    @main.build.close
    async def count_running() -> None:
        _runtime = ChannelCtx.runtime()
        if _runtime:
            done.append(1)

    async with main.bootstrap() as runtime:
        task = runtime.execute_command("foo")
        await task

    assert len(done) == 2


@pytest.mark.asyncio
async def test_py_channel_on_running_and_task_callback() -> None:
    main = PyChannel(name="main")

    @main.build.command()
    async def foo() -> bool:
        return True

    done = []

    @main.build.running
    async def count_tasks() -> None:
        _runtime = ChannelCtx.runtime()

        def add_done_tasks(_task: CommandTask) -> None:
            done.append(_task)

        _runtime.on_task_done(add_done_tasks)
        await _runtime.wait_closed()

    async with main.bootstrap() as runtime:
        assert await runtime.execute_command("foo")
        await asyncio.sleep(0.0)
        r = await runtime.execute_command("foo")
        assert r
        await runtime.wait_idle()
    await asyncio.sleep(0.2)
    assert len(done) == 2


@pytest.mark.asyncio
async def test_py_channel_child_orders() -> None:
    main = PyChannel(name="main")
    a_chan = PyChannel(name="a_chan")
    b_chan = PyChannel(name="b_chan")
    c_chan = PyChannel(name="c_chan")
    d_chan = PyChannel(name="d_chan")
    e_chan = PyChannel(name="e_chan")
    main.import_channels(a_chan, b_chan)
    a_chan.import_channels(c_chan, d_chan)
    b_chan.import_channels(e_chan)

    async with main.bootstrap() as runtime:
        # 深度优先排序.
        all_runtimes = runtime.tree.all()
        order = [b.channel.id() for b in all_runtimes.values()]
        assert order == [main.id(), a_chan.id(), c_chan.id(), d_chan.id(), b_chan.id(), e_chan.id()]
        # 运行第二次.
        order = [b.channel.id() for b in all_runtimes.values()]
        assert order == [main.id(), a_chan.id(), c_chan.id(), d_chan.id(), b_chan.id(), e_chan.id()]


@pytest.mark.asyncio
async def test_py_channel_parent_idle() -> None:
    main = PyChannel(name="main")
    a_chan = PyChannel(name="a_chan")
    b_chan = PyChannel(name="b_chan")
    main.import_channels(a_chan, b_chan)

    order = []

    @main.build.command()
    @a_chan.build.command()
    @b_chan.build.command()
    async def foo(sleep: float) -> None:
        task = ChannelCtx.task()
        await asyncio.sleep(sleep)
        order.append(task)

    async with main.bootstrap() as runtime:
        assert runtime.is_running()
        task1 = runtime.create_command_task("foo", args=(0.1,))
        task2 = runtime.create_command_task("a_chan:foo", args=(0.4,))
        task3 = runtime.create_command_task("b_chan:foo", args=(0.1,))
        task4 = runtime.create_command_task("foo", args=(0.2,))
        # 先执行完.
        runtime.push_task(task1, task2, task3, task4)
        await asyncio.sleep(0.001)
        assert not runtime.is_idle()
        # 等待运行完. 子命令都运行完, 父轨才会 idle.
        await task1
        await runtime.wait_idle()
        assert task3.exec_chan == b_chan.id()
        assert order == [task1, task3, task4, task2]
        metas = runtime.metas()
        assert len(metas) == 3
        assert "" in metas
        assert "a_chan" in metas
        assert "b_chan" in metas
        # assert metas[""].children == ["a_chan", "b_chan"]
        for meta in metas.values():
            assert len(meta.commands) == 1


@pytest.mark.asyncio
async def test_channel_fetch_level2():
    main = PyChannel(name="main")
    a_chan = PyChannel(name="a_chan")
    b_chan = PyChannel(name="b_chan")
    # b_chan 被引用了两次, 但是只会有一个生效.
    a_chan.import_channels(b_chan)
    main.import_channels(a_chan, b_chan)
    async with main.bootstrap() as runtime:
        b1 = runtime.fetch_sub_runtime("b_chan")
        b2 = runtime.fetch_sub_runtime("a_chan.b_chan")
        assert not (b1 and b2)
        assert b1 or b2


def test_channel_split_path():
    _chan = "a.b.c"
    got = PyChannel.split_channel_path_to_names(_chan, 1)
    assert len(got) == 2


@pytest.mark.asyncio
async def test_py_channel_topics():
    from ghoshell_moss.core import ErrorTopic

    main = PyChannel(name="main")
    child = PyChannel(name="child")
    main.import_channels(child)

    produce_done = asyncio.Event()
    consume_done = asyncio.Event()
    consumed = []

    @child.build.running
    async def producer():
        _runtime = ChannelCtx.runtime()
        for i in range(10):
            _runtime.pub_topic(ErrorTopic(errmsg="hello"))
        produce_done.set()

    @main.build.running
    async def consumer():
        _runtime = ChannelCtx.runtime()
        async with _runtime.topic_subscriber(ErrorTopic) as subscriber:
            count = 0
            while subscriber.is_running():
                topic = await subscriber.poll_model()
                consumed.append(topic)
                count += 1
                if count == 10:
                    break
        consume_done.set()

    async with main.bootstrap() as runtime:
        assert runtime.is_running()
        await produce_done.wait()
        await consume_done.wait()
    assert len(consumed) == 10


@pytest.mark.asyncio
async def test_py_channel_instruction_message():
    main = PyChannel(name="main")

    @main.build.instruction
    async def messages() -> str:
        return 'hello'

    async with main.bootstrap() as runtime:
        assert len(runtime.metas()[""].instruction) > 0


@pytest.mark.asyncio
async def test_py_channel_observe_command():
    from ghoshell_moss.core.concepts.command import Observe

    main = PyChannel(name="main")

    @main.build.command()
    async def bar() -> Observe | None:
        return Observe()

    async with main.bootstrap() as runtime:
        assert runtime.is_running()
        bar_task = runtime.create_command_task("bar")
        runtime.push_task(bar_task)
        result = await bar_task
        assert isinstance(result, Observe)
        assert len(result.messages) == 0
        task_result = bar_task.task_result()
        assert task_result.observe


@pytest.mark.asyncio
async def test_py_channel_call_soon_command():
    main = PyChannel(name="main")

    exec_log = []

    @main.build.command()
    async def foo() -> None:
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            exec_log.append("cancelled")

    @main.build.command(
        call_soon=True,
        blocking=True,
    )
    async def bar() -> None:
        return

    async with main.bootstrap() as runtime:
        assert runtime.is_running()
        _foo = runtime.create_command_task("foo")
        _bar = runtime.create_command_task("bar")
        runtime.push_task(_foo)
        # makesure foo has bee called
        await asyncio.sleep(0.1)
        runtime.push_task(_bar)
        await _bar
        assert exec_log == ["cancelled"], _bar.done_at


@pytest.mark.asyncio
async def test_py_channel_priority_command():
    main = PyChannel(name="main")

    cancelled = []

    @main.build.command(
        priority=-1,
    )
    async def foo() -> None:
        try:
            await asyncio.sleep(1)
        except asyncio.CancelledError:
            cancelled.append("foo")

    bar_sleep = 0.1

    @main.build.command(priority=0)
    async def bar() -> None:
        nonlocal bar_sleep
        try:
            await asyncio.sleep(bar_sleep)
        except asyncio.CancelledError:
            cancelled.append("bar")

    @main.build.command(priority=1)
    async def baz() -> None:
        return

    @main.build.command(
        priority=100,
        blocking=False,
    )
    async def nonblock() -> None:
        try:
            await asyncio.sleep(bar_sleep)
        except asyncio.CancelledError:
            cancelled.append("nonblock")

    async with main.bootstrap() as runtime:
        _foo = runtime.create_command_task("foo")
        _bar = runtime.create_command_task("bar")
        runtime.push_task(_foo)
        await asyncio.sleep(0.01)
        runtime.push_task(_bar)
        await _bar
        assert cancelled == ["foo"]

    cancelled.clear()
    bar_sleep = 1.0
    async with main.bootstrap() as runtime:
        _bar = runtime.create_command_task("bar")
        _baz = runtime.create_command_task("baz")
        _nonblock = runtime.create_command_task("nonblock")
        runtime.push_task(_bar)
        await asyncio.sleep(0.1)
        runtime.push_task(_baz, _nonblock)
        await _baz
        assert not _nonblock.done()
        assert cancelled == ["bar"]
        _nonblock.cancel()

    cancelled.clear()
    bar_sleep = 1.0
    async with main.bootstrap() as runtime:
        _foo = runtime.create_command_task("foo")
        _bar = runtime.create_command_task("bar")
        _baz = runtime.create_command_task("baz")
        runtime.push_task(_foo)
        await asyncio.sleep(0.05)
        runtime.push_task(_bar)
        await asyncio.sleep(0.05)
        runtime.push_task(_baz)
        await _baz
        assert cancelled == ["foo", "bar"]


@pytest.mark.asyncio
async def test_py_channel_context_message():
    main = PyChannel(name="channel")

    @main.build.context_messages
    async def messages() -> list[Message]:
        return [Message.new().with_content('hello')]

    async with main.bootstrap() as runtime:
        meta = runtime.self_meta()
        assert len(meta.context) == 1


@pytest.mark.asyncio
async def test_py_channel_multiple_context_message():
    main = PyChannel(name="channel")

    @main.build.context_messages
    async def messages1() -> list[Message]:
        return [Message.new().with_content('hello')]

    @main.build.context_messages
    async def messages2() -> list[Message]:
        return [Message.new().with_content('world')]

    async with main.bootstrap() as runtime:
        meta = runtime.self_meta()
        assert len(meta.context) == 2


@pytest.mark.asyncio
async def test_py_channel_instruction_message():
    main = PyChannel(name="channel")

    @main.build.instruction
    async def hello_message() -> str:
        return 'hello'

    @main.build.instruction
    async def world_message() -> str:
        return 'world'

    async with main.bootstrap() as runtime:
        meta = runtime.self_meta()
        assert 'world' == meta.instruction


@pytest.mark.asyncio
async def test_py_builder_dynamic():
    builder = PyChannelBuilder(name="test")
    assert not builder.is_dynamic()

    async def foo():
        return 123

    def doc() -> str:
        return ''

    async def on_startup():
        return

    builder.command()(foo)
    assert not builder.is_dynamic()
    builder.startup(on_startup)
    assert not builder.is_dynamic()

    builder.command(doc=doc)(foo)
    assert builder.is_dynamic()


@pytest.mark.asyncio
async def test_py_channel_refresh_own_metas():
    main = PyChannel(name="channel")

    expect = "hello"

    def doc() -> str:
        nonlocal expect
        return expect

    @main.build.command(doc=doc)
    async def foo():
        return 123

    async with main.bootstrap() as runtime:
        foo_cmd = runtime.get_own_command('foo')
        assert foo_cmd is not None
        assert foo_cmd.meta().description == expect

        expect = "world"
        await runtime.refresh_own_metas()
        foo_cmd = runtime.get_own_command('foo')
        assert foo_cmd.meta().description == expect
        command_meta = runtime.self_meta().commands[0]
        assert command_meta.name == "foo"
        assert command_meta.description == expect


@pytest.mark.asyncio
async def test_py_channel_with_context_message_but_string():
    main = PyChannel(name="channel")

    @main.build.context_messages
    async def messages() -> list[str]:
        return ["hello"]

    async with main.bootstrap() as runtime:
        await runtime.refresh_metas()
        meta = runtime.self_meta()
        assert len(meta.context) == 1
        assert Text.from_content(meta.context[0].contents[0]).text == "hello"


@pytest.mark.asyncio
async def test_py_channel_virtual_children():
    main = PyChannel(name="channel")
    sub_main = PyChannel(name="sub_channel")
    async with main.bootstrap() as runtime:
        assert runtime.virtual_sub_channels() == {}
        main.build.add_virtual_channel(sub_main)
        await runtime.refresh_metas()
        assert len(runtime.virtual_sub_channels()) == 1


@pytest.mark.asyncio
async def test_py_channel_run_task_with_timeout():
    main = PyChannel(name="channel")

    @main.build.command()
    async def foo() -> int:
        await asyncio.sleep(1)
        return 123

    err = None
    async with main.bootstrap() as runtime:
        try:
            await runtime.execute_command("foo", timeout=0.01)
        except CommandError as e:
            err = e
    assert err is not None
    assert err.code == CommandErrorCode.TIMEOUT.value


@pytest.mark.asyncio
async def test_py_channel_none_block_commands():
    main = PyChannel(name="channel")

    data = []

    @main.build.command(blocking=False)
    async def foo() -> int:
        await asyncio.sleep(0.05)
        data.append(1)
        return 1

    task_done = asyncio.Event()

    def on_task_done(t) -> None:
        task_done.set()

    async with main.bootstrap() as runtime:
        runtime.on_task_done(on_task_done)
        for i in range(10):
            t = runtime.create_command_task('foo')
            runtime.push_task(t)
        # 所有的 task 都应该入队执行完了. 这仍是一个性能敏感测试, 不过有 0.5s buffer. 几乎一定能完成.
        await asyncio.sleep(0.1)
        assert task_done.is_set()
        assert runtime.is_idle()
        assert len(data) == 10


@pytest.mark.asyncio
async def test_py_channel_import_factory_return_none():
    from ghoshell_container import IoCContainer
    main = PyChannel(name="channel")

    def foo(ioc: IoCContainer) -> None:
        return None

    main.build.import_channels(foo)
    async with main.bootstrap() as runtime:
        assert runtime.is_idle()


# --- on_refresh_meta ---


@pytest.mark.asyncio
async def test_on_refresh_meta_basic():
    """build.refresh_meta 注册的回调在 bootstrap 和每次 refresh_metas 时被调用。"""
    main = PyChannel(name="main")
    refreshed: list[int] = []

    # 强制 dynamic，确保 static cache 不跳过 on_refresh_meta
    main.build._dynamic = True

    @main.build.refresh_meta
    async def on_refresh() -> None:
        refreshed.append(1)

    async with main.bootstrap() as runtime:
        # bootstrap 时已经触发了一次 refresh
        before = len(refreshed)
        assert before >= 1
        await runtime.refresh_metas()
        assert len(refreshed) == before + 1


@pytest.mark.asyncio
async def test_on_refresh_meta_multiple():
    """多个 refresh_meta 回调并行执行。"""
    main = PyChannel(name="main")
    order: list[str] = []

    @main.build.refresh_meta
    async def first() -> None:
        order.append("a")

    @main.build.refresh_meta
    async def second() -> None:
        order.append("b")

    async with main.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert sorted(order) == ["a", "b"]


@pytest.mark.asyncio
async def test_on_refresh_meta_sync_callback():
    """同步回调也能被正确包装。"""
    main = PyChannel(name="main")
    called: list[int] = []

    @main.build.refresh_meta
    def sync_refresh() -> None:
        called.append(1)

    async with main.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert called == [1]


@pytest.mark.asyncio
async def test_on_refresh_meta_exception_isolated():
    """一个回调抛异常不影响其他回调，也不影响 refresh 流程。"""
    main = PyChannel(name="main")
    second_called: list[int] = []

    @main.build.refresh_meta
    async def broken() -> None:
        raise RuntimeError("boom")

    @main.build.refresh_meta
    async def survivor() -> None:
        second_called.append(1)

    async with main.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert second_called == [1]
        assert runtime.is_running()


@pytest.mark.asyncio
async def test_on_refresh_meta_with_virtual_children():
    """Hub 模式：on_refresh_meta 更新内部状态，get_virtual_children 返回最新。

    需要 channel 是 dynamic 的，否则 static cache 会跳过 on_refresh_meta。
    """
    main = PyChannel(name="main")
    child_a = PyChannel(name="a")
    child_b = PyChannel(name="b")
    children_list: list[PyChannel] = []

    # 强制 dynamic，确保每轮 refresh 都走 on_refresh_meta
    main.build._dynamic = True

    @main.build.refresh_meta
    async def prepare() -> None:
        main.build._virtual_children.clear()
        for ch in children_list:
            main.build.add_virtual_channel(ch)

    async with main.bootstrap() as runtime:
        # bootstrap 已经刷新过一次，列表为空
        assert len(runtime.virtual_sub_channels()) == 0

        children_list.append(child_a)
        await runtime.refresh_metas()
        assert len(runtime.virtual_sub_channels()) == 1
        assert "a" in runtime.virtual_sub_channels()

        children_list.append(child_b)
        await runtime.refresh_metas()
        assert len(runtime.virtual_sub_channels()) == 2

        children_list.clear()
        await runtime.refresh_metas()
        assert len(runtime.virtual_sub_channels()) == 0


@pytest.mark.asyncio
async def test_on_refresh_meta_py_builder_direct():
    """PyChannelBuilder.refresh_meta 注册 + on_refresh_meta 调用。"""
    builder = PyChannelBuilder(name="test")
    called: list[int] = []

    @builder.refresh_meta
    async def on_refresh() -> None:
        called.append(1)

    await builder.on_refresh_meta()
    assert called == [1]
