import asyncio
import pytest

from ghoshell_moss import new_ctml_shell
from ghoshell_moss.core import Command, CommandError, CommandToken
from ghoshell_moss.core.duplex.thread_channel import create_thread_channel
from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.core import ChannelCtx
from ghoshell_moss.core.concepts.topic import LogTopic, TopicService


@pytest.mark.asyncio
async def test_thread_channel_start_and_close():
    provider, proxy = create_thread_channel("proxy")
    chan = PyChannel(name="provider")
    async with provider.arun(chan):
        runtime = provider.runtime
        assert runtime is not None
        assert runtime.is_running()
    assert not runtime.is_running()
    assert not provider.is_running()


@pytest.mark.asyncio
async def test_thread_channel_raise_in_proxy():
    provider, proxy = create_thread_channel("proxy")
    chan = PyChannel(name="provider")
    # 测试 channel 能够正常被启动.
    async with provider.arun(chan):
        with pytest.raises(RuntimeError):
            async with proxy.bootstrap():
                raise RuntimeError()


@pytest.mark.asyncio
async def test_thread_channel_run_in_thread():
    provider, proxy = create_thread_channel("proxy")
    chan = PyChannel(name="provider")
    provider.run_in_thread(chan)

    await provider.aclose()
    await provider.wait_closed()
    assert not provider.is_running()


@pytest.mark.asyncio
async def test_thread_channel_run_in_tasks():
    provider, proxy = create_thread_channel("proxy")
    chan = PyChannel(name="provider")
    provider_run_task = asyncio.create_task(provider.arun_until_closed(chan))

    async def _cancel():
        await asyncio.sleep(0.2)
        await provider.aclose()

    # 0.2 秒后关闭 provider run task
    await asyncio.gather(provider_run_task, _cancel())
    assert not provider.is_running()
    await provider.wait_closed()
    assert provider_run_task.done()
    await provider_run_task
    # 正常退出了.


@pytest.mark.asyncio
async def test_thread_channel_run_in_thread():
    provider, proxy = create_thread_channel("proxy")
    chan = PyChannel(name="provider")
    # 重新创建 provider.
    provider = provider.copy()
    provider.run_in_thread(chan)
    await provider.aclose()
    await provider.wait_closed()
    assert not provider.is_running()


@pytest.mark.asyncio
async def test_thread_channel_baseline():
    async def foo() -> int:
        return 123

    async def bar() -> int:
        return 456

    chan = PyChannel(name="provider")
    a_chan = PyChannel(name="a")
    # provider channel 注册 foo.
    foo_cmd: Command = chan.build.command(return_command=True)(foo)
    assert isinstance(foo_cmd, Command)
    chan.import_channels(a_chan)
    # a_chan 增加 command bar.
    a_chan.build.command()(bar)

    provider, proxy_chan = create_thread_channel("proxy")

    # 在另一个线程中运行.
    async with provider.arun(chan):
        # 判断 channel 已经启动.
        main_runtime = provider.runtime
        metas = main_runtime.metas()
        assert len(metas) == 2
        assert "a" in metas
        assert main_runtime.name == "provider"
        assert main_runtime.is_running()
        assert main_runtime.is_connected()
        assert main_runtime.is_running()
        proxy_side_foo_meta = main_runtime.own_meta()
        assert proxy_side_foo_meta.available
        assert len(proxy_side_foo_meta.commands) > 0
        assert proxy_side_foo_meta.name == "provider"

        async with proxy_chan.bootstrap() as proxy_runtime:
            await proxy_runtime.wait_connected()
            await proxy_runtime.refresh_metas()
            metas = proxy_runtime.metas()
            assert len(metas) == 2
            # 阻塞等待连接成功.
            proxy_meta = proxy_runtime.own_meta()
            assert proxy_meta.name == "proxy"
            assert proxy_meta is not None
            # 名字被替换了.
            assert proxy_meta.available is True
            # 存在目标命令.
            assert len(proxy_meta.commands) == 1
            foo_cmd_meta = proxy_meta.commands[0]
            # 服务端和客户端的 command 使用的 chan 会变更
            # proxy.a / proxy.b
            assert foo_cmd_meta.name == foo_cmd.meta().name

            # 判断仍然有一个子 channel.
            assert "a" in chan.children()
            # 判断 proxy 也有 children
            metas = proxy_runtime.metas()
            assert "a" in metas
            assert main_runtime.own_meta().name == "provider"
            assert proxy_meta.name == "proxy"

            # 客户端仍然可以调用命令.
            proxy_side_foo = proxy_runtime.get_command("foo")
            assert proxy_side_foo is not None

            assert proxy_runtime.is_available()
            assert provider.is_running()
            result = await proxy_side_foo()
            assert result == 123

        assert not proxy_runtime.is_running()
    assert not provider.is_running()


def test_thread_channel_lost_connection():
    async def foo() -> int:
        return 123

    chan = PyChannel(name="provider")
    chan.build.command(return_command=True)(foo)
    provider, proxy = create_thread_channel("proxy")
    provider.run_in_thread(chan)

    async def proxy_main():
        # 启动 proxy
        async with proxy.bootstrap() as proxy_runtime:
            await proxy_runtime.wait_connected()
            # 验证连接正常
            assert proxy_runtime.is_running()
            _foo = proxy_runtime.get_command("foo")
            assert _foo is not None

            # 模拟连接中断（通过关闭 provider）
            provider.close()
            assert not provider.is_running()
            assert proxy_runtime.is_running()
            _foo = proxy_runtime.get_command("foo")
            # 中断后抛出 command error.
            with pytest.raises(CommandError):
                result = await _foo()
            assert not proxy_runtime.is_running()

    asyncio.run(proxy_main())
    provider.close()
    provider.wait_closed_sync()


@pytest.mark.asyncio
async def test_thread_channel_refresh_meta():
    foo_doc = "hello"

    def doc_fn() -> str:
        return foo_doc

    chan = PyChannel(name="provider")

    @chan.build.command(doc=doc_fn)
    async def foo() -> int:
        return 123

    provider, proxy = create_thread_channel("proxy")

    async with provider.arun(chan):
        async with proxy.bootstrap() as runtime:
            await runtime.wait_connected()
            # 验证连接正常
            assert runtime.is_running()

            foo = runtime.get_command("foo")
            assert "hello" in foo.meta().interface

            foo_doc = "world"
            generated_foo_doc = doc_fn()
            assert generated_foo_doc == foo_doc

            # 没有立刻变更:
            foo1 = runtime.get_command("foo")
            assert foo1 is not None
            assert "hello" in foo1.meta().interface

            # 刷新了 meta 才会变更.
            await runtime.refresh_metas()
            # 这时判断, provider 侧已经更新了.
            provider_foo = provider.runtime.get_command("foo")
            assert provider_foo is not None
            assert "world" in provider_foo.meta().interface

            foo2 = runtime.get_command("foo")

            assert foo2 is not foo1
            assert "hello" not in foo2.meta().interface
            assert "world" in foo2.meta().interface


@pytest.mark.asyncio
async def test_thread_channel_has_child():
    chan = PyChannel(name="provider")

    @chan.build.command()
    async def foo() -> int:
        return 123

    sub1 = PyChannel(name="sub1")
    chan.import_channels(sub1)

    @sub1.build.command()
    async def bar() -> int:
        return 456

    provider, proxy = create_thread_channel("proxy")
    provider.run_in_thread(chan)
    try:
        async with proxy.bootstrap() as runtime:
            assert runtime.is_running()
            await runtime.wait_connected()
            metas = runtime.metas()

            assert "sub1" in metas
            # # 判断子 channel 存在.
            value = await runtime.execute_command("sub1:bar")
            assert value == 456
    finally:
        provider.close()
        await provider.wait_closed()


@pytest.mark.asyncio
async def test_thread_channel_exception():
    chan = PyChannel(name="provider")

    @chan.build.command()
    async def foo() -> int:
        raise ValueError("foo")

    provider, proxy = create_thread_channel("proxy")
    provider.run_in_thread(chan)
    try:
        async with proxy.bootstrap() as proxy_runtime:
            await proxy_runtime.wait_connected()
            assert proxy_runtime.is_available()
            assert proxy_runtime.is_running()
            _foo = proxy_runtime.get_command("foo")
            with pytest.raises(CommandError):
                await _foo()

    finally:
        provider.close()
    await provider.wait_closed()


@pytest.mark.asyncio
async def test_thread_channel_idle():
    chan = PyChannel(name="provider")

    idled = []
    idled_done = asyncio.Event()

    @chan.build.command()
    async def foo() -> int:
        return 123

    @chan.build.idle
    async def idle():
        try:
            idled.append(True)
        finally:
            idled_done.set()

    provider, proxy = create_thread_channel("proxy")
    provider.run_in_thread(chan)
    try:
        async with proxy.bootstrap() as proxy_runtime:
            await proxy_runtime.wait_connected()
            assert proxy_runtime.is_idle()
            assert provider.runtime.is_idle()
            await proxy_runtime.wait_idle()
            assert len(idled) == 1
            idled_done.clear()

            r = await proxy_runtime.execute_command("foo")
            assert r == 123
            assert proxy_runtime.is_idle()
            await proxy_runtime.wait_idle()
            await idled_done.wait()
            # assert provider.runtime.is_idle()
            assert len(idled) == 2

    finally:
        provider.close()
    await provider.wait_closed()


@pytest.mark.asyncio
async def test_thread_channel_with_delta_func():
    chan = PyChannel(name="provider")

    @chan.build.command()
    async def chunks(chunks__) -> int:
        count = 0
        async for chunk in chunks__:
            count += 1
        return count

    @chan.build.command()
    async def text(text__) -> str:
        return text__

    async def generate():
        for i in range(10):
            yield "i"

    @chan.build.command()
    async def tokens(tokens__) -> int:
        count = 0
        async for token in tokens__:
            count += 1
        return count

    async def generate_tokens():
        for i in range(10):
            yield CommandToken(seq="delta", name="tokens", content="%d" % i)

    provider, proxy = create_thread_channel("proxy")
    async with provider.arun(chan):
        async with proxy.bootstrap() as runtime:
            await runtime.wait_connected()
            value = await runtime.execute_command("chunks", kwargs=dict(chunks__=generate()))
            assert value == 10
            value = await runtime.execute_command("text", kwargs=dict(text__="hello"))
            assert value == "hello"
            value = await runtime.execute_command("tokens", kwargs=dict(tokens__=generate_tokens()))
            assert value == 10


@pytest.mark.asyncio
async def test_thread_provider_pub_topic():
    chan = PyChannel(name="provider")

    wait_connected = asyncio.Event()

    @chan.build.running
    async def send_topic() -> None:
        await wait_connected.wait()
        _runtime = ChannelCtx.runtime()
        async with _runtime.topic_publisher() as publisher:
            for i in range(10):
                await asyncio.sleep(0.0)
                await publisher.pub(LogTopic(level="info", message=str(i)))

    provider, proxy = create_thread_channel("proxy")

    main = PyChannel(name="main")
    main.import_channels(proxy)

    received = []

    async with provider.arun(chan):
        assert provider.container.get(TopicService) is provider.runtime.importlib.topics
        async with main.bootstrap() as runtime:
            proxy_runtime = await runtime.fetch_sub_runtime("proxy")
            await proxy_runtime.wait_connected()
            # 保证连接后才有消息体广播.
            wait_connected.set()

            # 接受 provider 侧的 topic.
            async with runtime.topic_subscriber(LogTopic) as subscriber:
                count = 0
                while count < 10:
                    topic = await subscriber.poll_model()
                    received.append(topic)
                    count += 1
    assert len(received) == 10


@pytest.mark.asyncio
async def test_thread_proxy_pub_topic():
    chan = PyChannel(name="provider")
    a_chan = PyChannel(name="a_channel")
    chan.import_channels(a_chan)

    provider, proxy = create_thread_channel("proxy")

    main = PyChannel(name="main")
    main.import_channels(proxy)

    received = []
    receive_done = asyncio.Event()

    @a_chan.build.running
    async def receive_topic() -> None:
        """
        这次是 provider 的 a_channel 监听事件.
        """
        _runtime = ChannelCtx.runtime()
        async with _runtime.topic_subscriber(LogTopic) as subscriber:
            count = 0
            while count < 10:
                topic = await subscriber.poll_model()
                received.append(topic)
                count += 1
        receive_done.set()

    async with main.bootstrap() as runtime:
        proxy_runtime = await runtime.fetch_sub_runtime("proxy")
        async with provider.arun(chan):
            await proxy_runtime.wait_connected()
            # 保证连接后才有消息体广播.

            # 从 proxy 侧的 main channel 发送消息给 provider 侧.
            async with runtime.topic_publisher() as publisher:
                for i in range(10):
                    await asyncio.sleep(0.0)
                    await publisher.pub(LogTopic(level="info", message=str(i)))
            await receive_done.wait()
    assert len(received) == 10


@pytest.mark.asyncio
async def test_thread_provider_lazy_subscribe():
    chan = PyChannel(name="provider")
    a_chan = PyChannel(name="a_channel")
    chan.import_channels(a_chan)

    provider, proxy = create_thread_channel("proxy")

    main = PyChannel(name="main")
    main.import_channels(proxy)

    received = []
    receive_done = asyncio.Event()

    @a_chan.build.running
    async def receive_topic() -> None:
        """
        这次是 provider 的 a_channel 监听事件.
        """
        _runtime = ChannelCtx.runtime()
        async with _runtime.topic_subscriber(LogTopic) as subscriber:
            count = 0
            while count < 10:
                topic = await subscriber.poll_model()
                received.append(topic)
                count += 1
        receive_done.set()

    # provider 侧先运行, 已经开始监听.
    async with provider.arun(chan):
        async with main.bootstrap() as runtime:
            # proxy 侧后运行, 这时 provider 已经开始监听了. 要在建连后重新开始监听.
            proxy_runtime = await runtime.fetch_sub_runtime("proxy")
            await proxy_runtime.wait_connected()
            # 从 proxy 侧的 main channel 发送消息给 provider 侧.
            async with runtime.topic_publisher() as publisher:
                for i in range(10):
                    await asyncio.sleep(0.0)
                    await publisher.pub(LogTopic(level="info", message=str(i)))
            await receive_done.wait()
    assert len(received) == 10


@pytest.mark.asyncio
async def test_thread_channel_do_not_share_local_topic():
    chan = PyChannel(name="provider")
    a_chan = PyChannel(name="a_channel")
    chan.import_channels(a_chan)

    provider, proxy = create_thread_channel("proxy")

    # provider 侧先运行, 已经开始监听.
    async with provider.arun(chan):
        async with proxy.bootstrap() as proxy_runtime:
            # proxy 侧后运行, 这时 provider 已经开始监听了. 要在建连后重新开始监听.
            await proxy_runtime.wait_connected()

            async with proxy_runtime.topic_subscriber(LogTopic) as subscriber:
                poll_task = asyncio.create_task(subscriber.poll_model())
                async with provider.runtime.topic_publisher() as publisher:
                    for i in range(10):
                        await asyncio.sleep(0.0)
                        topic = LogTopic(level="info", message=str(i))
                        # 关键在这里, topic 改成 local 类型.
                        topic.meta.local = True
                        await publisher.pub(topic)
                await asyncio.sleep(0.1)

                # 仍然没有收到.
                assert not poll_task.done()

    provider, proxy = create_thread_channel("proxy")
    # 第二次, 交换发送者和接受者.
    async with provider.arun(chan):
        async with proxy.bootstrap() as proxy_runtime:
            # proxy 侧后运行, 这时 provider 已经开始监听了. 要在建连后重新开始监听.
            await proxy_runtime.wait_connected()

            async with provider.runtime.topic_subscriber(LogTopic) as subscriber:
                poll_task = asyncio.create_task(subscriber.poll_model())
                # proxy 侧发送.
                async with proxy_runtime.topic_publisher() as publisher:
                    for i in range(10):
                        await asyncio.sleep(0.0)
                        topic = LogTopic(level="info", message=str(i))
                        # 关键在这里, topic 改成 local 类型.
                        topic.meta.local = True
                        await publisher.pub(topic)
                await asyncio.sleep(0.1)

                # 仍然没有收到.
                assert not poll_task.done()


@pytest.mark.asyncio
async def test_thread_channel_shell():
    chan = PyChannel(name="provider")

    res = []
    @chan.build.command()
    async def foo():
        res.append("1")

    provider, proxy_chan = create_thread_channel("proxy")
    async with provider.arun(chan):
        shell = new_ctml_shell()
        shell.main_channel.import_channels(proxy_chan)
        async with shell:
            await shell.wait_connected("proxy")
            metas = shell.channel_metas()
            assert 'proxy' in metas
            shell_commands = shell.commands()
            assert "proxy" in shell_commands

            async with shell.interpreter_in_ctx() as interpreter:
                interpreter.feed("<proxy:foo/>")
                interpreter.commit()
                tasks = await interpreter.wait_tasks()
                assert len(tasks) == 1
                interpreter.raise_exception()

    assert res == ["1"]
