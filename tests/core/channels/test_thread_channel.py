import asyncio

import pytest

from ghoshell_moss.core import Command, CommandError
from ghoshell_moss.core.duplex.thread_channel import create_thread_channel
from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.core.concepts.runtime import ChannelTreeRuntime


@pytest.mark.asyncio
async def test_thread_channel_start_and_close():
    provider, proxy = create_thread_channel("proxy")
    chan = PyChannel(name="provider")
    async with provider.run_in_ctx(chan):
        broker = provider.broker
        assert broker is not None
        assert broker.is_running()
    assert not broker.is_running()
    assert not provider.is_running()


@pytest.mark.asyncio
async def test_thread_channel_raise_in_proxy():
    provider, proxy = create_thread_channel("proxy")
    chan = PyChannel(name="provider")
    # 测试 channel 能够正常被启动.
    async with provider.run_in_ctx(chan):
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
    assert not chan.is_running()
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
    provider.run_in_thread(chan)

    await provider.aclose()
    await provider.wait_closed()
    assert not chan.is_running()
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

    assert len(chan.all_channels()) == 2
    assert 'a' in chan.all_channels()

    provider, proxy_chan = create_thread_channel("proxy")

    # 在另一个线程中运行.
    async with provider.run_in_ctx(chan):
        # 判断 channel 已经启动.
        main_broker = provider.broker
        assert main_broker.name == "provider"
        assert main_broker.is_running()
        assert main_broker.is_connected()
        assert main_broker.is_running()
        proxy_side_foo_meta = main_broker.meta()
        assert proxy_side_foo_meta.available
        assert len(proxy_side_foo_meta.commands) > 0
        assert proxy_side_foo_meta.name == "provider"

        async with ChannelTreeRuntime.bootstrap(proxy_chan) as proxy_runtime:
            await proxy_runtime.broker.wait_connected()
            await proxy_runtime.refresh_all_metas()
            metas = proxy_runtime.metas()
            assert len(metas) == 2
            proxy_broker = proxy_runtime.broker
            # 阻塞等待连接成功.
            await proxy_broker.wait_connected()
            proxy_meta = proxy_broker.meta()
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
            assert foo_cmd_meta.chan == "proxy"
            assert foo_cmd.meta().chan == "provider"

            # 判断仍然有一个子 channel.
            assert "a" in chan.children()
            # 判断 proxy 也有 children
            proxy_chan_children = proxy_chan.children()
            assert "a" in proxy_chan_children
            assert main_broker.meta().name == "provider"
            assert proxy_meta.name == "proxy"

            # 获取这个子 channel, 它应该已经启动了.
            a_chan = chan.get_channel("a")
            assert a_chan is not None
            assert a_chan.is_running()

            # 客户端仍然可以调用命令.
            proxy_side_foo = proxy_broker.get_command("foo")
            assert proxy_side_foo is not None
            proxy_side_foo_meta = proxy_side_foo.meta()
            # 这里虽然来自 provider, 但是 chan 被改写成了 proxy.
            assert proxy_side_foo_meta.chan == "proxy"
            result = await proxy_side_foo()
            assert result == 123
        assert not proxy_broker.is_running()
    assert not provider.is_running()


@pytest.mark.asyncio
async def test_thread_channel_lost_connection():
    async def foo() -> int:
        return 123

    chan = PyChannel(name="provider")
    chan.build.command(return_command=True)(foo)
    provider, proxy = create_thread_channel("proxy")
    provider.run_in_thread(chan)
    await asyncio.sleep(0.1)

    # 启动 proxy
    async with proxy.bootstrap():
        await proxy.broker.wait_connected()
        # 验证连接正常
        assert proxy.is_running()

        # 模拟连接中断（通过关闭 provider）
        provider.close()
        assert proxy.is_running()
        foo = proxy.broker.get_command("foo")
        # 中断后抛出 command error.
        with pytest.raises(CommandError):
            result = await foo()
        assert not proxy.is_running()


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
    provider.run_in_thread(chan)

    async with ChannelTreeRuntime.bootstrap(proxy) as runtime:
        await runtime.wait_connected()
        # 验证连接正常
        assert runtime.broker.is_running()

        foo = runtime.get_command("foo")
        assert "hello" in foo.meta().interface

        foo_doc = "world"

        # 没有立刻变更:
        foo1 = runtime.get_command("foo")
        assert "hello" in foo1.meta().interface

        await runtime.refresh_all_metas()
        foo2 = proxy.broker.get_command("foo")

        assert foo2 is not foo1
        assert "hello" not in foo2.meta().interface
        assert "world" in foo2.meta().interface
    provider.close()
    await provider.wait_closed()


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
        async with ChannelTreeRuntime.bootstrap(proxy) as runtime:
            assert runtime.is_running()
            await runtime.wait_connected()

            assert "sub1" in proxy.children()
            # # 判断子 channel 存在.
            _sub1_runtime = await runtime.fetch_node("sub1")
            assert _sub1_runtime is not None
            assert _sub1_runtime.is_running()
            value = await _sub1_runtime.execute_command("bar")
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
    async with proxy.bootstrap() as proxy_broker:
        await proxy_broker.wait_connected()
        assert proxy_broker.is_available()
        assert proxy_broker.is_running()
        _foo = proxy_broker.get_command("foo")
        with pytest.raises(CommandError):
            await _foo()

    provider.close()
    await provider.wait_closed()
