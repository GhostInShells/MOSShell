import asyncio

import pytest

from ghoshell_moss.core.concepts.command import Command, CommandError
from ghoshell_moss.core.duplex.thread_channel import create_thread_channel
from ghoshell_moss.core.py_channel import PyChannel


@pytest.mark.asyncio
async def test_thread_channel_start_and_close():
    provider, proxy = create_thread_channel("client")
    chan = PyChannel(name="provider")
    async with provider.run_in_ctx(chan):
        assert chan.is_running()
    assert not chan.is_running()
    assert not provider.is_running()


@pytest.mark.asyncio
async def test_thread_channel_raise_in_proxy():
    provider, proxy = create_thread_channel("client")
    chan = PyChannel(name="provider")
    # 测试 channel 能够正常被启动.
    async with provider.run_in_ctx(chan):
        with pytest.raises(RuntimeError):
            async with proxy.bootstrap():
                raise RuntimeError()


@pytest.mark.asyncio
async def test_thread_channel_run_in_thread():
    provider, proxy = create_thread_channel("client")
    chan = PyChannel(name="provider")
    provider.run_in_thread(chan)

    await provider.aclose()
    await provider.wait_closed()
    assert not chan.is_running()
    assert not provider.is_running()


@pytest.mark.asyncio
async def test_thread_channel_run_in_tasks():
    provider, proxy = create_thread_channel("client")
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

    provider, proxy_chan = create_thread_channel("client")

    # 在另一个线程中运行.
    async with provider.run_in_ctx(chan):
        # 判断 channel 已经启动.
        assert chan.is_running()
        assert chan.broker.is_connected()
        assert chan.broker.is_running()
        meta = chan.broker.meta()
        assert meta.available
        assert len(meta.commands) > 0
        assert meta.name == "provider"

        async with proxy_chan.bootstrap():
            # 阻塞等待连接成功.
            await proxy_chan.broker.wait_connected()
            meta = proxy_chan.broker.meta()
            assert meta is not None
            # 名字被替换了.
            assert meta.name == "client"
            assert meta.available is True
            # 存在目标命令.
            assert len(meta.commands) == 1
            foo_cmd_meta = meta.commands[0]
            # 服务端和客户端的 command 使用的 chan 会变更
            # client.a / client.b
            assert foo_cmd_meta.name == foo_cmd.meta().name
            assert foo_cmd_meta.chan == "client"
            assert foo_cmd.meta().chan == "provider"

            # 判断仍然有一个子 channel.
            assert "a" in chan.children()
            assert "a" in proxy_chan.children()
            assert chan.broker.meta().name == "provider"
            assert proxy_chan.broker.meta().name == "client"

            # 获取这个子 channel, 它应该已经启动了.
            a_chan = chan.get_channel("a")
            assert a_chan is not None
            assert a_chan.is_running()

            # 客户端仍然可以调用命令.
            proxy_side_foo = proxy_chan.broker.get_command("foo")
            assert proxy_side_foo is not None
            meta = proxy_side_foo.meta()
            # 这里虽然来自 provider, 但是 chan 被改写成了 client.
            assert meta.chan == "client"
            result = await proxy_side_foo()
            assert result == 123
        assert not proxy_chan.is_running()
    assert not provider.is_running()


@pytest.mark.asyncio
async def test_thread_channel_lost_connection():
    async def foo() -> int:
        return 123

    chan = PyChannel(name="provider")
    chan.build.command(return_command=True)(foo)
    provider, proxy = create_thread_channel("client")
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

    provider, proxy = create_thread_channel("client")
    provider.run_in_thread(chan)

    async with proxy.bootstrap():
        await proxy.broker.wait_connected()
        # 验证连接正常
        assert proxy.is_running()

        foo = proxy.broker.get_command("foo")
        assert "hello" in foo.meta().interface

        foo_doc = "world"

        # 没有立刻变更:
        foo1 = proxy.broker.get_command("foo")
        assert "hello" in foo1.meta().interface

        await proxy.broker.refresh_meta()
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

    provider, proxy = create_thread_channel("client")
    provider.run_in_thread(chan)
    async with proxy.run_in_ctx():
        assert proxy.is_running()
        await proxy.broker.wait_connected()
        assert "sub1" in proxy.children()
        # 判断子 channel 存在.
        _sub1 = proxy.get_channel("sub1")
        assert _sub1 is not None
        assert sub1.is_running()
        bar = sub1.broker.get_command("bar")
        value = await sub1.execute_command(bar)
        assert value == 456

    provider.close()
    await provider.wait_closed()


@pytest.mark.asyncio
async def test_thread_channel_exception():
    chan = PyChannel(name="provider")

    @chan.build.command()
    async def foo() -> int:
        raise ValueError("foo")

    provider, proxy = create_thread_channel("client")
    provider.run_in_thread(chan)
    async with proxy.run_in_ctx():
        await proxy.broker.wait_connected()
        assert proxy.broker.is_available()
        assert proxy.is_running()
        _foo = proxy.broker.get_command("foo")
        with pytest.raises(CommandError):
            await _foo()

    provider.close()
    await provider.wait_closed()
