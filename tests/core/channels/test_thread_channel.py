import asyncio

import pytest

from ghoshell_moss.core import Command, CommandError
from ghoshell_moss.core.duplex.thread_channel import create_thread_channel
from ghoshell_moss.core.py_channel import PyChannel


@pytest.mark.asyncio
async def test_thread_channel_start_and_close():
    provider, proxy = create_thread_channel("proxy")
    chan = PyChannel(name="provider")
    async with provider.arun(chan):
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

    provider, proxy_chan = create_thread_channel("proxy")

    # 在另一个线程中运行.
    async with provider.arun(chan):
        # 判断 channel 已经启动.
        main_broker = provider.broker
        metas = main_broker.metas()
        assert len(metas) == 2
        assert 'a' in metas
        assert main_broker.name == "provider"
        assert main_broker.is_running()
        assert main_broker.is_connected()
        assert main_broker.is_running()
        proxy_side_foo_meta = main_broker.self_meta()
        assert proxy_side_foo_meta.available
        assert len(proxy_side_foo_meta.commands) > 0
        assert proxy_side_foo_meta.name == "provider"

        async with proxy_chan.bootstrap() as proxy_broker:
            await proxy_broker.wait_connected()
            await proxy_broker.refresh_metas()
            metas = proxy_broker.metas()
            assert len(metas) == 2
            # 阻塞等待连接成功.
            proxy_meta = proxy_broker.self_meta()
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
            metas = proxy_broker.metas()
            assert "a" in metas
            assert main_broker.self_meta().name == "provider"
            assert proxy_meta.name == "proxy"

            # 客户端仍然可以调用命令.
            proxy_side_foo = proxy_broker.get_self_command("foo")
            assert proxy_side_foo is not None

            result = await proxy_side_foo()
            assert result == 123

        assert not proxy_broker.is_running()
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
        async with proxy.bootstrap() as proxy_broker:
            await proxy_broker.wait_connected()
            # 验证连接正常
            assert proxy_broker.is_running()
            _foo = proxy_broker.get_self_command("foo")
            assert _foo is not None

            # 模拟连接中断（通过关闭 provider）
            provider.close()
            assert not provider.is_running()
            assert proxy_broker.is_running()
            _foo = proxy_broker.get_self_command("foo")
            # 中断后抛出 command error.
            with pytest.raises(CommandError):
                result = await _foo()
            assert not proxy_broker.is_running()

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

            foo = runtime.get_self_command("foo")
            assert "hello" in foo.meta().interface

            foo_doc = "world"
            generated_foo_doc = doc_fn()
            assert generated_foo_doc == foo_doc

            # 没有立刻变更:
            foo1 = runtime.get_self_command("foo")
            assert foo1 is not None
            assert "hello" in foo1.meta().interface

            # 刷新了 meta 才会变更.
            await runtime.refresh_metas()
            foo2 = runtime.get_self_command("foo")

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

            assert "sub1" in runtime.metas()
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
    async with proxy.bootstrap() as proxy_broker:
        await proxy_broker.wait_connected()
        assert proxy_broker.is_available()
        assert proxy_broker.is_running()
        _foo = proxy_broker.get_self_command("foo")
        with pytest.raises(CommandError):
            await _foo()

    provider.close()
    await provider.wait_closed()


@pytest.mark.asyncio
async def test_thread_channel_idle():
    chan = PyChannel(name="provider")

    idled = []

    @chan.build.command()
    async def foo() -> int:
        return 123

    @chan.build.idle
    async def idle():
        idled.append(True)

    provider, proxy = create_thread_channel("proxy")
    provider.run_in_thread(chan)
    try:
        async with proxy.bootstrap() as proxy_broker:
            await proxy_broker.wait_connected()
            assert proxy_broker.is_idle()
            assert provider.broker.is_idle()
            assert len(idled) == 1

            r = await proxy_broker.execute_command("foo")
            assert r == 123
            assert proxy_broker.is_idle()
            assert provider.broker.is_idle()
            assert len(idled) == 2

    finally:
        provider.close()
    await provider.wait_closed()
