import asyncio

from ghoshell_moss.channels.thread_channel import create_thread_channel
from ghoshell_moss.channels.py_channel import PyChannel
from ghoshell_moss.concepts.command import Command, CommandError
import pytest


@pytest.mark.asyncio
async def test_thread_channel_start_and_close():
    server, proxy = create_thread_channel("client")
    chan = PyChannel(name="server")
    async with server.run_in_ctx(chan):
        assert chan.is_running()
    assert not chan.is_running()
    assert not server.is_running()


@pytest.mark.asyncio
async def test_thread_channel_raise_in_proxy():
    server, proxy = create_thread_channel("client")
    chan = PyChannel(name="server")
    # 测试 channel 能够正常被启动.
    async with server.run_in_ctx(chan):
        with pytest.raises(RuntimeError):
            async with proxy.bootstrap():
                raise RuntimeError()


@pytest.mark.asyncio
async def test_thread_channel_run_in_thread():
    server, proxy = create_thread_channel("client")
    chan = PyChannel(name="server")
    server.run_in_thread(chan)

    await server.aclose()
    await server.wait_closed()
    assert not chan.is_running()
    assert not server.is_running()


@pytest.mark.asyncio
async def test_thread_channel_run_in_tasks():
    server, proxy = create_thread_channel("client")
    chan = PyChannel(name="server")
    server_run_task = asyncio.create_task(server.arun_until_closed(chan))

    async def _cancel():
        await asyncio.sleep(0.2)
        await server.aclose()

    # 0.2 秒后关闭 server run task
    await asyncio.gather(server_run_task, _cancel())
    assert not server.is_running()
    await server.wait_closed()
    assert server_run_task.done()
    await server_run_task
    server.run_in_thread(chan)

    await server.aclose()
    await server.wait_closed()
    assert not chan.is_running()
    assert not server.is_running()


@pytest.mark.asyncio
async def test_thread_channel_baseline():
    async def foo() -> int:
        return 123

    async def bar() -> int:
        return 456

    chan = PyChannel(name="server")
    # server channel 注册 foo.
    foo_cmd: Command = chan.build.command(return_command=True)(foo)
    assert isinstance(foo_cmd, Command)
    a_chan = chan.new_child("a")
    # a_chan 增加 command bar.
    a_chan.build.command()(bar)

    server, proxy_chan = create_thread_channel("client")

    # 在另一个线程中运行.
    async with server.run_in_ctx(chan):
        async with proxy_chan.bootstrap():
            meta = proxy_chan.broker.meta()
            assert meta is not None
            # 名字被替换了.
            assert meta.name == "client"
            # 存在目标命令.
            assert len(meta.commands) == 1
            foo_cmd_meta = meta.commands[0]
            # 服务端和客户端的 command 使用的 chan 会变更
            # client.a / client.b
            assert foo_cmd_meta.name == foo_cmd.meta().name
            assert foo_cmd_meta.chan == "client"
            assert foo_cmd.meta().chan == "server"

            # 判断仍然有一个子 channel.
            assert "a" in chan.children()
            assert "a" in proxy_chan.children()
            assert chan.broker.meta().name == "server"
            assert proxy_chan.broker.meta().name == "client"

            # 获取这个子 channel, 它应该已经启动了.
            a_chan = chan.get_channel('a')
            assert a_chan is not None
            assert a_chan.is_running()

            # 客户端仍然可以调用命令.
            proxy_side_foo = proxy_chan.broker.get_command("foo")
            assert proxy_side_foo is not None
            meta = proxy_side_foo.meta()
            # 这里虽然来自 server, 但是 chan 被改写成了 client.
            assert meta.chan == "client"
            result = await proxy_side_foo()
            assert result == 123
        assert not proxy_chan.is_running()
    assert not server.is_running()


@pytest.mark.asyncio
async def test_thread_channel_lost_connection():
    async def foo() -> int:
        return 123

    chan = PyChannel(name="server")
    chan.build.command(return_command=True)(foo)
    server, proxy = create_thread_channel("client")
    server.run_in_thread(chan)
    await asyncio.sleep(0.1)

    # 启动 proxy
    async with proxy.bootstrap():
        # 验证连接正常
        assert proxy.is_running()

        # 模拟连接中断（通过关闭 server）
        server.close()
        assert proxy.is_running()
        foo = proxy.broker.get_command("foo")
        # 中断后抛出 command error.
        with pytest.raises(CommandError):
            result = await foo()
        assert not proxy.is_running()
