import asyncio

from ghoshell_moss.channels.thread_channel import create_thread_channel
from ghoshell_moss.channels.py_channel import PyChannel
from ghoshell_moss.concepts.command import Command
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
    task = asyncio.create_task(server.arun_until_closed(chan))

    async def _cancel():
        await asyncio.sleep(0.2)
        await server.aclose()

    await asyncio.gather(task, _cancel())
    assert not server.is_running()
    await server.wait_closed()
    assert task.done()
    await task
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
    foo_cmd: Command = chan.build.command(return_command=True)(foo)
    a_chan = chan.new_child("a")
    a_chan.build.command()(bar)

    server, proxy = create_thread_channel("client")

    # 在另一个线程中运行.
    async with server.run_in_ctx(chan):
        async with proxy.bootstrap():
            meta = proxy.client.meta()
            assert meta is not None
            # 名字被替换了.
            assert meta.name == "client"
            # 存在目标命令.
            assert len(meta.commands) == 1
            foo_cmd_meta = meta.commands[0]
            # 服务端和客户端的 command 使用的 chan 会变更
            assert foo_cmd_meta.name == foo_cmd.meta().name
            assert foo_cmd_meta.chan == "client"
            assert foo_cmd.meta().chan == "server"

            # 判断仍然有一个子 channel.
            assert "a" in chan.children()
            assert "a" in proxy.children()

            # 客户端仍然可以调用命令.
            proxy_side_foo = proxy.client.get_command("foo")
            assert proxy_side_foo is not None
            meta = proxy_side_foo.meta()
            # 这里虽然来自 server, 但是 chan 被改写成了 client.
            assert meta.chan == "client"
            result = await proxy_side_foo()
            assert result == 123
        assert not proxy.is_running()
    assert not server.is_running()
