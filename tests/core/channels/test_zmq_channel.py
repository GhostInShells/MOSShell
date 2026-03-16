import asyncio

import pytest

from ghoshell_moss import PyChannel, Command, new_ctml_shell
from ghoshell_moss.transports.zmq_channel.zmq_channel import create_zmq_channel


@pytest.mark.asyncio
async def test_zmq_channel_baseline():
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

    provider, proxy_chan = create_zmq_channel("proxy", address="tcp://0.0.0.0:9527")

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


@pytest.mark.asyncio
async def test_zmq_channel_shell():
    chan = PyChannel(name="provider")

    res = []
    @chan.build.command()
    async def foo():
        res.append("1")

    provider, proxy_chan = create_zmq_channel("proxy", address="tcp://0.0.0.0:9527")
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
