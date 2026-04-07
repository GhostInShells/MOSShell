from ghoshell_moss.core.py_channel import PyChannel
from ghoshell_moss.core.duplex.suite_for_test import BridgeSuite, ThreadBridgeSuite
from ghoshell_moss.core.concepts.command import CommandError, CommandToken
import pytest
import asyncio

suite_configs = [
    {"name": "thread", "suite": ThreadBridgeSuite()},
]


@pytest.fixture(params=suite_configs, ids=lambda c: c["name"])
def suite(request):
    suite = request.param["suite"]
    yield suite
    suite.cleanup()


@pytest.mark.usefixtures("suite")
class TestBridgeSuite:

    @pytest.mark.asyncio
    async def test_provider_closed(self, suite: BridgeSuite) -> None:
        provider, proxy = suite.create()
        chan = PyChannel(name="provider")

        async with provider.arun(channel=chan):
            assert provider.is_running()
        assert not provider.is_running()

    @pytest.mark.asyncio
    async def test_thread_channel_run_in_thread(self, suite: BridgeSuite) -> None:
        provider, proxy = suite.create()
        chan = PyChannel(name="provider")
        provider.run_in_thread(chan)

        await provider.aclose()
        await provider.wait_closed()
        assert not provider.is_running()

    @pytest.mark.asyncio
    async def test_thread_channel_run_in_tasks(self, suite: BridgeSuite) -> None:
        provider, proxy = suite.create()
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

    @pytest.mark.asyncio
    async def test_thread_channel_run_in_thread_and_aclose(self, suite: BridgeSuite) -> None:
        provider, proxy = suite.create()
        chan = PyChannel(name="provider")
        # 重新创建 provider.
        provider.run_in_thread(chan)
        await provider.aclose()
        await provider.wait_closed()
        assert not provider.is_running()

    @pytest.mark.asyncio
    async def test_thread_channel_baseline(self, suite: BridgeSuite) -> None:
        async def foo() -> int:
            return 123

        async def bar() -> int:
            return 456

        provider_main_chan = PyChannel(name="provider")
        a_chan = PyChannel(name="a")
        # provider channel 注册 foo.
        foo_cmd = provider_main_chan.build.command(return_command=True)(foo)
        provider_main_chan.import_channels(a_chan)
        # a_chan 增加 command bar.
        a_chan.build.command()(bar)

        provider, proxy_chan = suite.create("proxy")

        # 在另一个线程中运行.
        async with provider.arun(provider_main_chan):
            # 判断 channel 已经启动.
            main_runtime = provider.runtime
            metas = main_runtime.metas()
            assert len(metas) == 2
            assert "a" in metas
            assert main_runtime.name == "provider"
            assert main_runtime.is_running()
            assert main_runtime.is_connected()
            assert main_runtime.is_running()
            proxy_side_foo_meta = main_runtime.self_meta()
            assert proxy_side_foo_meta.available
            assert len(proxy_side_foo_meta.commands) > 0
            assert proxy_side_foo_meta.name == "provider"

            async with proxy_chan.bootstrap() as proxy_runtime:
                await proxy_runtime.wait_connected()
                await proxy_runtime.refresh_metas()

                assert proxy_runtime.has_own_command("foo")
                assert proxy_runtime.has_own_command("a:bar")
                commands = proxy_runtime.commands()
                assert 'a' in commands
                assert '' in commands
                assert len(commands['a']) == 1

                metas = proxy_runtime.metas()
                assert len(metas) == 2
                # 阻塞等待连接成功.
                proxy_meta = proxy_runtime.self_meta()
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
                assert "a" in provider_main_chan.children()
                # 判断 proxy 也有 children
                metas = proxy_runtime.metas()
                assert "a" in metas
                assert main_runtime.self_meta().name == "provider"
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

    def test_thread_channel_lost_connection(self, suite: BridgeSuite) -> None:
        async def foo() -> int:
            return 123

        chan = PyChannel(name="provider")
        chan.build.command(return_command=True)(foo)
        provider, proxy = suite.create("proxy")
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
    async def test_thread_channel_refresh_meta(self, suite: BridgeSuite) -> None:
        foo_doc = "hello"

        def doc_fn() -> str:
            return foo_doc

        chan = PyChannel(name="provider")

        @chan.build.command(doc=doc_fn)
        async def foo() -> int:
            return 123

        assert chan.main_state().is_dynamic()
        provider, proxy = suite.create("proxy")

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

                # 这时, provider 侧的runtime 也应该刷新了.
                # assert by state
                foo = chan.main_state().get_own_command("foo")
                assert foo is not None
                assert "world" in foo.meta().interface
                # assert by runtime
                # 这时判断, provider 侧已经更新了.
                provider_metas = provider.runtime.tree.metas()
                assert len(provider_metas) == 1
                assert len(provider_metas[''].commands) == 1
                assert 'world' in provider_metas[''].commands[0].interface

                provider_foo = provider.runtime.get_command("foo")
                assert provider_foo is not None
                assert "world" in provider_foo.meta().interface

                foo2 = runtime.get_command("foo")

                assert foo2 is not foo1
                assert "hello" not in foo2.meta().interface
                assert "world" in foo2.meta().interface

    @pytest.mark.asyncio
    async def test_thread_channel_has_child(self, suite: BridgeSuite) -> None:
        chan = PyChannel(name="provider")

        @chan.build.command()
        async def foo() -> int:
            return 123

        sub1 = PyChannel(name="sub1")
        chan.import_channels(sub1)

        @sub1.build.command()
        async def bar() -> int:
            return 456

        provider, proxy = suite.create("proxy")
        provider.run_in_thread(chan)
        try:
            async with proxy.bootstrap() as runtime:
                assert runtime.is_running()
                await runtime.wait_connected()
                metas = runtime.metas()

                assert "sub1" in metas
                sub1_meta = metas["sub1"]
                assert len(sub1_meta.commands) == 1
                # # 判断子 channel 存在.
                value = await runtime.execute_command("sub1:bar")
                assert value == 456
        finally:
            provider.close()
            await provider.wait_closed()

    @pytest.mark.asyncio
    async def test_thread_channel_exception(self, suite: BridgeSuite) -> None:
        chan = PyChannel(name="provider")

        @chan.build.command()
        async def foo() -> int:
            raise ValueError("foo")

        provider, proxy = suite.create("proxy")
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
    async def test_thread_channel_idle(self, suite: BridgeSuite) -> None:
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

        provider, proxy = suite.create("proxy")
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
    async def test_thread_channel_with_delta_func(self, suite: BridgeSuite) -> None:
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

        provider, proxy = suite.create("proxy")
        async with provider.arun(chan):
            async with proxy.bootstrap() as runtime:
                await runtime.wait_connected()
                value = await runtime.execute_command("chunks", kwargs=dict(chunks__=generate()))
                assert value == 10
                value = await runtime.execute_command("text", kwargs=dict(text__="hello"))
                assert value == "hello"
                value = await runtime.execute_command("tokens", kwargs=dict(tokens__=generate_tokens()))
                assert value == 10
