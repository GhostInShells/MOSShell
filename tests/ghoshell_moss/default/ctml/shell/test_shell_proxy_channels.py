import pytest
from ghoshell_moss.core.ctml.shell import new_ctml_shell
from ghoshell_moss.core.duplex.thread_channel import create_thread_bridge
from ghoshell_moss.core.py_channel import PyChannel
import asyncio


@pytest.mark.asyncio
async def test_shell_with_virtual_sub_depth_channel():
    from ghoshell_moss.core.runtime import BaseChannelTree
    provider_main = PyChannel(name="provider")
    static_sub = PyChannel(name="static_sub")
    virtual_sub = PyChannel(name="virtual_sub")

    @virtual_sub.build.command()
    async def foo():
        return 123

    virtual_sub_depth_2 = PyChannel(name="virtual_sub_depth_2")

    provider_main.import_channels(static_sub)

    provider, proxy = create_thread_bridge('proxy')
    shell = new_ctml_shell("test")
    shell.main_channel.import_channels(proxy)

    async with provider.arun(provider_main):
        async with shell:
            tree = shell.runtime.tree
            assert isinstance(tree, BaseChannelTree)
            tree.config.node_refresh_interval = 0.0
            assert len(tree.metas()) == 2
            # 方便快速验证.
            await shell.wait_connected("proxy")
            assert len(shell.channel_metas()) == 3
            # 添加动态
            provider_main.add_virtual_channel(virtual_sub)
            await shell.refresh_metas()
            # 拿到了新的节点.
            assert len(shell.channel_metas()) == 4
            virtual_sub.add_virtual_channel(virtual_sub_depth_2)
            # 继续添加动态节点.
            await shell.refresh_metas()
            metas = shell.channel_metas()
            assert len(metas) == 5
            assert len(shell.channel_metas()) == 5
            commands = shell.commands()
            assert 'proxy.virtual_sub' in commands
            assert 'foo' in commands['proxy.virtual_sub']
            count = 0
            command = await shell.get_command("proxy.virtual_sub", "foo")
            assert command is not None
            assert command.meta().available

            # 判断 provider 和 proxy 都有正确的命令.
            proxy_count = 0
            for path, meta in shell.channel_metas().items():
                if path.startswith("proxy"):
                    proxy_count += 1
                    assert meta.proxy
                assert meta.available
                for command in meta.commands:
                    assert command.available
                count += 1
            assert count == 5
            assert shell.runtime.self_meta().proxy is False
            assert proxy_count == 4
            assert 'virtual_sub' in provider.runtime.commands()
            assert 'foo' in provider.runtime.commands()['virtual_sub']
            cmd = provider.runtime.get_command("virtual_sub:foo")
            assert cmd is not None

            # 少一个节点.
            virtual_sub.remove_virtual_channel(virtual_sub_depth_2.name())
            await shell.refresh_metas()
            assert len(shell.channel_metas()) == 4

            async with shell.interpreter_in_ctx() as i:
                i.feed("<proxy.virtual_sub:foo />")
                i.commit()
                await i.wait_compiled()
                tasks = await i.wait_tasks()
                assert len(tasks) == 1
                t = list(tasks.values())[0]
                e = t.exception()
                assert e is None
                assert t.success()
                assert t.result() == 123

            command_group = shell.commands()
            assert 'proxy.virtual_sub' in command_group
            assert 'foo' in command_group['proxy.virtual_sub']


@pytest.mark.asyncio
async def test_shell_proxy_delta_calls_in_double_proxy():
    provider_1_main = PyChannel(name="provider1")
    provider_1, proxy_1 = create_thread_bridge('proxy_1')
    provider_2_main = PyChannel(name="provider2")

    provider_2_main.import_channels(proxy_1)
    provider_2, proxy_2 = create_thread_bridge('proxy_2')

    shell = new_ctml_shell(
        "test",
    )
    shell.main_channel.import_channels(proxy_2)

    got = ''

    @provider_1_main.build.command()
    async def chunks(chunks__):
        nonlocal got
        async for c in chunks__:
            got += c

    # provider 1 将 provider 1 main 提供出来,
    async with provider_1.arun(provider_1_main):
        # 启动 provider 2 main 时启动了 proxy.
        async with provider_2.arun(provider_2_main):
            async with shell:
                await shell.wait_connected("proxy_2")
                async with shell.interpreter_in_ctx() as i:
                    i.feed("<proxy_2.proxy_1:chunks>hello world</proxy_2.proxy_1:chunks>")
                    i.commit()
                    await i.wait_stopped()
                    i.raise_exception()
    assert got == 'hello world'


@pytest.mark.asyncio
async def test_shell_proxy_channel_with_other_magic_command():
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    got = ''

    @provider_main.build.command()
    async def __hello__():
        nonlocal got
        got = "world"
        return got

    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy)

    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            # 魔术方法不对外暴露即可.
            for msg in shell.dynamic_messages():
                assert "__hello__" not in msg.to_content_string()

            assert "__hello__" not in shell.static_messages()
            assert "__hello__" not in shell.meta_instruction()


@pytest.mark.asyncio
async def test_shell_proxy_channel_with_magic_delta_calls():
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    got = ''

    @provider_main.build.command()
    async def __magic__(chunks__):
        nonlocal got
        async for c in chunks__:
            got += c

    shell = new_ctml_shell(
        "test",
    )
    shell.main_channel.import_channels(proxy)

    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            assert "proxy" in shell.commands()
            async with shell.interpreter_in_ctx() as i:
                i.feed("<proxy:__magic__>hello world</proxy:__magic__>")
                i.commit()
                await i.wait_tasks()
    assert got == 'hello world'


@pytest.mark.asyncio
async def test_shell_proxy_channel_with_delta_calls():
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    got = ''

    @provider_main.build.command()
    async def chunks(chunks__):
        nonlocal got
        async for c in chunks__:
            got += c

    shell = new_ctml_shell(
        "test",
    )
    shell.main_channel.import_channels(proxy)

    errors = []

    def report(err):
        errors.append(err)

    provider.on_error(report)

    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            async with shell.interpreter_in_ctx() as i:
                i.feed("<proxy:chunks>hello world</proxy:chunks>")
                i.commit()
                try:
                    await i.wait_compiled()
                except Exception as e:
                    print(e)
                tasks = i.compiled_tasks()
                for task in tasks.values():
                    assert task.meta.available
                assert len(tasks) == 1
                await shell.clear()
                interpretation = await i.wait_stopped()
                assert len(interpretation.failed_tasks) == 0
    assert len(errors) == 0


@pytest.mark.asyncio
async def test_shell_proxy_channel_with_remote_content_command():
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    got = ''

    @provider_main.build.content_command
    async def chunks(chunks__):
        nonlocal got
        async for c in chunks__:
            got += c

    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy)
    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            await shell.refresh_metas()
            metas = shell.channel_metas()
            assert 'proxy' in metas
            async with shell.interpreter_in_ctx() as i:
                i.feed("<proxy:__content__>hello world</proxy:__content__>")
                i.commit()
                await i.wait_tasks()
    assert got == 'hello world'


@pytest.mark.asyncio
async def test_shell_proxy_channel_with_content_command_that_not_exists():
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy)

    errors = []

    def print_error(err):
        errors.append(err)

    provider.on_error(print_error)
    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            await shell.refresh_metas()
            metas = shell.channel_metas()
            assert 'proxy' in list(metas.keys())
            async with shell.interpreter_in_ctx() as i:
                i.feed("<proxy:__content__>hello world</proxy:__content__>")
                i.commit()
                tasks = await i.wait_tasks(timeout=2, throw_task_error=True)
                for task in tasks.values():
                    assert task.exception() is None, "++++++ the task is:" + task.caller_name()
                i.raise_exception()

    assert len(errors) == 0


@pytest.mark.asyncio
async def test_shell_proxy_channel_with_content_command_by_scope():
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy)

    errors = []

    got = ''

    @provider_main.build.content_command
    async def chunks(chunks__):
        try:
            nonlocal got
            async for c in chunks__:
                got += c
        except asyncio.CancelledError:
            got = 'canceled'

    def get_error(err):
        errors.append(err)

    proxy_events = []

    def get_proxy_event(e):
        proxy_events.append(e)

    provider.on_error(get_error)
    provider.on_proxy_event(get_proxy_event)
    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            await shell.refresh_metas()
            metas = shell.channel_metas()
            assert 'proxy' in metas
            async with shell.interpreter_in_ctx() as i:
                i.feed("<_ channel='proxy'>hello world</_>")
                i.commit()
                tasks = await i.wait_tasks(timeout=10)
                assert len(tasks) == 3
                for t in tasks.values():
                    assert t.chan == 'proxy', t.caller_name() + " failed"
                i.raise_exception()

    assert len(errors) == 0
    assert got == 'hello world'


@pytest.mark.asyncio
async def test_shell_proxy_channel_with_scope_call():
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    got = []

    @provider_main.build.command()
    async def foo():
        got.append(1)
        await asyncio.sleep(0.1)
        return "hello"

    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy)

    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            assert provider.runtime.is_running()
            assert shell.runtime.is_running()
            proxy_runtime = shell.runtime.fetch_sub_runtime('proxy')
            assert proxy_runtime.is_running()

            assert len(got) == 0
            async with shell.interpreter_in_ctx() as i:
                i.feed("<_ channel='proxy' until='any'><foo /><foo /><foo /></_>")
                i.commit()
                tasks = await i.wait_tasks(timeout=1)
                i.raise_exception()
            assert len(got) == 1
            await shell.clear()

            got.clear()
            async with shell.interpreter_in_ctx() as i:
                i.feed("<_ channel='proxy' until='all'><foo /><foo /><foo /></_>")
                i.commit()
                tasks = await i.wait_tasks(timeout=1)
                i.raise_exception()
            assert len(got) == 3
            await shell.clear()

            got.clear()
            async with shell.interpreter_in_ctx() as i:
                i.feed("<_ channel='proxy' until='all'> <_><foo /></_> <foo />  <_><foo /></_>  </_> <proxy:foo />")
                i.commit()
                tasks = await i.wait_tasks(timeout=1)
                for task in tasks.values():
                    if exp := task.exception():
                        print(repr(exp))
                i.raise_exception()
            assert len(got) == 4


@pytest.mark.asyncio
async def test_remote_scope_timeout_cancels_proxy_command():
    """远程 scope timeout — proxy 侧下发超时作用域，provider 侧命令被取消"""
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    cancelled = False

    @provider_main.build.command()
    async def slow_work():
        nonlocal cancelled
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            cancelled = True
            raise

    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy)

    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            async with shell.interpreter_in_ctx() as i:
                # timeout 极短，slow_work 来不及完成
                i.feed("<_ channel='proxy' timeout='0.01'><slow_work /></_>")
                i.commit()
                await i.wait_tasks(timeout=2)
            await shell.clear()

    assert cancelled is True


@pytest.mark.asyncio
async def test_main_scope_run_remote_task():
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    got = []

    @provider_main.build.command()
    async def foo() -> str:
        got.append(1)
        await asyncio.sleep(0.05)
        got.append(1)
        return "hello"

    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy)

    @shell.main_channel.build.content_command
    async def content(chunks__):
        await asyncio.sleep(0.01)
        return None

    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            async with shell.interpreter_in_ctx() as i:
                i.feed("<proxy:foo/>")
                i.commit()
                tasks = await i.wait_tasks(timeout=2)
                i.raise_exception()
                assert len(tasks) == 1
                assert len(got) == 2
                got.clear()
            async with shell.interpreter_in_ctx() as i:
                i.feed("<_><proxy:foo/>hello</_>")
                i.commit()
                tasks = await i.wait_tasks(timeout=2)
                for t in tasks.values():
                    assert t.done()
                i.raise_exception()
            assert len(got) == 1
            got.clear()

            async with shell.interpreter_in_ctx() as i:
                i.feed("<_><proxy:foo/></_>")
                i.commit()
                tasks = await i.wait_tasks(timeout=2)
            assert len(tasks) >= 3
            for t in tasks.values():
                assert t.success()
            assert len(got) == 2


@pytest.mark.asyncio
async def test_remote_scope_until_any_cancels_slower_task():
    """远程 scope until='any' — 快命令先完成，慢命令被 scope 取消"""
    provider_main = PyChannel(name="provider")
    provider, proxy = create_thread_bridge('proxy')

    fast_done = False
    slow_completed = False

    @provider_main.build.command()
    async def fast():
        nonlocal fast_done
        await asyncio.sleep(0.01)
        fast_done = True

    @provider_main.build.command()
    async def slow():
        nonlocal slow_completed
        # 长时任务 — 在 until='any' 下应该被 scope 取消，不会跑完
        await asyncio.sleep(1.0)
        slow_completed = True

    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy)

    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            async with shell.interpreter_in_ctx() as i:
                i.feed("<_ channel='proxy' until='any'><fast /><slow /></_>")
                i.commit()
                tasks = await i.wait_tasks(timeout=2)
                i.raise_exception()
            await shell.clear()

    assert fast_done is True
    # slow 被 scope 标记取消，不会完成
    assert slow_completed is False


@pytest.mark.asyncio
async def test_remote_two_proxy_channels_parallel():
    """两个 proxy channel 并行执行 — a 和 b 的命令互不阻塞"""
    provider_a = PyChannel(name="pa")
    provider_b = PyChannel(name="pb")
    p_a, proxy_a = create_thread_bridge('proxy_a')
    p_b, proxy_b = create_thread_bridge('proxy_b')

    order = []

    @provider_a.build.command()
    async def cmd_a():
        await asyncio.sleep(0.03)
        order.append('a')

    @provider_b.build.command()
    async def cmd_b():
        await asyncio.sleep(0.01)
        order.append('b')

    shell = new_ctml_shell()
    shell.main_channel.import_channels(proxy_a)
    shell.main_channel.import_channels(proxy_b)

    async with p_a.arun(provider_a):
        async with p_b.arun(provider_b):
            async with shell:
                await shell.wait_connected("proxy_a", "proxy_b")
                async with shell.interpreter_in_ctx() as i:
                    i.feed("<proxy_a:cmd_a /><proxy_b:cmd_b />")
                    i.commit()
                    await i.wait_tasks(timeout=2)
                await shell.clear()

    assert order == ['b', 'a']


# -- stale_time / refresh interval ----------------------------------------


@pytest.mark.asyncio
async def test_refresh_metas_stale_protection_skips_redundant_refresh():
    """node_refresh_interval 保护期内 refresh_metas 不触发新的刷新周期。

    即便 interval 为 0 时，shell 层的 stale_time 也能起到保护；
    这里通过 tree config 的保护期验证等价的跳过逻辑。
    """
    shell = new_ctml_shell()
    chan = PyChannel(name="chan")
    shell.main_channel.import_channels(chan)
    invoked: list[int] = []

    @chan.build.refresh_meta
    async def count_refresh() -> None:
        invoked.append(1)

    async with shell:
        assert len(invoked) >= 1  # bootstrap 触发了一次

        # 设保护期 10s，多次调用应全部跳过
        invoked.clear()
        refreshing_tasks = []
        for _ in range(3):
            refreshing_tasks.append(shell.refresh_metas())
        # 并发等所有任务刷新完毕.
        await asyncio.gather(*refreshing_tasks)
        # 多次并发只会有一次刷新成功.
        assert len(invoked) == 1, f"expected 0 refreshes, got {len(invoked)}"
        invoked.clear()
        # 连续等待两次刷新, 应该多刷新了两次.
        await shell.refresh_metas()
        await shell.refresh_metas()
        assert len(invoked) == 2
        # 然后加 stale
        await shell.refresh_metas()
        assert len(invoked) == 3
        await shell.refresh_metas(stale_time=0.1)
        # 不会有新的刷新结果.
        assert len(invoked) == 3
