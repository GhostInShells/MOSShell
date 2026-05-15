import pytest
from ghoshell_moss.core.ctml.shell import new_ctml_shell
from ghoshell_moss.core.duplex.thread_channel import create_thread_bridge
from ghoshell_moss.core import PyChannel


@pytest.mark.asyncio
async def test_shell_with_virtual_sub_depth_channel():
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
            for path, meta in shell.channel_metas().items():
                if path == 'proxy':
                    assert len(meta.proxy) == 3
                assert meta.available
                for command in meta.commands:
                    assert command.available
                count += 1
            assert count == 5
            assert 'virtual_sub' in provider.runtime.commands()
            assert 'foo' in provider.runtime.commands()['virtual_sub']
            cmd = provider.runtime.get_command("virtual_sub:foo")
            assert cmd is not None

            # 少一个节点.
            virtual_sub.remove_virtual_channel(virtual_sub_depth_2.name())
            await shell.refresh_metas()
            assert len(shell.channel_metas()) == 4

            # cmd = shell.runtime.get_command('proxy.virtual_sub:foo')
            # assert cmd is not None
            # assert await cmd() == 123

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
