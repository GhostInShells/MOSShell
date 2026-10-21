import pytest

from ghoshell_moss.core import PyChannel, new_ctml_shell
from ghoshell_moss.core.duplex.thread_channel import create_thread_bridge


@pytest.mark.asyncio
async def test_all_scope_with_proxied_child_scope_finishes():
    """<all> 套一个 **远端 (proxy) 子 channel** 的 scope 时必须能收口.

    D21 17:13 的真实现场与本地子 channel 的区别就在这里: `matrix.mesh.webview_screen`
    是 duplex 代理, scope 的 enter/exit 要跨桥走一趟.
    """
    provider_main = PyChannel(name="provider")
    child = PyChannel(name="child")

    @child.build.command()
    async def quick() -> int:
        return 1

    provider_main.import_channels(child)

    provider, proxy = create_thread_bridge("proxy")
    shell = new_ctml_shell("scope-all-proxy")
    shell.main_channel.import_channels(proxy)

    async with provider.arun(provider_main):
        async with shell:
            await shell.wait_connected("proxy")
            async with shell.interpreter_in_ctx(clear_after_exit=True) as interpreter:
                interpreter.feed("<all><proxy.child:_><proxy.child:quick/></proxy.child:_></all>\n")
                tasks = await interpreter.wait_tasks(timeout=5.0, throw=False)
                pending = [t.caller_name() for t in tasks.values() if not t.done()]
                assert not pending, f"scope never closed, still pending: {pending}"
