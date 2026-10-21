import pytest

from ghoshell_moss.core import PyChannel, new_ctml_shell


@pytest.mark.asyncio
async def test_all_scope_with_child_channel_scope_finishes():
    """<all> 里套一个子 channel 的 scope 时, 整个 <all> 必须能收口.

    D21 17:13 的真实现场: `<all><webview_screen:_>...</_><_><say/>...</_></all>`
    之后 __main__ 被永久占据, 后面所有命令(含 clear_speech)全部排队不执行 —
    表现就是"声音突然没了, 重启才好".
    """
    shell = new_ctml_shell("scope-all-child")
    child = PyChannel(name="child")

    @child.build.command()
    async def quick() -> int:
        return 1

    shell.main_channel.import_channels(child)

    async with shell:
        async with shell.interpreter_in_ctx(clear_after_exit=True) as interpreter:
            interpreter.feed("<all><child:_><child:quick/></child:_><_><child:quick/><child:quick/></_></all>\n")
            tasks = await interpreter.wait_tasks(timeout=5.0, throw=False)
            pending = [t.caller_name() for t in tasks.values() if not t.done()]
            assert not pending, f"scope never closed, still pending: {pending}"
