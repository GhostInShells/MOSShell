"""一条"开了却没收尾"的 action, 能不能冻住整个 action loop?

线上背景 (2026-10-13, moss-in-reachy-mini):
一次 ``moss_interpret`` 的参数解析失败, 它的 ``tool-call-delta`` 已经在 MOSS 侧开出了
articulator, 但没有任何人去关它 —— 于是这条 action 永不 commit, 全厨房唯一一口灶被它占死,
紧随其后的第二条 ``moss_interpret`` 永久阻塞, 整帧冻结 (只能重启 host 才解).

这个文件不用替身, 只跑生产的 ``MindflowInShell`` 三循环 (经 ``MindflowInShellTestSuite``
装线), 把那条 kernel 规则单独立住:

    **一条被抛弃的 action, 不得让 action loop 停摆.**

- 复现 (现在就是红的): ``articulate`` 里开一个 articulator、喂点 logos、**永不 commit**;
  然后照常开第二个并等它跑完。现在第二个等不到 —— 因为 action loop 停在第一个身上。
- 这个文件测的是 **kernel 的鲁棒性**, 和"Dolores 的参数解析失败分支该不该收尾"是两个决定:
  即使 Dolores 侧补上了收尾, 这里仍然是红的; 反过来如果给 kernel 的等待加了界, 这里会自动变绿。
  两边都该有测试, 互不替代。
"""
import asyncio

import pytest

from ghoshell_moss.core.blueprint.mindflow import Thinking

from .mindflow_in_shell_test_suite import MindflowInShellTestSuite, input_signal


@pytest.mark.asyncio
async def test_abandoned_articulator_does_not_freeze_the_action_loop():
    """第一个 articulator 永不收尾时, 第二个必须还能跑完."""
    suite = MindflowInShellTestSuite()

    seen: list[str] = []

    async def content_func(chunks__):
        async for chunk in chunks__:
            seen.append(chunk)

    suite.shell.main_channel.build.content_command(content_func)

    second_done = asyncio.Event()
    text_articulator = suite.text_articulator("second frame")

    async def articulate(thinking: Thinking) -> None:
        # ① 被抛弃的那一条: 边界打开、logos 送进去了, 但永不 commit / 永不 __aexit__。
        #    这就是线上"delta 开了流而收尾的人没来"的 kernel 侧形状。
        abandoned = thinking.articulator()
        await abandoned.__aenter__()
        abandoned.send_nowait("abandoned frame ")

        # ② 紧接着的第二帧: 它和线上那条重发的 moss_interpret 一样, 必须能做出来。
        await text_articulator(thinking)
        second_done.set()

    suite.articulate = articulate

    async with suite:
        suite.add_signal(input_signal("hello"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        try:
            await asyncio.wait_for(second_done.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pytest.fail(
                "第二帧卡死了: 上一条 action 没有 commit, 而 action loop 一次只服务一个 action "
                "(mindflow_in_shell.py:342 await action.wait_compiled()), 所以它永远轮不到 —— "
                "这就是线上整帧冻结在 kernel 侧的根因"
            )
        assert not suite.exceptions, suite.exceptions
    assert "second frame" in "".join(seen)


@pytest.mark.asyncio
async def test_voiding_an_articulator_releases_the_action_loop():
    """同一条戏, 但第一帧**被作废** —— 第二帧必须能跑完.

    这是 kernel 给出的那条"只放弃这一个 action、不 abort 整帧"的路: 以**非 StatementExitedException**
    的异常退出 ``BaseArticulator.__aexit__``, 它才会落 action 的 stop event。若传
    ActionExitedException / StatementExitedException, 走的是"attention 已经 abort"那一支, 什么都不做
    —— 那样等于没有放行。

    Dolores 侧 ``_void_stream`` (参数被拒时作废 CTML 流) 骑的就是这条路, 这里在生产装线上把它钉住。
    """
    suite = MindflowInShellTestSuite()

    seen: list[str] = []

    async def content_func(chunks__):
        async for chunk in chunks__:
            seen.append(chunk)

    suite.shell.main_channel.build.content_command(content_func)

    second_done = asyncio.Event()
    text_articulator = suite.text_articulator("second frame")

    async def articulate(thinking: Thinking) -> None:
        # ① 开过边界、送过 logos, 然后作废: 不 commit, 只让这条 action 出栈。
        voided = thinking.articulator()
        await voided.__aenter__()
        voided.send_nowait("voided frame ")
        void = RuntimeError("voided: arguments rejected")
        await voided.__aexit__(type(void), void, None)

        # ② 第二帧照常跑完。
        await text_articulator(thinking)
        second_done.set()

    suite.articulate = articulate

    async with suite:
        suite.add_signal(input_signal("hello"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        try:
            await asyncio.wait_for(second_done.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pytest.fail(
                "作废之后 action loop 仍然没放行 —— 异常退出这条路没有落 action 的 stop event"
            )
        assert not suite.exceptions, suite.exceptions
    assert "second frame" in "".join(seen)
    assert "voided frame" not in "".join(seen), "被作废的 CTML 不得执行"
