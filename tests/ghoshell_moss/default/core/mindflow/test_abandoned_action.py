"""一条 action 被弃掉时, 该由谁、以什么方式放它走?

线上背景 (2026-10-13, moss-in-reachy-mini):
一次 ``moss_interpret`` 的参数解析失败, 它的 ``tool-call-delta`` 已经在 MOSS 侧开出了
articulator, 但没有任何人去关它 —— 于是这条 action 永不 commit, 厨房唯一一口灶被它占死,
紧随其后的第二条 ``moss_interpret`` 永久阻塞, 整帧冻结 (只能靠人打断才解).

**修它在"弃它的人"那一侧, 不在 kernel**: Dolores 的 ``_void_stream`` 在参数被拒的那一刻
作废这条流, 骑的就是本文件钉住的那条 kernel 路 ——

    **弃掉一条 action, 是以"异常退出"收它的边界; 不是提交, 也不是等一个钟.**

本文件只留这一条契约. 同一份 PR 里还有过一条
``test_abandoned_articulator_does_not_freeze_the_action_loop`` (要求"没人收尾的 action 也不许
让 action loop 停摆"), 已移除 —— 那是在要求 kernel 去猜"没人会来了", 而 kernel 猜不到:
唯一能"猜"的办法是给那次等待设一个上限, 而它等的是模型自己生成 arguments 的那一段时间
(第一个 delta → arguments 到位), 天然没有上界. 给它加钟, 等于给整帧的生命周期加钟;
按设计, 能结束一帧的只有 attention 与 abort. 所以这条要求归**开边界的人**, 不归 kernel.

不用替身, 只跑生产的 ``MindflowInShell`` 三循环 (经 ``MindflowInShellTestSuite`` 装线).
"""
import asyncio

import pytest

from ghoshell_moss.core.blueprint.mindflow import Thinking

from .mindflow_in_shell_test_suite import MindflowInShellTestSuite, input_signal


@pytest.mark.asyncio
async def test_voiding_an_articulator_releases_the_action_loop():
    """同一条戏, 但第一帧**被作废** —— 第二帧必须能跑完.

    这是 kernel 给出的那条"只放弃这一个 action、不 abort 整帧"的路: ``BaseArticulator.__aexit__``
    收到异常退出时落 ``BaseAction.close_lifecycle()`` —— stop / compiled / observed 三个端点一起
    落地, 于是 action loop 那次 ``await action.wait_compiled()`` 被放行; 走正常退出(commit) 反而会
    去等一个不会再有人生产的编译位, 把整条消费链拖住。

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
                "作废之后 action loop 仍然没放行 —— 异常退出这条路没有落地 action 的生命周期端点"
            )
        assert not suite.exceptions, suite.exceptions
    assert "second frame" in "".join(seen)
    assert "voided frame" not in "".join(seen), "被作废的 CTML 不得执行"
