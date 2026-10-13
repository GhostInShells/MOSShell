"""ListenerNucleus + Mindflow + Shell 集成 — 观察语音感知的注意力拓扑.

用 ``MindflowInShellTestSuite`` 把 ListenerNucleus 接进真实三循环, 喂 listener
signal, 观察打断包 → 发送包的注意力拓扑:

- 打断包 (complete=False) incomplete 抢占占坑 → 发送包 (complete=True) same-id
  absorb 填充 → 单 attention, ghost 只响应一次.

这是 per-nucleus 单测 (``test_listener_nucleus``) 之外的集成接线观测; 两者独立.
"""
import asyncio
from typing import AsyncIterable

import pytest

from ghoshell_moss.core.blueprint.mindflow import Priority, Thinking
from ghoshell_moss.core.mindflow import (
    BaseMindflow,
    CommandNucleus,
    InputSignalNucleus,
    InterruptNucleus,
    ListenerNucleus,
    NotifyNucleus,
)
from ghoshell_moss.core.mindflow.listener_nucleus import new_listener_signal
from ghoshell_moss.core.mindflow.notify_nucleus import new_notify_signal

from .mindflow_in_shell_test_suite import MindflowInShellTestSuite


def build_listener_mindflow(**kwargs) -> BaseMindflow:
    """最小 mindflow 集合 + ListenerNucleus."""
    return BaseMindflow(
        ListenerNucleus(**kwargs),
        InputSignalNucleus(),
        InterruptNucleus(suppress_seconds=0.05),
        CommandNucleus(),
        NotifyNucleus(),
    )


def _percept_texts(thinking: Thinking) -> list[str]:
    """抽取当前 thinking 帧的 percepts 文本 (ghost 感知到的输入)."""
    texts = []
    for msg in thinking.moment.percepts_messages():
        for c in msg.contents:
            if isinstance(c, dict) and c.get('type') == 'text':
                texts.append(c.get('text', ''))
    return texts


async def _noop_content(chunks__: AsyncIterable[str]) -> None:
    async for _ in chunks__:
        pass


@pytest.mark.asyncio
async def test_turn_interrupt_packet_preempt_then_deliver_response():
    """打断包 incomplete 抢占占坑, 发送包 complete 响应一次全量.

    契约: 一个 turn 只产生一个 attention (发送包 same-id 吸收打断包), ghost 只 articulate 一次.
    """
    suite = MindflowInShellTestSuite(mindflow=build_listener_mindflow())
    suite.shell.main_channel.build.content_command(_noop_content)

    articulated: list[list[str]] = []

    async def articulate(thinking: Thinking) -> None:
        articulated.append(_percept_texts(thinking))
        art = thinking.articulator()
        async with art:
            art.send_nowait("ok")
            if not thinking.is_aborted():
                await art.wait_action_done()

    suite.articulate = articulate

    async with suite:
        suite.add_signal(new_listener_signal(
            segment_id='t1', interrupt=True, complete=False, priority=Priority.WARNING,
        ))
        suite.add_signal(new_listener_signal('今天天气不错', segment_id='t1', complete=True))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=2)

    # 单 attention: 打断包 incomplete 不响应, 发送包 complete 响应一次.
    assert suite.attention_count == 1
    assert len(articulated) == 1
    assert articulated[0] == ['今天天气不错']
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_deliver_must_not_be_lost_to_impulse_ranking():
    """一条 complete 发送包与更高优先级 impulse 同时到达时, 它不能输在"排名"上被丢掉.

    契约 (ListenerNucleus): 发送包 (complete=True) 携带内容 —— **内容绝不丢**;
    且 onset (complete=False) 占住的 attention 必须能完成 (三循环不能被钉死).

    机制: ``Mindflow._rank_best_impulse_from_nuclei`` 对同一轮 rank 中落选的 impulse 调
    ``nucleus.suppress()``。发送包是**来补完当前这一帧**的 (onset 占坑 → deliver 填满),
    不是来抢注意力的挑战者 —— 它一旦输在 rank 上, 它占着的那个 attention 就永远等不到
    complete, ``_generate_thinking`` 钉在 ``attention.wait_ready()`` 上, thinking 通路停摆
    (信号通路独立, 于是表现为"set attention 照打但系统无响应")。

    修法: 被等的尾包走高优通道直接胜出 (``_awaited_impulse_id`` / ``_AWAITED_IMPULSE_LANE``),
    且 ``ListenerNucleus.suppress`` 按契约保留 impulse、只进冷却。

    判据 (外部可观测): 发送包的文本必须最终进入 ghost 的感知 (被 articulate).
    """
    suite = MindflowInShellTestSuite(mindflow=build_listener_mindflow())
    suite.shell.main_channel.build.content_command(_noop_content)

    articulated: list[list[str]] = []

    async def articulate(thinking: Thinking) -> None:
        articulated.append(_percept_texts(thinking))
        art = thinking.articulator()
        async with art:
            art.send_nowait("ok")
            if not thinking.is_aborted():
                await art.wait_action_done()

    suite.articulate = articulate

    async with suite:
        # 1) onset 抢占注意力 (complete=False, 只占坑, 不响应).
        suite.add_signal(new_listener_signal(
            segment_id='t1', interrupt=True, complete=False, priority=Priority.WARNING,
        ))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)

        # 2) 同一轮 rank 里一起到达: 发送包 (INFO) + 一个更高优先级的 notify (NOTICE).
        #    两次 add_signal 之间不 await —— 它们必须出现在同一次 rank 里.
        suite.add_signal(new_listener_signal(
            '今天天气不错', segment_id='t1', complete=True, priority=Priority.INFO,
        ))
        suite.add_signal(new_notify_signal('旁路通知'))

        await asyncio.sleep(1.0)

    # 内容绝不丢: 发送包文本必须最终落到某个被 articulate 的帧里.
    assert any('今天天气不错' in texts for texts in articulated), (
        f"发送包内容丢失或被钉死 — articulated={articulated}"
    )


@pytest.mark.asyncio
async def test_deliver_survives_when_it_ranks_in_its_own_round():
    """对照组: 发送包不与更高优先级 impulse 撞进同一轮 rank 时, 内容正常送达.

    与上一条只差一处 —— 两条 signal 之间让出一轮, 于是各排各的 rank. 用来把"同一轮 rank
    里落选"钉成触发条件: 只要不在同一轮, 就没有东西在跟发送包抢, 内容不丢.
    """
    suite = MindflowInShellTestSuite(mindflow=build_listener_mindflow())
    suite.shell.main_channel.build.content_command(_noop_content)

    articulated: list[list[str]] = []

    async def articulate(thinking: Thinking) -> None:
        articulated.append(_percept_texts(thinking))
        art = thinking.articulator()
        async with art:
            art.send_nowait("ok")
            if not thinking.is_aborted():
                await art.wait_action_done()

    suite.articulate = articulate

    async with suite:
        suite.add_signal(new_listener_signal(
            segment_id='t2', interrupt=True, complete=False, priority=Priority.WARNING,
        ))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)

        # 先让发送包走完它自己那一轮 rank...
        suite.add_signal(new_listener_signal(
            '今天天气不错', segment_id='t2', complete=True, priority=Priority.INFO,
        ))
        await asyncio.sleep(0.2)
        # ...再让高优先级的那条进来.
        suite.add_signal(new_notify_signal('旁路通知'))

        await asyncio.sleep(0.8)

    assert any('今天天气不错' in texts for texts in articulated), (
        f"对照组本该送达 — articulated={articulated}"
    )
