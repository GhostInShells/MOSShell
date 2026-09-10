"""ListenerNucleus + Mindflow + Shell 集成 — 观察语音感知的注意力拓扑.

用 ``MindflowInShellTestSuite`` 把 ListenerNucleus 接进真实三循环, 喂 listener
signal, 观察两类拓扑:

- 默认 (clause_response off): 首包 incomplete 抢占 + 尾包 complete 响应 — 单 attention,
  ghost 只响应一次.
- 分句响应 (clause_response on): 每句独立 impulse id 互相打断 — 多 attention,
  ghost 逐句思考、被下一句打断, 产生连续思考帧.

这是 per-nucleus 单测 (``test_listener_nucleus``) 之外的集成接线观测; 两者独立.
"""
import asyncio
from typing import AsyncIterable

import pytest

from ghoshell_moss.core.blueprint.mindflow import Thinking
from ghoshell_moss.core.mindflow import (
    BaseMindflow,
    CommandNucleus,
    InputSignalNucleus,
    InterruptNucleus,
    ListenerNucleus,
    NotifyNucleus,
)
from ghoshell_moss.core.mindflow.listener_nucleus import ListenerPacket, new_listener_signal

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
async def test_turn_first_packet_preempt_then_tail_response():
    """默认拓扑: 首包 incomplete 抢占不响应, 尾包 complete 响应一次全量.

    契约: 一个 turn 只产生一个 attention (尾包 same-id 吸收首包), ghost 只 articulate 一次.
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
        suite.add_signal(new_listener_signal(ListenerPacket.FIRST, turn_id='t1'))
        suite.add_signal(new_listener_signal(ListenerPacket.CLAUSE, '今天天气不错', turn_id='t1', clause_index=1))
        suite.add_signal(new_listener_signal(ListenerPacket.TAIL, '', turn_id='t1'))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=2)

    # 单 attention: 首包 incomplete 不响应, 尾包 complete 响应一次.
    assert suite.attention_count == 1
    assert len(articulated) == 1
    assert articulated[0] == ['今天天气不错']  # 分句未送达, 尾包发全量
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_clause_response_produces_continuous_thinking_frames():
    """分句响应拓扑: 每句独立 impulse id 互相打断 → 连续思考帧.

    契约: 每个分句各自抢占上一个 attention, ghost 逐句 articulate (累计 buffer),
    产生多个 attention (连续思考帧), 而不是默认的单 attention 一次响应.
    """
    suite = MindflowInShellTestSuite(mindflow=build_listener_mindflow(clause_response=True))
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
        suite.add_signal(new_listener_signal(ListenerPacket.FIRST, turn_id='t1'))
        # 每句之间留一点间隙, 让每个 attention 有机会起帧, 再被下一句打断.
        for i, text in enumerate(('句1', '句2'), start=1):
            suite.add_signal(new_listener_signal(ListenerPacket.CLAUSE, text, turn_id='t1', clause_index=i))
            await asyncio.sleep(0.05)
        suite.add_signal(new_listener_signal(ListenerPacket.TAIL, '', turn_id='t1'))
        # 等所有 attention 走完 (articulate 至少 2 次 + 系统回到 idle).
        for _ in range(300):
            if len(articulated) >= 2 and suite.attention_stopped.is_set():
                break
            await asyncio.sleep(0.02)

    # 连续思考帧: 不止一个 attention (首包 + 每句), 每句各 articulate 一次.
    assert suite.attention_count >= 2
    # 至少两个分句各触发一次 articulate, 且内容累计增长.
    assert len(articulated) >= 2
    assert articulated[0] == ['句1']
    assert '句1' in articulated[1][0] and '句2' in articulated[1][0]
    assert not suite.exceptions
