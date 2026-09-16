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
