"""Mindflow channel — 真链路 CTML 测试.

与 ``test_mindflow_channel`` 的差别: 这里不走 channel 的 ``bootstrap`` 直连 runtime,
而是复用 ``MindflowInShellTestSuite`` 的真实三循环 + CTML interpreter, 验证 mindflow
channel 以 virtual child 挂进 shell 后, 模型能通过 CTML 直接操纵自身注意力.

契约:
    - ``MindflowInShell._thinking_loop`` 把 ``mindflow.as_channel()`` 以 virtual child
      挂到 ``shell.main_channel``, 因此 CTML 用 ``<mindflow:command/>`` 寻址.
    - set-* 命令是"确认"语义 (always_observe=False), 单帧即可收线; 状态变更直接落在
      真实 mindflow 实例上, 可断言.
"""

from __future__ import annotations

import asyncio

import pytest

from ghoshell_moss.core.blueprint.mindflow import Priority

from .mindflow_in_shell_test_suite import (
    MindflowInShellTestSuite,
    input_signal,
)


@pytest.mark.asyncio
async def test_ctml_set_impulse_bar_changes_mindflow():
    """CTML ``<mindflow:set-impulse-bar/>`` 经真实 interpreter 落到 mindflow 实例."""
    suite = MindflowInShellTestSuite()
    suite.articulate = suite.text_articulator('<mindflow:set-impulse-bar priority="CRITICAL"/>')

    async with suite:
        suite.add_signal(input_signal("set bar"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=1)

    assert suite.mindflow.impulse_priority_bar() == Priority.CRITICAL
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_ctml_set_signal_bar_changes_mindflow():
    """CTML ``<mindflow:set-signal-bar/>`` 经真实 interpreter 落到 mindflow 实例."""
    suite = MindflowInShellTestSuite()
    suite.articulate = suite.text_articulator('<mindflow:set-signal-bar priority="WARNING"/>')

    async with suite:
        suite.add_signal(input_signal("set bar"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=1)

    assert suite.mindflow.signal_priority_bar() == Priority.WARNING
    assert not suite.exceptions


@pytest.mark.asyncio
async def test_ctml_set_priority_operates_on_current_attention():
    """CTML ``<mindflow:set-priority/>`` 修改当前 attention 的优先级.

    输入 signal 起的 attention 就是 set-priority 操作的目标; 命令经 interpreter
    执行后, 该 attention 的优先级应被覆盖为 CRITICAL.
    """
    suite = MindflowInShellTestSuite()
    suite.articulate = suite.text_articulator('<mindflow:set-priority priority="CRITICAL"/>')

    async with suite:
        suite.add_signal(input_signal("user_msg"))
        await asyncio.wait_for(suite.attention_started.wait(), timeout=1)
        await asyncio.wait_for(suite.attention_stopped.wait(), timeout=1)

    assert suite.last_attention is not None
    assert suite.last_attention.priority() == Priority.CRITICAL
    assert not suite.exceptions
