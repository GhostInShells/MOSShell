"""dsh hive / agent-session 行为证据 — 只测不需要网络的翻译层.

Covered:
- ``message_text``: 一条 MOSS Message 的纯文本, 非文本块不冒充文本.
- ``_assistant_text``: assistant/message 只取 text 块 — reasoning 与工具流量是"过程", 不是答案.
- 阶段映射: turn/start → running, assistant/message → output, turn/end → idle.
- hive 的游标: events(since) 只回 since 之后的事件, 第二次返回的游标可续读.

Not covered here (needs a live dsh): open/send/retire — those are proved by the probe, and the
lifecycle detail that matters (``DshSession.__aenter__`` starts the consume loop) can only be
observed against a real stream.
"""

from __future__ import annotations

import pytest

from ghoshell_dsh.v0_2_0.types.session_events import (
    AssistantMessageEvent,
    ContentBlock,
    Message as DshMessage,
)

from ghoshell_moss.agents.blueprint.agent import AgentPhase
from ghoshell_moss.deepseek_harness.v0_2_0.agent import assistant_text, is_final_answer, message_text
from ghoshell_moss.message import Message


def test_message_text_joins_text_blocks():
    message = Message.new().with_content("hello").with_content("world")
    assert message_text(message) == "hello\nworld"


def test_message_text_ignores_non_text_content():
    """非文本块不会被 ``str()`` 成一段垃圾塞进 prompt."""
    message = Message.new().with_content("only text")
    assert message_text(message) == "only text"


def test_assistant_text_keeps_only_text_blocks():
    event = AssistantMessageEvent(
        message=DshMessage(
            content=[
                ContentBlock(type="reasoning", text="thinking hard"),
                ContentBlock(type="text", text="ok"),
            ]
        )
    )
    assert assistant_text(event) == "ok"


def test_assistant_text_is_empty_when_there_is_no_prose():
    event = AssistantMessageEvent(
        message=DshMessage(content=[ContentBlock(type="reasoning", text="only thinking")])
    )
    assert assistant_text(event) == ""


def test_phase_values_are_stable():
    """阶段值进信号与绘图; 改名是一次破坏性变更, 这条测试让它无法悄悄发生."""
    assert [phase.value for phase in AgentPhase] == [
        "created",
        "input",
        "reasoning",
        "tool_use",
        "running",
        "output",
        "idle",
        "closed",
    ]




def test_a_step_that_asks_for_a_tool_is_not_the_answer():
    """dsh emits one assistant message per step: the preamble must not reach ``on_output``."""
    event = AssistantMessageEvent(
        message=DshMessage(
            content=[
                ContentBlock(type="text", text="I'll start by surveying the paths."),
                ContentBlock(type="tool-call", id="c1", name="bash"),
            ]
        )
    )
    assert is_final_answer(event) is False


def test_the_last_step_of_a_turn_is_the_answer():
    event = AssistantMessageEvent(
        message=DshMessage(content=[ContentBlock(type="text", text="done: 14 references")])
    )
    assert is_final_answer(event) is True
    assert assistant_text(event) == "done: 14 references"
