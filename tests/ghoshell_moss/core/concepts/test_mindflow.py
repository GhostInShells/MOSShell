import pytest
import time
from datetime import datetime, timezone
from ghoshell_moss.message import Message
from ghoshell_moss.core.concepts.mindflow import (
    Signal, Impulse, Observation, Outcome, Priority
)


# 1. 测试 Signal 到 Impulse 的转换逻辑
def test_signal_to_impulse_conversion():
    # 创建一个原始信号
    msg = Message.new().with_content("Hello MOSS")
    signal = Signal.new(
        "test_signal",
        msg,
        priority=Priority.WARNING,
        description="test",
        stale_timeout=2.0
    )

    # 执行转换
    impulse = Impulse.from_signal(signal, source="test_nucleus")

    # 验证数据对齐
    assert impulse.source == "test_nucleus"
    assert impulse.priority == Priority.WARNING
    assert impulse.messages[0].contents[0]['text'] == "Hello MOSS"
    assert impulse.stale_timeout > 0
    # 验证 trace_id 继承
    assert impulse.trace_id == signal.id


# 2. 测试 Observation 与 Outcome 的缝合 (核心认知流)
def test_observation_outcome_stitching():
    # 模拟第一轮 Observation
    obs = Observation()
    obs.inputs = [Message.new().with_content("Input 1")]

    # 生成 Outcome
    outcome = obs.new_outcome()
    outcome.logos = "MoveForward"
    outcome.messages = [Message.new().with_content("Action Done")]

    # 缝合到下一轮 Observation
    obs2 = outcome.new_observation()

    # 验证上下文连贯性
    assert obs2.last is not None
    assert obs2.last.logos == "MoveForward"
    assert obs2.last.messages[0].contents[0]['text'] == "Action Done"

    # 验证 as_request_messages 结构
    msgs = list(obs2.as_request_messages())
    # 应该包含 <outcomes> 标签及内部消息
    content_tags = [m.meta.tag for m in msgs if m.meta.tag]
    assert 'stop_reason' not in content_tags  # 此时 stop_reason 应为空


# 3. 测试 Impulse 的保鲜逻辑 (Stale Timeout)
def test_impulse_stale_logic():
    signal = Signal.new("test", stale_timeout=0.1)
    impulse = Impulse.from_signal(signal, source="test")

    assert impulse.is_stale() is False
    time.sleep(0.2)
    assert impulse.is_stale() is True


# 4. 测试优先级抢占判定逻辑 (on_challenge 核心模拟)
def test_attention_preemption_logic():
    # 模拟一个正在运行的 Attention 的 Impulse
    current_impulse = Impulse(source="nucleus_a", priority=Priority.INFO, strength=100)

    # 模拟一个高优先级的挑战
    challenge = Impulse(source="nucleus_b", priority=Priority.CRITICAL, strength=100)

    # 模拟 Attention 内部的仲裁 (simplified)
    # 规则：CRITICAL > INFO -> 必须被抢占
    assert challenge.priority > current_impulse.priority

    # 模拟同优先级，强弱对抗
    weak_challenge = Impulse(source="nucleus_b", priority=Priority.INFO, strength=50)
    assert weak_challenge.strength < current_impulse.strength
