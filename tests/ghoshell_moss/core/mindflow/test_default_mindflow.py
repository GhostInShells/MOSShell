import pytest
from ghoshell_moss.core.blueprint.mindflow import Signal, InputSignal
from ghoshell_moss.core.mindflow.priority_mindflow import new_default_mindflow, PriorityMindflow
from ghoshell_moss.core.mindflow.priority_attention import PriorityProtectionAttention
from ghoshell_moss.core.mindflow.input_signal_nucleus import InputSignalNucleus
from ghoshell_moss.message import Message
import asyncio


@pytest.mark.asyncio
async def test_new_default_mindflow_factory():
    """工厂函数返回 PriorityMindflow, 注册了 InputSignalNucleus."""
    mindflow = new_default_mindflow(protection_seconds=0.1)
    assert isinstance(mindflow, PriorityMindflow)
    # 判断使用了默认值.
    assert InputSignalNucleus().name() in mindflow.faculties()
    async with mindflow:
        assert mindflow.is_running()


@pytest.mark.asyncio
async def test_default_mindflow_creates_priority_attention():
    """default mindflow 使用 PriorityProtectionAttention."""
    mindflow = new_default_mindflow(protection_seconds=0.1)
    async with mindflow:
        mindflow.add_signal(Signal.new("input", Message.new().with_content("hello")))
        await asyncio.sleep(0.05)
        async for attention in mindflow.loop():
            assert isinstance(attention, PriorityProtectionAttention)
            break
