import pytest
from ghoshell_moss.core.blueprint.mindflow import Signal, Impulse, Priority
from ghoshell_moss.core.mindflow.input_signal_nucleus import InputSignalNucleus
from ghoshell_moss.message import Message
import asyncio


@pytest.mark.asyncio
async def test_basic_enqueue_and_peek():
    """信号入队, peek 可见."""
    async with InputSignalNucleus() as nuc:
        nuc.add_signal(Signal.new("input", Message.new().with_content("hello")))
        await asyncio.sleep(0.01)
        imp = nuc.peek()
        assert imp is not None
        assert len(imp.messages) == 1


@pytest.mark.asyncio
async def test_status_red_dot():
    """status() 返回红点格式."""
    async with InputSignalNucleus() as nuc:
        nuc.add_signal(Signal.new("input", description="user says hi"))
        await asyncio.sleep(0.01)
        status = nuc.status()
        assert "pending: 1" in status
        assert "user says hi" in status


@pytest.mark.asyncio
async def test_pop_clears_all():
    """pop 后 buffer 清空."""
    async with InputSignalNucleus() as nuc:
        for i in range(3):
            nuc.add_signal(Signal.new("input", Message.new().with_content(f"msg{i}")))
        await asyncio.sleep(0.01)
        imp = nuc.peek()
        assert imp is not None
        nuc.pop_impulse(imp)
        await asyncio.sleep(0.01)
        assert nuc.peek() is None
        assert nuc.status() == ""


@pytest.mark.asyncio
async def test_full_messages_in_impulse():
    """pop 时的 Impulse 包含全部入队消息 (FIFO)."""
    async with InputSignalNucleus() as nuc:
        nuc.add_signal(Signal.new("input", Message.new().with_content("a")))
        nuc.add_signal(Signal.new("input", Message.new().with_content("b")))
        nuc.add_signal(Signal.new("input", Message.new().with_content("c")))
        await asyncio.sleep(0.01)
        imp = nuc.peek()
        texts = [m.to_content_string() for m in imp.messages]
        assert texts == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_priority_is_max():
    """Impulse.priority = max of buffered signals."""
    async with InputSignalNucleus() as nuc:
        nuc.add_signal(Signal.new("input", priority=Priority.NOTICE))
        nuc.add_signal(Signal.new("input", priority=Priority.WARNING))
        nuc.add_signal(Signal.new("input", priority=Priority.INFO))
        await asyncio.sleep(0.01)
        imp = nuc.peek()
        assert imp.priority == Priority.WARNING


@pytest.mark.asyncio
async def test_buffer_limit():
    """超过 buffer_size 时淘汰最早的."""
    async with InputSignalNucleus(buffer_size=3) as nuc:
        for i in range(5):
            nuc.add_signal(Signal.new("input", Message.new().with_content(f"msg{i}")))
        await asyncio.sleep(0.01)
        imp = nuc.peek()
        texts = [m.to_content_string() for m in imp.messages]
        assert len(texts) == 3
        assert texts == ["msg2", "msg3", "msg4"]


@pytest.mark.asyncio
async def test_suppress_cooldown():
    """被压制后 suppress_seconds 内不通知."""
    notified = []
    async with InputSignalNucleus(suppress_seconds=0.2) as nuc:
        nuc.with_bus(
            signal_broadcast=lambda s: None,
            impulse_notify=lambda imp: notified.append(imp),
        )
        nuc.add_signal(Signal.new("input", Message.new().with_content("first")))
        await asyncio.sleep(0.01)
        assert len(notified) == 1

        nuc.suppress(Impulse(source="test"))
        nuc.add_signal(Signal.new("input", Message.new().with_content("second")))
        await asyncio.sleep(0.01)
        # 被压制, 不会通知
        assert len(notified) == 1

        # 冷静期过后, 新信号可以通知
        await asyncio.sleep(0.2)
        nuc.add_signal(Signal.new("input", Message.new().with_content("third")))
        await asyncio.sleep(0.01)
        assert len(notified) == 2


@pytest.mark.asyncio
async def test_ignores_wrong_signal_name():
    """忽略不匹配的 signal name."""
    async with InputSignalNucleus() as nuc:
        nuc.add_signal(Signal.new("vision", Message.new().with_content("img")))
        await asyncio.sleep(0.01)
        assert nuc.peek() is None


@pytest.mark.asyncio
async def test_filters_low_priority():
    """过滤低于 min_priority 的信号."""
    async with InputSignalNucleus(min_priority=Priority.NOTICE) as nuc:
        nuc.add_signal(Signal.new("input", priority=Priority.INFO))
        await asyncio.sleep(0.01)
        assert nuc.peek() is None
