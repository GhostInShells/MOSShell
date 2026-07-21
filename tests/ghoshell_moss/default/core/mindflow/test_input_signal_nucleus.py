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


@pytest.mark.asyncio
async def test_suppress_clears_impulse_from_peek():
    """suppress 后 peek 应返回 None — 被压制的 impulse 不再参与 mindflow ranking.

    复现场景:
      1. signal 入队 → impulse 缓存 → mindflow peek 可见
      2. challenge 失败, default 路径调 suppress() → 只设 _suppress_until
      3. peek() 不检查 suppress 状态 → 返回同一 impulse
      4. mindflow 每 0.5s timeout 重新 rank → peek → 同一 impulse → 反复 suppress → 重放

    当前行为: suppress 后 peek 仍返回 impulse (BUG).
    预期行为: suppress 后 peek 返回 None.
    """
    async with InputSignalNucleus(suppress_seconds=0.5) as nuc:
        nuc.add_signal(Signal.new(
            "input",
            Message.new().with_content("hello"),
            stale_timeout=0,  # 永不 stale, 模拟默认信号
        ))
        await asyncio.sleep(0.01)
        imp_before = nuc.peek()
        assert imp_before is not None, "信号入队后 peek 应该有 impulse"

        # 模拟 mindflow challenge 失败 → suppress
        nuc.suppress(Impulse(source="other_nucleus"))

        # 关键断言: suppress 后 peek 不应再看到该 impulse
        imp_after = nuc.peek()
        assert imp_after is None, (
            f"suppress 后 peek 应返回 None, 但实际返回了 {imp_after.id} "
            f"(source={imp_after.source}). "
            f"这会导致 mindflow 在 _on_impulse_consuming_loop 的 0.5s timeout 循环中 "
            f"反复 rank 到同一 impulse, 造成消息重放."
        )


@pytest.mark.asyncio
async def test_suppress_expired_then_new_signal_revives():
    """suppress 期满后新信号到达才重新产出 impulse."""
    async with InputSignalNucleus(suppress_seconds=0.1) as nuc:
        nuc.add_signal(Signal.new("input", Message.new().with_content("first")))
        await asyncio.sleep(0.01)
        assert nuc.peek() is not None

        nuc.suppress(Impulse(source="other"))
        # 压制期内
        assert nuc.peek() is None, "suppress 期内 peek 应返回 None"

        await asyncio.sleep(0.15)
        # 压制期满, 但没有新信号 → 缓存仍为空 (被 suppress 清理了)
        assert nuc.peek() is None

        # 新信号到达 → 重新构建
        nuc.add_signal(Signal.new("input", Message.new().with_content("second")))
        await asyncio.sleep(0.01)
        imp = nuc.peek()
        assert imp is not None
        texts = [m.to_content_string() for m in imp.messages]
        assert texts == ["second"]
