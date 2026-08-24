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
    """status() 返回红点摘要: pending 计数 + 最新 pending 消息预览, 无消息回退 description."""
    async with InputSignalNucleus() as nuc:
        nuc.add_signal(Signal.new("input", Message.new().with_content("hello there")))
        await asyncio.sleep(0.01)
        status = nuc.status()
        assert "pending: 1" in status
        assert "hello there" in status

        # 无消息内容时回退到 description 字段.
        nuc.clear()
        nuc.add_signal(Signal.new("input", description="user says hi"))
        await asyncio.sleep(0.01)
        status = nuc.status()
        assert "pending: 1" in status
        assert "user says hi" in status


@pytest.mark.asyncio
async def test_status_empty_when_no_pending():
    """无 pending 信号时 status() 为空 — nucleus 从 perspective 排除."""
    async with InputSignalNucleus() as nuc:
        assert nuc.status() == ""


@pytest.mark.asyncio
async def test_description_is_static_label():
    """description() 是稳定标签 — 不随 pending 变化, 计数不塞进 description."""
    async with InputSignalNucleus() as nuc:
        desc_before = nuc.description()
        nuc.add_signal(Signal.new("input", Message.new().with_content("hi")))
        await asyncio.sleep(0.01)
        assert nuc.description() == desc_before
        assert "pending" not in nuc.description()


@pytest.mark.asyncio
async def test_pending_count_reflects_buffer_lifecycle():
    """pending_count() 随入队增加, attended 消费后归零."""
    async with InputSignalNucleus() as nuc:
        assert nuc.pending_count() == 0
        nuc.add_signal(Signal.new("input", Message.new().with_content("a")))
        nuc.add_signal(Signal.new("input", Message.new().with_content("b")))
        await asyncio.sleep(0.01)
        assert nuc.pending_count() == 2
        imp = nuc.peek()
        assert imp is not None
        nuc.attended(imp)
        await asyncio.sleep(0.01)
        assert nuc.pending_count() == 0


@pytest.mark.asyncio
async def test_pending_count_excludes_stale():
    """pending_count() 排除已 stale 的信号 — 不虚报 pending."""
    async with InputSignalNucleus() as nuc:
        nuc.add_signal(Signal.new("input", Message.new().with_content("fresh")))
        nuc.add_signal(Signal.new("input", Message.new().with_content("stale"), stale_timeout=0.001))
        await asyncio.sleep(0.02)
        assert nuc.pending_count() == 1
        assert "pending: 1" in nuc.status()


@pytest.mark.asyncio
async def test_attended_clears_all():
    """attended 后 buffer 清空."""
    async with InputSignalNucleus() as nuc:
        for i in range(3):
            nuc.add_signal(Signal.new("input", Message.new().with_content(f"msg{i}")))
        await asyncio.sleep(0.01)
        imp = nuc.peek()
        assert imp is not None
        nuc.attended(imp)
        await asyncio.sleep(0.01)
        assert nuc.peek() is None
        assert nuc.status() == ""


@pytest.mark.asyncio
async def test_full_messages_in_impulse():
    """peek 到的 Impulse 包含全部入队消息 (FIFO)."""
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
            fire_impulse=lambda imp: notified.append(imp),
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
    """suppress 期满后新信号到达时, 累积的旧消息与新消息合并重建 impulse."""
    async with InputSignalNucleus(suppress_seconds=0.1) as nuc:
        nuc.add_signal(Signal.new("input", Message.new().with_content("first")))
        await asyncio.sleep(0.01)
        assert nuc.peek() is not None

        nuc.suppress(Impulse(source="other"))
        # 压制期内 — cache 已清, peek 不可见
        assert nuc.peek() is None, "suppress 期内 peek 应返回 None"

        await asyncio.sleep(0.15)
        # 压制期满, 但没有新信号 → cache 为空 (suppress 清了 cache)
        assert nuc.peek() is None

        # 新信号到达 → 从累积的 signals (first + second) 重建
        nuc.add_signal(Signal.new("input", Message.new().with_content("second")))
        await asyncio.sleep(0.01)
        imp = nuc.peek()
        assert imp is not None
        texts = [m.to_content_string() for m in imp.messages]
        assert texts == ["first", "second"], (
            f"suppress 保留 signals, 新信号应合并旧消息, 实际: {texts}"
        )


# -- impulse 生命周期观测 (public-internal) --


@pytest.mark.asyncio
async def test_attended_counts_and_brief():
    """attended 回调后: attended_count +1, 简介为 impulse 消息, pending 清空."""
    async with InputSignalNucleus() as nuc:
        assert nuc.attended_count() == 0
        nuc.add_signal(Signal.new("input", Message.new().with_content("hello")))
        await asyncio.sleep(0.01)
        imp = nuc.peek()
        assert imp is not None
        nuc.attended(imp)
        assert nuc.attended_count() == 1
        assert nuc.counters()["last_attended"] == "hello"
        assert nuc.pending_count() == 0


@pytest.mark.asyncio
async def test_ignored_counts_and_brief():
    """ignored 回调后: ignored_count +1, 简介为 impulse 消息."""
    async with InputSignalNucleus() as nuc:
        assert nuc.ignored_count() == 0
        nuc.add_signal(Signal.new("input", Message.new().with_content("bye")))
        await asyncio.sleep(0.01)
        imp = nuc.peek()
        assert imp is not None
        nuc.ignored(imp)
        assert nuc.ignored_count() == 1
        assert nuc.counters()["last_ignored"] == "bye"


@pytest.mark.asyncio
async def test_suppressed_counts_and_brief():
    """suppress 回调后: suppressed_count +1, 简介取被压制的 impulse (而非压制方)."""
    async with InputSignalNucleus() as nuc:
        assert nuc.suppressed_count() == 0
        nuc.add_signal(Signal.new("input", Message.new().with_content("lost")))
        await asyncio.sleep(0.01)
        nuc.suppress(Impulse(source="other_nucleus"))
        assert nuc.suppressed_count() == 1
        assert nuc.counters()["last_suppressed"] == "lost"


@pytest.mark.asyncio
async def test_counters_aggregate_all_actions():
    """三类回调各触发一次后, counters() 汇总与分量方法一致."""
    async with InputSignalNucleus() as nuc:
        nuc.add_signal(Signal.new("input", Message.new().with_content("a")))
        await asyncio.sleep(0.01)
        nuc.attended(nuc.peek())

        nuc.add_signal(Signal.new("input", Message.new().with_content("b")))
        await asyncio.sleep(0.01)
        nuc.ignored(nuc.peek())

        nuc.add_signal(Signal.new("input", Message.new().with_content("c")))
        await asyncio.sleep(0.01)
        nuc.suppress(Impulse(source="other"))

        c = nuc.counters()
        assert c["attended"] == nuc.attended_count() == 1
        assert c["ignored"] == nuc.ignored_count() == 1
        assert c["suppressed"] == nuc.suppressed_count() == 1
