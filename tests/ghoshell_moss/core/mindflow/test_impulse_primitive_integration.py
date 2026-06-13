"""
Impulse + ImpulsePrimitive 集成测试.

绕过 nucleus 信号链, 用新的公开 ``Mindflow.add_impulse`` 直接注入 impulse,
验证 ImpulsePrimitive 各组合在协议层的确定性行为.

只测协议级命题:
- add_impulse 直接路径走标准 rank/challenge 流程
- last-impulse cache 语义 (后注入覆盖前)
- 各 primitive 组合的仲裁结果与 attention 状态变化
"""
import asyncio
import pytest

from ghoshell_moss.contracts.logger import get_console_logger
from ghoshell_moss.core.blueprint.mindflow import (
    Impulse, Priority, ImpulsePrimitive, ChallengeMode,
)
from ghoshell_moss.core.mindflow.base_mindflow import BaseMindflow, _DirectImpulseNucleus
from ghoshell_moss.message import Message


def _new_mindflow() -> BaseMindflow:
    return BaseMindflow(logger=get_console_logger())


def _imp(
        *,
        priority: int = Priority.NOTICE,
        messages: list[Message] | None = None,
) -> Impulse:
    return Impulse(
        priority=priority,
        messages=messages or [Message.new().with_content('m')],
    )


async def _first_attention(mindflow: BaseMindflow, timeout: float = 2.0):
    """阻塞拿到 mindflow.loop() yield 的第一个 attention. 超时则 fail."""
    async for attention in mindflow.loop():
        return attention
    raise AssertionError("mindflow.loop() exited without yielding an attention")


# ============================================================
# add_impulse 直接路径
# ============================================================

@pytest.mark.asyncio
async def test_add_impulse_creates_attention_with_direct_source():
    """直接注入的 impulse 应触发 attention 创建, source 锚定为 _direct."""
    mindflow = _new_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        impulse = _imp(messages=[Message.new().with_content('inject')])
        mindflow.add_impulse(impulse)
        attention = await asyncio.wait_for(_first_attention(mindflow), timeout=2.0)
        async with attention:
            drawn = attention.draw_from()
            assert drawn is impulse  # 没有拷贝, 引用透传
            assert drawn.source == _DirectImpulseNucleus.NAME
            assert drawn.source == '_direct'
            attention.abort('test done')


@pytest.mark.asyncio
async def test_add_impulse_when_not_running_is_no_op():
    """未启动 mindflow 时调用 add_impulse 应被静默丢弃, 不抛异常."""
    mindflow = _new_mindflow()
    # 不进入 async with → not running.
    impulse = _imp()
    mindflow.add_impulse(impulse)  # 不应抛
    # direct nucleus 内部 cache 仍为空 (because add_impulse early return).
    assert mindflow._direct_nucleus.peek() is None


@pytest.mark.asyncio
async def test_add_impulse_drops_stale_impulse():
    """stale impulse 应在入口被丢弃, 不进入 cache."""
    mindflow = _new_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        import time
        stale = _imp()
        stale.stale_timeout = 0.01
        time.sleep(0.02)
        assert stale.is_stale()
        mindflow.add_impulse(stale)
        assert mindflow._direct_nucleus.peek() is None


@pytest.mark.asyncio
async def test_add_impulse_last_wins_before_consume():
    """连续两次 add_impulse, consume 前 cache 只保留最新的 (last-impulse cache 语义)."""
    mindflow = _new_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        first = _imp(messages=[Message.new().with_content('first')])
        second = _imp(messages=[Message.new().with_content('second')])
        # Pause mindflow 防止 consume 抢先消费.
        mindflow.pause(True)
        mindflow.pause(False)  # 实际上 paused 的 add_impulse 会被丢弃, 改用别的方式.
        # 直接用 nucleus set_impulse 测纯 cache 语义.
        mindflow._direct_nucleus.set_impulse(first)
        mindflow._direct_nucleus.set_impulse(second)
        assert mindflow._direct_nucleus.peek() is second


# ============================================================
# ImpulsePrimitive 组合行为
# ============================================================

@pytest.mark.asyncio
async def test_execute_command_only_propagates_thinking_effort_none():
    """execute_command_only: thinking_effort='none' 应卸载到 attention/articulator;
    command_logos 应进入 moment.command_logos."""
    mindflow = _new_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        base = _imp(messages=[Message.new().with_content('go')])
        ImpulsePrimitive.execute_command_only(base, command_logos='do_it')
        mindflow.add_impulse(base)
        attention = await asyncio.wait_for(_first_attention(mindflow), timeout=2.0)
        async with attention:
            # 协议: thinking_effort='none' 落到 attention.
            assert attention.thinking_effort == 'none'
            # 通过 loop yield 的 articulator 反映同一值.
            art, act = await anext(attention.loop())
            async with art, act:
                assert art.thinking_effort() == 'none'
                # command_logos 沉淀到 moment.
                assert art.moment.command_logos == 'do_it'
            attention.abort('test done')


@pytest.mark.asyncio
async def test_superior_execute_command_uses_fatal_priority():
    """superior_execute_command: 在 execute_command_only 基础上加 FATAL.
    协议命题: FATAL 应能抢占任意普通 priority defender."""
    mindflow = _new_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        # 先注入一个普通 NOTICE defender.
        mindflow.add_impulse(_imp(priority=Priority.NOTICE,
                                  messages=[Message.new().with_content('defender')]))
        defender_att = await asyncio.wait_for(_first_attention(mindflow), timeout=2.0)
        # 进入 defender 但不消费, 等待被抢占.
        loop_gen = mindflow.loop()
        async with defender_att:
            # 注入 superior_execute_command — 应抢占成功.
            challenger = _imp(messages=[Message.new().with_content('cmd')])
            ImpulsePrimitive.superior_execute_command(challenger, command_logos='sup_cmd')
            assert challenger.priority == Priority.FATAL.value
            mindflow.add_impulse(challenger)
            # 等待 defender 被 abort.
            await asyncio.wait_for(defender_att.wait_aborted(), timeout=2.0)
            assert defender_att.is_aborted()
        # 新 attention 应已创建.
        new_att = await asyncio.wait_for(anext(loop_gen), timeout=2.0)
        async with new_att:
            assert new_att.thinking_effort == 'none'
            new_att.abort('test done')


@pytest.mark.asyncio
async def test_interrupt_only_silent_buffers_without_new_attention():
    """interrupt_only: FATAL + silent + thinking_effort='none'.
    协议命题: silent 模式抢占成功后不创建新 attention, messages 进入 mindflow buffer."""
    mindflow = _new_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        # 先创建 defender attention.
        mindflow.add_impulse(_imp(priority=Priority.NOTICE,
                                  messages=[Message.new().with_content('defender')]))
        defender_att = await asyncio.wait_for(_first_attention(mindflow), timeout=2.0)
        async with defender_att:
            # 注入 interrupt_only.
            silent_imp = _imp(messages=[Message.new().with_content('silent_msg')])
            ImpulsePrimitive.interrupt_only(silent_imp)
            assert silent_imp.mode == ChallengeMode.silent.value
            assert silent_imp.priority == Priority.FATAL.value
            mindflow.add_impulse(silent_imp)
            # 给 consume loop 时间.
            await asyncio.sleep(0.2)
            # 协议命题 1: defender 没有被 abort (silent 不会替换 attention).
            assert not defender_att.is_aborted()
            # 协议命题 2: silent 的 messages 进入 mindflow buffer.
            buffered = mindflow.get_buffered(pop=False)
            buffered_texts = [c['text'] for m in buffered for c in m.contents if 'text' in c]
            assert 'silent_msg' in buffered_texts
            defender_att.abort('test done')


@pytest.mark.asyncio
async def test_always_buffer_notify_buffers_without_new_attention():
    """always_buffer: BACKGROUND + notify.
    协议命题: BACKGROUND 不抢占; notify 模式失败时 messages 进 mindflow buffer."""
    mindflow = _new_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        # 先创建普通 defender attention.
        mindflow.add_impulse(_imp(priority=Priority.NOTICE,
                                  messages=[Message.new().with_content('defender')]))
        defender_att = await asyncio.wait_for(_first_attention(mindflow), timeout=2.0)
        async with defender_att:
            # 注入 always_buffer.
            bg_imp = _imp(messages=[Message.new().with_content('bg_msg')])
            ImpulsePrimitive.always_buffer(bg_imp)
            assert bg_imp.priority == Priority.BACKGROUND.value
            assert bg_imp.mode == ChallengeMode.notify.value
            mindflow.add_impulse(bg_imp)
            await asyncio.sleep(0.2)
            # 协议命题 1: defender 仍在 (BACKGROUND 永不抢占).
            assert not defender_att.is_aborted()
            # 协议命题 2: notify 失败时 messages 进 buffer.
            buffered = mindflow.get_buffered(pop=False)
            buffered_texts = [c['text'] for m in buffered for c in m.contents if 'text' in c]
            assert 'bg_msg' in buffered_texts
            defender_att.abort('test done')
