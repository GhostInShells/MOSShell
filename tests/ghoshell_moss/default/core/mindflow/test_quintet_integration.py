"""五元 nucleus 集成测试 — 协议层等价性钉住.

把 InputSignalNucleus / SilentNucleus / NotifyNucleus / CommandNucleus /
InterruptNucleus 注册到同一 mindflow, 用各自的 SignalMeta 发 signal,
断言行为等价于绕过 nucleus 直接 add_impulse(primitive 路径).

这是 "Signal 作为开发者唯一接触面" 的核心命题: 上层只需懂 signal name +
priority, 不需要懂 ChallengeMode × thinking_effort × priority × logos
的正交组合. nucleus 层把组合糖封装好.

集成测试是协议层最后的兜底 — 单 nucleus 测过了 primitive 卸载, 这里测
"signal 入口 → nucleus → mindflow → attention/buffer" 全链路的行为不变性.

ImpulsePrimitive.broadcast 作为单原语对照纳入, 验证退回 primitive 后仍可用.

API 迁移说明 (重构 41f0cb63 — decouple attention/thinking/action):
    - ``mindflow.attention_loop()`` / ``attention.attention_loop()``
        → ``mindflow.thinking_loop()`` 产出 ``Thinking``, 经 ``Thinking.attention`` 取 Attention.
    - ``attention.abort_attention('reason')`` → ``thinking.abort('reason')``
    - ``attention.is_attention_aborted()`` → ``attention.is_aborted()``
    - ``attention.effort`` → ``thinking.effort()``
    - ``mindflow.project_percepts()`` → ``mindflow.moments.peek().percepts_messages()``
    - ``mindflow.faculties()`` → ``mindflow.nuclei()``
"""
import asyncio
import pytest

from ghoshell_moss.core.blueprint.mindflow import (
    ChallengeMode, Impulse, ImpulsePrimitive, Priority,
)
from ghoshell_moss.core.mindflow import (
    BaseMindflow,
    InputSignalNucleus,
    CommandNucleus,
    NotifyNucleus,
    SilentNucleus,
    InterruptNucleus,
)
from ghoshell_moss.core.mindflow.command_nucleus import new_command_signal
from ghoshell_moss.core.mindflow.notify_nucleus import new_notify_signal
from ghoshell_moss.core.mindflow.silent_nucleus import new_silent_signal
from ghoshell_moss.core.mindflow.interrupt_nucleus import new_interrupt_signal
from ghoshell_moss.core.blueprint.mindflow import InputSignalMeta
from ghoshell_moss.message import Message


# ============================================================
# Helpers
# ============================================================


def _quintet_mindflow() -> BaseMindflow:
    """五元 nucleus 全注册."""
    return BaseMindflow(
        InputSignalNucleus(),
        CommandNucleus(),
        NotifyNucleus(),
        SilentNucleus(suppress_seconds=0.05),
        InterruptNucleus(suppress_seconds=0.05),
    )


async def _first_thinking(mindflow, timeout: float = 2.0):
    """阻塞拿第一个 Thinking.
    async-for + return 会让生成器 finally 重置 _is_looping_thinking, 因此同一
    mindflow 下可重复调用以获取后续 thinking. 超时则 fail."""
    async for thinking in mindflow.thinking_loop():
        return thinking
    raise AssertionError("mindflow.thinking_loop() exited without yielding")


def _texts(messages) -> list[str]:
    return [c['text'] for m in messages for c in m.contents if 'text' in c]


def _input_signal(text: str, *, priority: Priority = Priority.NOTICE):
    return InputSignalMeta().to_signal(
        Message.new().with_content(text),
        priority=priority,
    )


# ============================================================
# 单 nucleus 端到端 — 验证 signal → attention 链路
# ============================================================


@pytest.mark.asyncio
async def test_input_signal_drives_attention():
    """InputSignalNucleus: input signal → default mode → attention with effort=''."""
    mindflow = _quintet_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        mindflow.add_signal(_input_signal('hello'))
        thinking = await asyncio.wait_for(_first_thinking(mindflow), timeout=2.0)
        async with thinking:
            impulse = thinking.attention.draw_from()
            assert impulse.source == 'input_signal_nucleus'
            assert impulse.priority == Priority.NOTICE
            assert impulse.mode == ''  # default
            assert impulse.thinking_effort == ''
            thinking.abort('test done')


@pytest.mark.asyncio
async def test_command_signal_drives_attention_with_logos():
    """CommandNucleus: command signal → command_only impulse → moment.command_logos 填好."""
    mindflow = _quintet_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        mindflow.add_signal(new_command_signal('exec_me'))
        thinking = await asyncio.wait_for(_first_thinking(mindflow), timeout=2.0)
        async with thinking:
            assert thinking.effort() == 'none'
            assert thinking.moment.command_logos == 'exec_me'
            thinking.abort('test done')


@pytest.mark.asyncio
async def test_notify_signal_default_path_creates_attention():
    """NotifyNucleus quiet 路径: 创建 attention (跟 default 一样)."""
    mindflow = _quintet_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        mindflow.add_signal(new_notify_signal(Message.new().with_content('user_msg')))
        thinking = await asyncio.wait_for(_first_thinking(mindflow), timeout=2.0)
        async with thinking:
            impulse = thinking.attention.draw_from()
            assert impulse.source == 'notify_nucleus'
            assert impulse.mode == ChallengeMode.notify.value
            thinking.abort('test done')


@pytest.mark.asyncio
async def test_silent_signal_quiet_path_buffers_not_attention():
    """SilentNucleus quiet 路径: silent mode 不创建 attention, messages 进 mindflow buffer."""
    mindflow = _quintet_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        mindflow.add_signal(new_silent_signal(Message.new().with_content('quiet_data')))
        # 给 consume loop 时间.
        await asyncio.sleep(0.2)
        # quiet 系统下 silent 不应创建 attention.
        assert mindflow.attention() is None
        buffered = mindflow.moments.peek().percepts_messages()
        assert 'quiet_data' in _texts(buffered)


@pytest.mark.asyncio
async def test_interrupt_signal_creates_fatal_attention():
    """InterruptNucleus: interrupt signal → FATAL + notify + effort=none + interrupt=True."""
    mindflow = _quintet_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        mindflow.add_signal(new_interrupt_signal(Message.new().with_content('halt')))
        thinking = await asyncio.wait_for(_first_thinking(mindflow), timeout=2.0)
        async with thinking:
            impulse = thinking.attention.draw_from()
            assert impulse.priority == Priority.FATAL
            assert impulse.mode == ChallengeMode.notify.value
            assert impulse.thinking_effort == 'none'
            assert impulse.interrupt is True
            thinking.abort('test done')


# ============================================================
# Signal → Nucleus → primitive 等价性
# 核心命题: nucleus 路径产出的 impulse 字段应等价于直接 add_impulse(primitive 路径).
# ============================================================


def _build_impulse_via_command_nucleus(text: str, priority: Priority) -> Impulse:
    nuc = CommandNucleus()
    return nuc.build_impulse(new_command_signal(text, priority=priority))


def test_command_nucleus_equivalent_to_command_only_primitive():
    """CommandNucleus(signal) ≡ ImpulsePrimitive.command_only(impulse, logos)."""
    via_nucleus = _build_impulse_via_command_nucleus('do_x', Priority.NOTICE)
    via_primitive = ImpulsePrimitive.command_only(
        Impulse(priority=Priority.NOTICE, messages=[]),
        command_logos='do_x',
    )
    assert via_nucleus.logos == via_primitive.logos == 'do_x'
    assert via_nucleus.thinking_effort == via_primitive.thinking_effort == 'none'
    assert via_nucleus.priority == via_primitive.priority == Priority.NOTICE


def test_command_nucleus_fatal_equivalent_to_fatal_command_primitive():
    """priority=FATAL signal 走 CommandNucleus ≡ fatal_command primitive."""
    via_nucleus = _build_impulse_via_command_nucleus('halt', Priority.FATAL)
    via_primitive = ImpulsePrimitive.fatal_command(Impulse(messages=[]), 'halt')
    assert via_nucleus.priority == via_primitive.priority == Priority.FATAL
    assert via_nucleus.logos == via_primitive.logos == 'halt'
    assert via_nucleus.thinking_effort == via_primitive.thinking_effort == 'none'


def test_notify_nucleus_equivalent_to_notify_primitive():
    nuc = NotifyNucleus()
    via_nucleus = nuc.build_impulse(new_notify_signal(
        Message.new().with_content('m'),
        priority=Priority.NOTICE,
    ))
    via_primitive = ImpulsePrimitive.notify(Impulse(
        priority=Priority.NOTICE,
        messages=[Message.new().with_content('m')],
    ))
    assert via_nucleus.mode == via_primitive.mode == ChallengeMode.notify.value
    assert via_nucleus.priority == via_primitive.priority == Priority.NOTICE


def test_interrupt_nucleus_equivalent_to_interrupt_primitive():
    nuc = InterruptNucleus()
    via_nucleus = nuc.build_impulse(new_interrupt_signal(Message.new().with_content('m')))
    via_primitive = ImpulsePrimitive.interrupt(Impulse(messages=[Message.new().with_content('m')]))
    # 4 字段完全一致.
    assert via_nucleus.priority == via_primitive.priority == Priority.FATAL
    assert via_nucleus.mode == via_primitive.mode == ChallengeMode.notify.value
    assert via_nucleus.thinking_effort == via_primitive.thinking_effort == 'none'
    assert via_nucleus.interrupt == via_primitive.interrupt is True


def test_broadcast_primitive_standalone_usability():
    """ImpulsePrimitive.broadcast 作为单原语仍可被 add_impulse 直接使用,
    退回 primitive 不影响调用. 这是"原语足够时不抽 nucleus"的承诺."""
    base = Impulse(messages=[Message.new().with_content('alert')])
    broadcast = ImpulsePrimitive.broadcast(base)
    assert broadcast.priority == Priority.FATAL
    assert broadcast.mode == ChallengeMode.silent.value
    assert broadcast.thinking_effort == 'none'
    assert broadcast.interrupt is False  # broadcast 不带 interrupt


# ============================================================
# 两两交互 — 五元在同一 mindflow 的并发场景
# ============================================================


@pytest.mark.asyncio
async def test_input_then_silent_silent_buffers_into_input_attention():
    """input 占住 attention, silent 抢占成功后 buffer messages → 下一帧 attention 从 percepts 看见."""
    mindflow = _quintet_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        # 先 input 占 attention.
        mindflow.add_signal(_input_signal('user_says', priority=Priority.NOTICE))
        defender = await asyncio.wait_for(_first_thinking(mindflow), timeout=2.0)
        async with defender:
            # silent FATAL: 抢占成功但 silent mode 偏离 default — buffer, 不接管 attention.
            mindflow.add_signal(new_silent_signal(
                Message.new().with_content('quiet_supplement'),
                priority=Priority.FATAL,
            ))
            await asyncio.sleep(0.2)
            # silent 抢占成功不接管 attention, 原 attention 仍活.
            assert not defender.is_aborted()
            # messages 进 buffer.
            buffered = mindflow.moments.peek().percepts_messages()
            assert 'quiet_supplement' in _texts(buffered)
            defender.abort('test done')


@pytest.mark.asyncio
async def test_input_then_interrupt_interrupt_aborts_input_attention():
    """input 占 attention, interrupt 必抢占 → defender abort + 新 attention 持 interrupt 字段."""
    mindflow = _quintet_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        mindflow.add_signal(_input_signal('user_says'))
        defender = await asyncio.wait_for(_first_thinking(mindflow), timeout=2.0)
        async with defender:
            att1 = defender.attention
            mindflow.add_signal(new_interrupt_signal(Message.new().with_content('halt')))
            await asyncio.wait_for(att1.wait_abort(), timeout=2.0)
            assert att1.is_aborted()
        # 新 attention 应已创建并持有 interrupt 字段.
        interrupt_think = await asyncio.wait_for(_first_thinking(mindflow), timeout=2.0)
        async with interrupt_think:
            impulse = interrupt_think.attention.draw_from()
            assert impulse.interrupt is True
            assert impulse.thinking_effort == 'none'
            interrupt_think.abort('test done')


@pytest.mark.asyncio
async def test_input_then_notify_notify_fails_and_buffers():
    """input 占 attention + 保护期内, 同优先级 notify 抢占失败 → messages 进 buffer."""
    mindflow = _quintet_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        # 注入带保护期的 input.
        mindflow.add_impulse(Impulse(
            priority=Priority.NOTICE,
            protection_time=10.0,
            messages=[Message.new().with_content('defender')],
        ))
        defender = await asyncio.wait_for(_first_thinking(mindflow), timeout=2.0)
        async with defender:
            # 同优先级 notify, 保护期内必败.
            mindflow.add_signal(new_notify_signal(
                Message.new().with_content('user_msg'),
                priority=Priority.NOTICE,
            ))
            await asyncio.sleep(0.2)
            # defender 仍在.
            assert not defender.is_aborted()
            # messages 进 buffer (notify 偏离侧).
            buffered = mindflow.moments.peek().percepts_messages()
            assert 'user_msg' in _texts(buffered)
            defender.abort('test done')


@pytest.mark.asyncio
async def test_input_then_command_fatal_command_takes_over():
    """input 占 attention, FATAL command 抢占成功 → 新 attention 带 command_logos."""
    mindflow = _quintet_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        mindflow.add_signal(_input_signal('user_says'))
        defender = await asyncio.wait_for(_first_thinking(mindflow), timeout=2.0)
        async with defender:
            att1 = defender.attention
            # FATAL command 抢占.
            mindflow.add_signal(new_command_signal('sup_cmd', priority=Priority.FATAL))
            await asyncio.wait_for(att1.wait_abort(), timeout=2.0)
            assert att1.is_aborted()
        cmd_think = await asyncio.wait_for(_first_thinking(mindflow), timeout=2.0)
        async with cmd_think:
            assert cmd_think.effort() == 'none'
            assert cmd_think.moment.command_logos == 'sup_cmd'
            cmd_think.abort('test done')


@pytest.mark.asyncio
async def test_silent_aggregates_multiple_signals_into_one_buffer_drain():
    """SilentNucleus 聚合多 signal → 一个 impulse → buffer 一次性 drain 多条 messages."""
    mindflow = _quintet_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        # 三个 silent signal 连续进, SilentNucleus 内部聚合.
        for i in range(3):
            mindflow.add_signal(new_silent_signal(
                Message.new().with_content(f'data_{i}'),
                priority=Priority.FATAL,  # 保证胜出
            ))
        await asyncio.sleep(0.2)
        # quiet → silent 不创建 attention.
        assert mindflow.attention() is None
        buffered = mindflow.moments.peek().percepts_messages()
        texts = _texts(buffered)
        # 三条都应在 buffer 里.
        for i in range(3):
            assert f'data_{i}' in texts


# ============================================================
# 五元拓扑健康检查 — 互不串流
# ============================================================


@pytest.mark.asyncio
async def test_signal_namespace_isolation():
    """signal 路由按 name 严格隔离: input signal 不应触发 command/notify/silent/interrupt nucleus."""
    mindflow = _quintet_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        mindflow.add_signal(_input_signal('only_input'))
        thinking = await asyncio.wait_for(_first_thinking(mindflow), timeout=2.0)
        async with thinking:
            impulse = thinking.attention.draw_from()
            # 来自 input_signal_nucleus, 不是别的.
            assert impulse.source == 'input_signal_nucleus'
            thinking.abort('test done')


@pytest.mark.asyncio
async def test_quintet_nuclei_discovered_in_mindflow():
    """五元 nucleus 都应正确注册到 mindflow.nuclei()."""
    mindflow = _quintet_mindflow()
    async with mindflow:
        await mindflow.wait_started()
        faculties = mindflow.nuclei()
        # 五元 + 内置 _direct (来自 base_mindflow add_impulse 入口) = 6.
        assert 'input_signal_nucleus' in faculties
        assert 'command_nucleus' in faculties
        assert 'notify_nucleus' in faculties
        assert 'silent_nucleus' in faculties
        assert 'interrupt_nucleus' in faculties
