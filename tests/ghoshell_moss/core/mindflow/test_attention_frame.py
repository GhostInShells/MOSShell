"""
Attention frame 协议: Action.wait_ready + 帧间衔接.

完全隔离 attention 层 (不依赖 ghost_runtime, 不依赖 mindflow 调度).
关键协议命题:
- wait_ready 在 abort / 首个 delta 上正确返回, 不挂死
- wait_ready 缓存的 delta 由 received_logos 优先消费
- frame N 的 model logos (经 executed_logos) 沉淀到 frame N+1 的 moment.previous
- frame N 的 moment 实例上 logos / command_logos 不被 next_frame 操作丢失
"""
import asyncio
import pytest

from ghoshell_moss.core.blueprint.mindflow import Impulse, Reaction, Priority
from ghoshell_moss.core.mindflow.base_attention import BaseAttention
from ghoshell_moss.message import Message


def _imp(
        *,
        source: str = 'src',
        priority: int = Priority.NOTICE,
        logos: str = '',
        messages: list[Message] | None = None,
) -> Impulse:
    return Impulse(
        source=source,
        priority=priority,
        logos=logos,
        messages=messages or [Message.new().with_content('input')],
    )


def _attention(impulse: Impulse) -> BaseAttention:
    return BaseAttention(previous=Reaction(), impulse=impulse)


def _text_in_messages(messages, query: str) -> bool:
    for m in messages:
        for c in m.contents:
            if c.get('text') == query:
                return True
    return False


# ============================================================
# Action.wait_ready
# ============================================================

@pytest.mark.asyncio
async def test_wait_ready_returns_immediately_when_already_aborted():
    """协议: 进入 wait_ready 时若已 aborted, 立刻返回. received_logos 应不 yield."""
    att = _attention(_imp())
    async with att:
        art, act = await anext(att.loop())
        async with art, act:
            att.abort('before wait')
            await act.wait_ready()  # 不应挂死
            received = [d async for d in act.received_logos()]
            assert received == []


@pytest.mark.asyncio
async def test_wait_ready_caches_first_delta_then_received_logos_drains_in_order():
    """协议: wait_ready 预取的首个 delta 应优先被 received_logos 消费, 后续从队列接力."""
    att = _attention(_imp())
    async with att:
        art, act = await anext(att.loop())
        async with art, act:
            art.send_nowait('first')
            await act.wait_ready()  # 预取 'first'
            art.send_nowait('second')
            received: list[str] = []
            async for delta in act.received_logos():
                received.append(delta)
                if len(received) == 2:
                    break
    assert received == ['first', 'second']


@pytest.mark.asyncio
async def test_wait_ready_returns_when_aborted_mid_wait():
    """协议: wait_ready 阻塞期间被 abort, 应正常返回, 不挂死."""
    att = _attention(_imp())
    async with att:
        art, act = await anext(att.loop())

        async def _abort_after_delay():
            await asyncio.sleep(0.1)
            att.abort('mid-wait abort')

        async with art, act:
            asyncio.create_task(_abort_after_delay())
            # 没有 delta, 没有 abort 前会等. 测试不挂死表示协议生效.
            await asyncio.wait_for(act.wait_ready(), timeout=2.0)
            received = [d async for d in act.received_logos()]
            assert received == []


# ============================================================
# Frame-to-frame carryover
# ============================================================

@pytest.mark.asyncio
async def test_frame_carryover_preserves_logos_and_command_logos():
    """关键协议命题 (单 case):

    模拟 articulator + action 完整消费 frame N, 触发 observe → frame N+1:
    - frame N+1 的 moment.previous.moment_id 衔接 frame N 的 moment.id
    - frame N 的 model logos 通过 buffer_executed_logos → executed_logos 沉淀到 previous
    - frame N 的 outcome messages 沉淀到 previous.messages
    - frame N 的 moment 实例上 logos / command_logos 在 next_frame 之后仍可读 (引用稳定)
    """
    impulse = _imp(logos='reflex!')  # impulse.logos → moment.command_logos
    att = _attention(impulse)

    async with att:
        loop_gen = att.loop()

        # === Frame 1 ===
        art1, act1 = await anext(loop_gen)
        # moment 必须在 articulator 生命周期内读 (_check_running 守卫).
        # 但 Moment 实例本身可以在 articulator 退出后被局部引用持有.
        moment1 = None
        received: list[str] = []

        async with art1, act1:
            moment1 = art1.moment
            # impulse.logos 经 _prepare_moment 落到 moment.command_logos.
            assert moment1.command_logos == 'reflex!'

            art1.send_nowait('model says hi')
            # 模拟 ghost_runtime _run_articulator finally 块的赋值.
            moment1.logos = 'model says hi'
            # 在块内直接 drain — 用 break 跳出, 避免 create_task 被 __aexit__ 提前 cancel.
            async for delta in act1.received_logos():
                received.append(delta)
                break
            act1.outcome(Message.new().with_content('done'), observe=True)

        assert received == ['model says hi']

        # === Frame 2 ===
        art2, act2 = await anext(loop_gen)
        async with art2, act2:
            moment2 = art2.moment

            # 协议命题 1: previous Reaction 衔接.
            assert moment2.previous is not None
            assert moment2.previous.moment_id == moment1.id

            # 协议命题 2: model logos 经 buffer_executed_logos 沉淀.
            assert moment2.previous.executed_logos == 'model says hi'

            # 协议命题 3: 上一帧 outcome messages 在 previous.
            assert _text_in_messages(moment2.previous.messages, 'done')

            # 协议命题 4: 上一帧 moment 实例字段稳定 — 不被 next_frame 清空.
            assert moment1.logos == 'model says hi'
            assert moment1.command_logos == 'reflex!'

            # 协议命题 5: 新帧的 moment 是独立实例, 字段重置 (无残留).
            assert moment2.id != moment1.id
            assert moment2.command_logos == ''  # impulse 已 drain, 不重复应用
            assert moment2.logos == ''  # 新帧, 模型尚未输出
            assert moment2.percepts == []

            att.abort('test done')
