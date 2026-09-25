"""
Attention.wait_ready 协议 — 首包抢占 / 尾包接力.

完全隔离 attention 层 (不依赖 mindflow 调度, 不依赖 Action/Think).
去生命周期化后, wait_ready 是"等第一个 complete impulse"的独立原语:

- complete 初始 impulse: 构造函数播种后事件已置位, wait_ready() 立即返回该 impulse, 不挂死.
- partial 初始 impulse: 首包只占注意力, 事件未置位, wait_ready() 阻塞;
  同 id complete 尾包被 absorb_impulse 吸收后返回尾包.
"""
import asyncio
import pytest

from ghoshell_moss.core.blueprint.mindflow import Impulse, Priority
from ghoshell_moss.core.mindflow import BaseAttention
from ghoshell_moss.message import Message


def _imp(
        *,
        complete: bool = True,
        id: str | None = None,
        priority: int = Priority.NOTICE,
        messages: list[Message] | None = None,
) -> Impulse:
    kwargs = dict(
        complete=complete,
        priority=priority,
        messages=messages or [Message.new().with_content('m')],
    )
    if id is not None:
        kwargs['id'] = id
    return Impulse(**kwargs)


@pytest.mark.asyncio
async def test_wait_ready_returns_init_impulse_when_complete():
    """complete 初始 impulse: wait_ready() 立即返回, 不需要额外驱动."""
    imp = _imp()
    att = BaseAttention(impulse=imp)
    result = await asyncio.wait_for(att.wait_ready(), timeout=2.0)
    assert result.id == imp.id
    assert result.complete


@pytest.mark.asyncio
async def test_wait_ready_blocks_on_partial_then_returns_complete_tail():
    """partial 首包占据注意力: wait_ready() 阻塞; 同 id complete 尾包被 absorb 后返回尾包."""
    partial = _imp(complete=False, id='shared-id')
    att = BaseAttention(impulse=partial)
    # 无 complete 尾包, 不返回 (partial 只占注意力).
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(att.wait_ready(), timeout=0.1)
    # 同 id complete 尾包接管.
    tail = _imp(complete=True, id='shared-id')
    att.absorb_impulse(tail)
    result = await asyncio.wait_for(att.wait_ready(), timeout=2.0)
    assert result.id == 'shared-id'
    assert result.complete


@pytest.mark.asyncio
async def test_wait_ready_unblocks_on_abort():
    """partial 首包占住注意力, 未等尾包即 abort: wait_ready() 应返回, 不挂死."""
    partial = _imp(complete=False, id='shared-id')
    att = BaseAttention(impulse=partial)
    att.abort('preempted')
    result = await asyncio.wait_for(att.wait_ready(), timeout=2.0)
    assert result.id == 'shared-id'
    assert att.is_aborted()


@pytest.mark.asyncio
async def test_wait_ready_unblocks_on_exit():
    """partial 首包占住注意力, 上下文退出 (__aexit__): wait_ready() 应被唤醒返回."""
    partial = _imp(complete=False, id='shared-id')
    att = BaseAttention(impulse=partial)

    async def _wait_ready():
        return await att.wait_ready()

    async with att:
        task = asyncio.create_task(_wait_ready())
        await asyncio.sleep(0)  # 让 wait_ready 挂起.
        assert not task.done()
    # __aexit__ 已 abort 并唤醒 wait_ready.
    result = await asyncio.wait_for(task, timeout=2.0)
    assert result.id == 'shared-id'
    assert att.is_aborted()
