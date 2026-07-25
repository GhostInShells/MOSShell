"""BaseAction 空流场景单测.

聚焦决策 6 (ghost-runtime-safemode FEATURE): _logos 预取 None 的 early-return
是不可达死代码 — wait_ready 把 None 塞进 _prefetched_delta 后, _logos 的
``if self._prefetched_delta is not None`` 挡住预取的 None, 空 articulator 时
action 挂死在 1s 轮询直到外部 abort.

SafeMode 的否决路径必然产生空流 (deny 时不 send_nowait, __aexit__ 直接放 None),
这个 bug 必须先修.
"""

import asyncio

import pytest
from ghoshell_moss.core.mindflow.base_attention import (
    AttentionContext,
    BaseAction,
    BaseArticulator,
)
from ghoshell_moss.core.blueprint.memento import Reaction
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent


def _make_ctx() -> AttentionContext:
    return AttentionContext(
        attention_id="test",
        moment=Reaction().new_moment(),
        aborted_event=ThreadSafeEvent(),
        flags={},
    )


@pytest.mark.asyncio
async def test_empty_stream_received_logos_returns_immediately():
    """空 articulator (context 打开后直接关) → received_logos() 零 yield 立即返回.

    验证决策 6: 否决路径 (articulator 一个 delta 都不 send) 时 action 不应
    挂在 1s 轮询上.
    """
    ctx = _make_ctx()
    articulate_exited = ThreadSafeEvent()
    action_exited = ThreadSafeEvent()

    articulator = BaseArticulator(
        ctx=ctx,
        exited_event=articulate_exited,
        thinking_effort='low',
    )
    action = BaseAction(ctx=ctx, exited_event=action_exited)

    # 模拟真实 GhostRuntime 路径: articulator 开一次 context 立即关 (空 articulate).
    async with articulator:
        pass  # __aexit__ 会 put None 哨兵

    # action 应立即结束, 不能挂 1s 轮询.
    yielded: list[str] = []

    async def _consume():
        async with action:
            await action.wait_ready()
            async for delta in action.received_logos():
                yielded.append(delta)

    await asyncio.wait_for(_consume(), timeout=0.3)
    assert yielded == []
