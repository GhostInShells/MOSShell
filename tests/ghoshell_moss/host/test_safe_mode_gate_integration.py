"""SafeMode gate 与 articulator.create_task 集成回归测试.

Round 1 (e60b4fda) 里 `_run_articulator` 写成:

    await articulator.create_task(asyncio.wrap_future(verdict_future))

`asyncio.wrap_future` 返回 Future 不是 coroutine, `articulator.create_task`
内调 `event_loop.create_task(cor)`, uvloop 严格要求 coroutine → TypeError.
错误被 `_run_articulator` 的 `except Exception` 静默吞成 log 一行, 表面上
pub_logos 仍显示 raw stream, gate 每次 activate 都在崩, 直到 round 2 手动
测试才看到日志里的错误.

修法: 用 async 函数包一层再喂 create_task.

单元测试 (test_safe_mode.py) 只测 SafeModeImpl 自身状态转移, 覆盖不到 gate
与 articulator 的实际 await 路径. 这个 gap 才是 round 1 bug 潜伏几周的根因.
"""

import asyncio

import pytest
from ghoshell_moss.core.blueprint.host import Verdict
from ghoshell_moss.core.blueprint.memento import Reaction
from ghoshell_moss.core.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_moss.core.mindflow.base_attention import (
    AttentionContext,
    BaseArticulator,
)
from ghoshell_moss.host.safe_mode import SafeModeImpl


def _make_ctx() -> AttentionContext:
    return AttentionContext(
        attention_id="test",
        moment=Reaction().new_moment(),
        aborted_event=ThreadSafeEvent(),
        flags={},
    )


def _make_articulator() -> BaseArticulator:
    return BaseArticulator(
        ctx=_make_ctx(),
        exited_event=ThreadSafeEvent(),
        thinking_effort='low',
    )


@pytest.mark.asyncio
async def test_gate_approve_via_articulator_create_task():
    """SafeMode.submit() 返回的 Future 必须能通过 articulator.create_task 成功 await.

    反例 (round 1 bug):
        await articulator.create_task(asyncio.wrap_future(future))
        # TypeError: a coroutine was expected, got <Future pending>

    正例 (round 2 修法): 用 async 函数包 wrap_future.
    """
    articulator = _make_articulator()
    sm = SafeModeImpl()
    sm.set_enabled(True)
    verdict_future = sm.submit("test logos")

    async with articulator:

        async def _fake_tui_approve():
            await asyncio.sleep(0.01)
            uuid = sm.pending()['uuid']
            sm.approve(uuid, note="looks good")

        approve_task = asyncio.create_task(_fake_tui_approve())

        async def _await_verdict():
            return await asyncio.wrap_future(verdict_future)

        verdict = await articulator.create_task(_await_verdict())
        await approve_task

    assert verdict == Verdict(kind='approved', message="looks good")


@pytest.mark.asyncio
async def test_gate_reject_via_articulator_create_task():
    """Reject 路径同样能通过 articulator.create_task 正常拿到 verdict."""
    articulator = _make_articulator()
    sm = SafeModeImpl()
    sm.set_enabled(True)
    verdict_future = sm.submit("risky logos")

    async with articulator:

        async def _fake_tui_reject():
            await asyncio.sleep(0.01)
            uuid = sm.pending()['uuid']
            sm.reject(uuid, "too risky")

        reject_task = asyncio.create_task(_fake_tui_reject())

        async def _await_verdict():
            return await asyncio.wrap_future(verdict_future)

        verdict = await articulator.create_task(_await_verdict())
        await reject_task

    assert verdict == Verdict(kind='rejected', message="too risky")


@pytest.mark.asyncio
async def test_gate_cancel_current_resolves_awaiter():
    """cancel_current 由拦截点 finally 幂等调用 (abort 或 shutdown 兜底);
    awaiter 应该拿到 cancelled verdict 而非挂死."""
    articulator = _make_articulator()
    sm = SafeModeImpl()
    sm.set_enabled(True)
    verdict_future = sm.submit("logos")

    async with articulator:

        async def _fake_cancel():
            await asyncio.sleep(0.01)
            sm.cancel_current()

        cancel_task = asyncio.create_task(_fake_cancel())

        async def _await_verdict():
            return await asyncio.wrap_future(verdict_future)

        verdict = await articulator.create_task(_await_verdict())
        await cancel_task

    assert verdict.kind == 'cancelled'
