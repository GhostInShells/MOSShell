"""SafeMode gate (approve 回调) 与 thinking/action 剥离后的裁决端到端测试.

定位 (与 test_safe_mode.py / test_thinking.py 的分工):
    - test_safe_mode.py          测 SafeModeImpl 自身状态转移, 不接 thinking.
    - test_thinking.py           测 BaseThinking/BaseArticulator/BaseAction 装线契约.
    - 本文件                     测 gate 的 approve 回调契约 + SafeMode 裁决端到端.

gate 契约 (重做后):
    ``ActionGate.approve(logos) -> (approved, message)`` 是 thinking 级别单一回调。
    articulator 在 commit (logos 写完) 时 await 它 — approved 才投递 action,
    rejected/cancelled 直接 abort action (进而 abort attention), 不重新 loop。
    回调默认放行 ``(True, '')``。
"""

import asyncio
import contextlib

import pytest

from ghoshell_moss.core.blueprint.mindflow import Impulse, Priority
from ghoshell_moss.core.blueprint.moment import BaseMomentsObserver
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.core.mindflow import BaseActionGate, BaseAttention, BaseThinking
from ghoshell_moss.host.safe_mode import SafeModeImpl


def _make_thinking(put_action) -> BaseThinking:
    """构造 BaseThinking (从三循环剥离). gate 惰性创建, 由 register 启用 gated."""
    attention = BaseAttention(impulse=Impulse(
        source='test',
        priority=Priority.NOTICE,
        thinking_effort='low',
        strength=100,
    ))
    return BaseThinking(
        attention=attention,
        observer=BaseMomentsObserver(max_size=10),
        put_action=put_action,
        mindflow_stop_event=ThreadSafeEvent(),
    )


def _safe_mode_approve(sm: SafeModeImpl):
    """SafeMode 裁决回调 — 与 GhostRuntime._approve_logos 同构.

    submit → await verdict → (approved, message). finally cancel_current 幂等兜底.
    """

    async def _approve(logos: str) -> tuple[bool, str]:
        verdict_future = sm.submit(logos)
        try:
            verdict = await asyncio.wrap_future(verdict_future)
            return (verdict.kind == 'approved', verdict.message)
        finally:
            sm.cancel_current()

    return _approve


async def _wait_pending(sm: SafeModeImpl):
    for _ in range(200):
        p = sm.pending()
        if p:
            return p
        await asyncio.sleep(0.01)
    raise TimeoutError("no pending approval")


# ── ActionGate approve 回调契约 ────────────────────────────────────────


@pytest.mark.asyncio
async def test_gate_approve_defaults_true_without_callback():
    """未注册回调的 gate 默认放行."""
    gate = BaseActionGate()
    assert await gate.approve("anything") == (True, '')


@pytest.mark.asyncio
async def test_gate_approve_delegates_to_callback():
    """注册回调后, approve 返回回调的裁决结果."""
    gate = BaseActionGate()

    async def _deny(logos: str) -> tuple[bool, str]:
        return (False, f"denied: {logos}")

    gate.register(_deny)
    assert await gate.approve("hello") == (False, "denied: hello")


# ── SafeMode 裁决端到端 (真实 thinking + approve 回调) ─────────────────


@pytest.mark.asyncio
async def test_gate_approved_logos_delivered_to_action():
    """approve: 回调返回 (True, msg) → logos 投递到 action."""
    put = []
    thinking = _make_thinking(put.append)
    sm = SafeModeImpl()
    sm.set_enabled(True)
    thinking.register_gate(_safe_mode_approve(sm))

    async def articulate():
        art = thinking.articulator()
        async with art:
            art.send_nowait("<probe:say hello/>")
            await art.wait_action_done()

    art_task = asyncio.create_task(articulate())
    try:
        p = await asyncio.wait_for(_wait_pending(sm), 2.0)
        sm.approve(p['uuid'], note="looks good")

        # approve → _commit 投递 action.
        for _ in range(100):
            if put:
                break
            await asyncio.sleep(0.01)
        assert len(put) == 1

        action = put[0]
        logos = ''
        async with action:
            async for delta in action.logos():
                logos += delta
            action.set_compiled()
        assert logos == "<probe:say hello/>"
        await asyncio.wait_for(art_task, 2.0)
    finally:
        art_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await art_task


@pytest.mark.asyncio
async def test_gate_rejected_logos_abort_without_delivery():
    """reject: 回调返回 (False, msg) → action abort, 不投递."""
    put = []
    thinking = _make_thinking(put.append)
    sm = SafeModeImpl()
    sm.set_enabled(True)
    thinking.register_gate(_safe_mode_approve(sm))

    async def articulate():
        art = thinking.articulator()
        async with art:
            art.send_nowait("<probe:drop/>")
            await art.wait_action_done()

    art_task = asyncio.create_task(articulate())
    try:
        p = await asyncio.wait_for(_wait_pending(sm), 2.0)
        sm.reject(p['uuid'], "too risky")

        await asyncio.wait_for(art_task, 2.0)
        assert put == []
    finally:
        art_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await art_task


@pytest.mark.asyncio
async def test_gate_cancel_current_aborts_without_delivery():
    """cancel_current: 拦截点 finally 幂等调用, awaiter 拿 cancelled → abort 不投递."""
    put = []
    thinking = _make_thinking(put.append)
    sm = SafeModeImpl()
    sm.set_enabled(True)
    thinking.register_gate(_safe_mode_approve(sm))

    async def articulate():
        art = thinking.articulator()
        async with art:
            art.send_nowait("<probe:mid/>")
            await art.wait_action_done()

    art_task = asyncio.create_task(articulate())
    try:
        await asyncio.wait_for(_wait_pending(sm), 2.0)
        sm.cancel_current()

        await asyncio.wait_for(art_task, 2.0)
        assert put == []
    finally:
        art_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await art_task
