"""Mindflow pause() 单测 — 与已有测试一致的 pattern: BufferNucleus + 直接 async for."""

import asyncio
import contextlib

import pytest
from ghoshell_moss.core.blueprint.mindflow import (
    MindflowHook, Signal, Priority, Impulse, Attention, ChallengeVerdict,
)
from ghoshell_moss.core.mindflow.buffer_nucleus import BufferNucleus
from ghoshell_moss.core.mindflow.base_mindflow import BaseMindflow


def make_base_mindflow() -> BaseMindflow:
    from ghoshell_moss.contracts.logger import get_console_logger
    return BaseMindflow(logger=get_console_logger())


def _make_nucleus(name: str = "test_sensor", target_signal: str = "test_event") -> BufferNucleus:
    return BufferNucleus(
        name=name,
        description="test sensor",
        target_signal=target_signal,
        suppress_seconds=0.01,
    )


# ── signal: pause 后 signal 不产生 attention ──────


@pytest.mark.asyncio
async def test_pause_prevents_signal_from_producing_attention():
    """pause(True) 后 add_signal 不通过 loop 产出 attention.

    这里必须用 task + 短超时: loop 在 pause 时不会 yield,
    async for 会死等.
    """
    mindflow = make_base_mindflow()
    mindflow.with_nucleus(_make_nucleus())

    async with mindflow:
        await mindflow.wait_started()
        mindflow.pause(True)
        mindflow.add_signal(Signal.new(name="test_event", priority=Priority.NOTICE))

        got: list[Attention] = []

        async def _consume():
            async for att in mindflow.loop():
                got.append(att)
                return

        task = asyncio.create_task(_consume())
        await asyncio.sleep(0.3)
        assert len(got) == 0
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_pause_resume_signal_produces_attention():
    """pause(False) 后 signal 恢复正常, loop 产出 attention."""
    mindflow = make_base_mindflow()
    mindflow.with_nucleus(_make_nucleus())

    async with mindflow:
        await mindflow.wait_started()
        mindflow.pause(True)
        mindflow.pause(False)

        mindflow.add_signal(Signal.new(name="test_event", priority=Priority.NOTICE))

        async for attention in mindflow.loop():
            async with attention:
                impulse = attention.draw_from()
                assert impulse.source == "test_sensor"
                break


# ── impulse: pause 后 add_impulse 不产生 attention ──


@pytest.mark.asyncio
async def test_pause_prevents_add_impulse_from_producing_attention():
    """pause(True) 后 add_impulse() 返回 None, attention() 为 None."""
    mindflow = make_base_mindflow()

    async with mindflow:
        await mindflow.wait_started()
        mindflow.pause(True)

        imp = Impulse(source="test", priority=Priority.NOTICE)
        result = mindflow.add_impulse(imp)
        assert result is None
        assert mindflow.attention() is None


# ── attention: pause 中断当前 attention ────────────


@pytest.mark.asyncio
async def test_pause_aborts_current_attention():
    """pause(True) 中断正在运行的 attention, is_aborted() → True."""
    mindflow = make_base_mindflow()
    mindflow.with_nucleus(_make_nucleus())

    async with mindflow:
        await mindflow.wait_started()
        mindflow.add_signal(Signal.new(name="test_event", priority=Priority.NOTICE))

        async for attention in mindflow.loop():
            mindflow.pause(True)
            assert attention.is_aborted()
            async with attention:
                pass
            break


# ── system command: set_impulse 在 pause 期间仍生效 ─


@pytest.mark.asyncio
async def test_set_impulse_still_works_during_pause():
    """pause 期间 set_impulse() 作为系统指令仍然创建 attention."""
    mindflow = make_base_mindflow()

    async with mindflow:
        await mindflow.wait_started()
        mindflow.pause(True)

        imp = Impulse(source="system_cmd", priority=Priority.FATAL)
        mindflow.set_impulse(imp)

        # set_impulse 内部 create_task, 让出 event loop
        await asyncio.sleep(0)
        att = mindflow.attention()
        assert att is not None
        assert not att.is_aborted()


# ── hook: impulse challenge 在 pause 期间不触发 ────


class _ChallengeRecorder(MindflowHook):
    def __init__(self):
        self.calls: list[tuple[str | None, str | None, ChallengeVerdict]] = []

    def on_impulse_challenged(
            self,
            challenger: Impulse,
            defender: Impulse | None,
            verdict: ChallengeVerdict,
    ) -> None:
        ch_src = challenger.source if challenger else None
        df_src = defender.source if defender else None
        self.calls.append((ch_src, df_src, verdict))


@pytest.mark.asyncio
async def test_pause_no_impulse_challenge_hook_fires():
    """pause 后 signal → nucleus 不触发 impulse challenge hook."""
    mindflow = make_base_mindflow()
    mindflow.with_nucleus(_make_nucleus())
    recorder = _ChallengeRecorder()
    mindflow.with_hook(recorder)

    async with mindflow:
        await mindflow.wait_started()
        mindflow.pause(True)

        mindflow.add_signal(Signal.new(name="test_event", priority=Priority.NOTICE))
        await asyncio.sleep(0)
        assert len(recorder.calls) == 0


@pytest.mark.asyncio
async def test_resume_allows_impulse_challenge():
    """resume 后 signal 正常触发 challenge hook."""
    mindflow = make_base_mindflow()
    mindflow.with_nucleus(_make_nucleus())
    recorder = _ChallengeRecorder()
    mindflow.with_hook(recorder)

    async with mindflow:
        await mindflow.wait_started()
        mindflow.pause(True)
        mindflow.pause(False)

        mindflow.add_signal(Signal.new(name="test_event", priority=Priority.NOTICE))
        async for attention in mindflow.loop():
            async with attention:
                break

        assert len(recorder.calls) > 0


# ── lifecycle ──────────────────────────────────────


@pytest.mark.asyncio
async def test_pause_idempotent():
    """pause(True) 多次调用不报错, 不崩溃."""
    mindflow = make_base_mindflow()

    async with mindflow:
        await mindflow.wait_started()
        mindflow.pause(True)
        mindflow.pause(True)
        # 不报错即通过


@pytest.mark.asyncio
async def test_pause_not_running_is_noop():
    """未启动时 pause() 不报错."""
    mindflow = make_base_mindflow()
    mindflow.pause(True)
    assert not mindflow.is_running()