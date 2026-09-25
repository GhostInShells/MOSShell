"""PauseController 级联测试 — 注入真 mindflow + shell, 验证级联效果.

测试纪律:
    - 不测内部调用 (不 mock/monkeypatch), 只测公开 API 效果.
    - mindflow: pause 后 add_signal 不产出 attention.
    - shell: pause 后 interpreter() 抛出 PausedError.
"""

import asyncio
import contextlib

import pytest
from ghoshell_moss.core.blueprint.mindflow import Signal, Priority, Attention, Thinking
from ghoshell_moss.core.concepts.errors import PausedError
from ghoshell_moss.core.ctml.shell import new_ctml_shell
from ghoshell_moss.core.mindflow import BaseMindflow
from ghoshell_moss.core.mindflow.buffer_nucleus import BufferNucleus
from ghoshell_moss.host.pause_controller import PauseController


def _make_mindflow() -> BaseMindflow:
    from ghoshell_moss.contracts.logger import get_console_logger
    return BaseMindflow(logger=get_console_logger())


def _make_nucleus() -> BufferNucleus:
    return BufferNucleus(
        name="sensor",
        description="test",
        target_signal="test_event",
        suppress_seconds=0.01,
    )


# ── 级联: 真 mindflow + shell ──────────────────────


@pytest.mark.asyncio
async def test_pause_cascades_to_mindflow():
    """pause(True) 后 mindflow 拒绝 signal — loop 不产出 attention."""
    mindflow = _make_mindflow()
    mindflow.with_nucleus(_make_nucleus())

    async with mindflow:
        await mindflow.wait_started()
        ctrl = PauseController(mindflow=mindflow)

        ctrl.pause(True)
        mindflow.add_signal(Signal.new(name="test_event", priority=Priority.NOTICE))

        # loop 不应产出 attention
        got: list[Thinking] = []

        async def _consume():
            async for thinking in mindflow.thinking_loop():
                got.append(thinking)
                return

        task = asyncio.create_task(_consume())
        await asyncio.sleep(0.3)
        assert len(got) == 0
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_pause_cascades_to_shell():
    """pause(True) 后 shell 拒绝 interpreter — 抛出 PausedError."""
    shell = new_ctml_shell()
    async with shell:
        ctrl = PauseController(shell=shell)
        ctrl.pause(True)

        with pytest.raises(PausedError):
            await shell.interpreter()


@pytest.mark.asyncio
async def test_pause_cascades_to_both():
    """pause(True) 同时级联 mindflow + shell."""
    mindflow = _make_mindflow()
    mindflow.with_nucleus(_make_nucleus())
    shell = new_ctml_shell()

    async with mindflow:
        await mindflow.wait_started()
        async with shell:
            ctrl = PauseController(mindflow=mindflow, shell=shell)
            ctrl.pause(True)

            # 两侧都生效
            with pytest.raises(PausedError):
                await shell.interpreter()

            mindflow.add_signal(Signal.new(name="test_event", priority=Priority.NOTICE))
            got: list[Attention] = []

            async def _consume():
                async for thinking in mindflow.thinking_loop():
                    got.append(thinking.attention)
                    return

            task = asyncio.create_task(_consume())
            await asyncio.sleep(0.3)
            assert len(got) == 0
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task


# ── 幂等 ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_pause_idempotent_does_not_double_cascade():
    """pause(True) 两次 — 返回 False, shell 仍然 paused, 不崩溃."""
    shell = new_ctml_shell()
    async with shell:
        ctrl = PauseController(shell=shell)

        changed = ctrl.pause(True)
        assert changed

        changed = ctrl.pause(True)
        assert not changed

        with pytest.raises(PausedError):
            await shell.interpreter()


# ── resume ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_pause_resume():
    """pause(False) 恢复 — shell 重新接受 interpreter."""
    shell = new_ctml_shell()
    async with shell:
        ctrl = PauseController(shell=shell)

        ctrl.pause(True)
        with pytest.raises(PausedError):
            await shell.interpreter()

        ctrl.pause(False)
        # 恢复后应能创建 interpreter
        interp = await shell.interpreter()
        assert interp is not None


# ── bind ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_bind_after_construction():
    """通过 bind() 延迟注入 mindflow + shell, 然后 pause 生效."""
    shell = new_ctml_shell()
    mindflow = _make_mindflow()
    async with shell:
        ctrl = PauseController()
        ctrl.bind(mindflow=mindflow, shell=shell)

        ctrl.pause(True)
        with pytest.raises(PausedError):
            await shell.interpreter()


# ── 无 mindflow/shell ───────────────────────────────


def test_pause_without_mindflow_or_shell_no_crash():
    """没有 mindflow/shell 时 pause(True) 只设状态, 不崩溃."""
    ctrl = PauseController()
    changed = ctrl.pause(True)
    assert changed
    assert ctrl.is_paused()
