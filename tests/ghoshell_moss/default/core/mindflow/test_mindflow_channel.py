"""Tests for the mindflow channel — 反身控制面 (reflexive mindflow control surface).

Covers ``build_mindflow_channel``:
- 常驻能力面直接挂在父 channel: ``set-priority`` / ``set-signal-bar`` /
  ``set-impulse-bar`` (注意力治理) + ``status`` / ``nuclei`` (自省).
- 治理状态是温数据, 走 ``notice``, 不进每帧 ``context_messages``.
- nucleus 讯息收敛进 ``nuclei`` 命令, 不在 notice 里重复罗列.
- gate 只折叠各 nucleus 的子通道 (默认开启), mindflow 自身的控制面不折叠.
"""

from __future__ import annotations

import asyncio

import pytest

from ghoshell_moss.core.blueprint.mindflow import Impulse, Priority
from ghoshell_moss.message import Message
from ghoshell_moss.core.mindflow import BaseMindflow, DirectImpulseNucleus
from ghoshell_moss.core.mindflow._channel import build_mindflow_channel
from ghoshell_moss.core.mindflow.listener_nucleus import ListenerNucleus

NUCLEUS_NAME = "cached"


class CachedNucleus(DirectImpulseNucleus):
    """A DirectImpulseNucleus with a distinct name so tests keep a handle on it."""

    NAME = NUCLEUS_NAME

    def description(self) -> str:
        return "test cache nucleus"


def _impulse(*, priority: Priority = Priority.NOTICE, text: str = "hello") -> Impulse:
    return Impulse(
        source=NUCLEUS_NAME,
        priority=priority,
        messages=[Message.new().with_content(text)],
    )


def _command_names(runtime) -> set[str]:
    return {c.name for c in runtime.self_meta().commands}


def _notice_text(runtime) -> str:
    return runtime.self_meta().notice or ""


# ── 常驻能力面 ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_attention_governance_is_resident_on_parent():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        names = _command_names(runtime)
        # 注意力治理是常驻一等能力, 直接挂在父 channel.
        assert "set-priority" in names
        assert "set-signal-bar" in names
        assert "set-impulse-bar" in names
        assert "status" in names
        assert "nuclei" in names
        # 没有 attention 子通道.
        assert "attention" not in runtime.virtual_sub_channels()


@pytest.mark.asyncio
async def test_set_priority_gated_by_flag():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf, enable_priority=False)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        names = _command_names(runtime)
        assert "set-priority" not in names
        assert "set-signal-bar" in names  # enable_bar 仍开


# ── 自省面: status / nuclei ───────────────────────────────────


@pytest.mark.asyncio
async def test_nuclei_reports_nucleus_and_peek():
    nuc = CachedNucleus()
    mf = BaseMindflow(nuc)
    nuc.set_impulse(_impulse())
    chan = build_mindflow_channel(mf)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("nuclei")
        assert NUCLEUS_NAME in result
        assert "test cache nucleus" in result
        assert "hello" in result


@pytest.mark.asyncio
async def test_status_reports_current_attention():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf)
    async with mf:
        mf.set_impulse(_impulse(priority=Priority.FATAL))
        await _wait_until(lambda: mf.attention() is not None)
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("status")
            assert "active attention" in result


# ── 治理状态走 notice (温数据), 不进 context ────────────────────


@pytest.mark.asyncio
async def test_set_impulse_bar_reflected_in_notice():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf, enable_bar=True)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert "impulse bar" in _notice_text(runtime).lower()
        assert not runtime.self_meta().context  # 温数据不进每帧 context

        await runtime.execute_command("set-impulse-bar", kwargs={"priority": "CRITICAL"})
        await runtime.refresh_metas()
        assert "CRITICAL" in _notice_text(runtime)


@pytest.mark.asyncio
async def test_set_signal_bar_reflected_in_notice():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf, enable_bar=True)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        await runtime.execute_command("set-signal-bar", kwargs={"priority": "WARNING"})
        await runtime.refresh_metas()
        assert "WARNING" in _notice_text(runtime)


@pytest.mark.asyncio
async def test_set_priority_operates_on_current_attention():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf, enable_priority=True)
    async with mf:
        mf.set_impulse(_impulse(priority=Priority.FATAL))
        await _wait_until(lambda: mf.attention() is not None)
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("set-priority", kwargs={"priority": "CRITICAL"})
            assert "CRITICAL" in result
            assert mf.attention().priority() == Priority.CRITICAL


@pytest.mark.asyncio
async def test_set_priority_without_attention_is_noop():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf, enable_priority=True)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("set-priority", kwargs={"priority": "CRITICAL"})
        assert "no active attention" in result.lower()


# ── instruction / nuclei 不重复罗列 ────────────────────────────


@pytest.mark.asyncio
async def test_instruction_explains_but_does_not_list_commands():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        instruction = runtime.self_meta().instruction
        assert "mindflow" in instruction.lower()
        # red line: instruction 不重复罗列命令
        assert "set-priority(" not in instruction


@pytest.mark.asyncio
async def test_nuclei_not_listed_in_notice():
    nuc = CachedNucleus()
    mf = BaseMindflow(nuc)
    chan = build_mindflow_channel(mf)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        # nucleus 讯息收敛进 `nuclei` 命令, notice 不再罗列 nucleus 目录.
        assert NUCLEUS_NAME not in runtime.self_meta().notice
        result = await runtime.execute_command("nuclei")
        assert NUCLEUS_NAME in result


async def _wait_until(cond, *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not cond():
        assert asyncio.get_event_loop().time() < deadline, "condition not met before timeout"
        await asyncio.sleep(0.01)


# ── gate: 只折叠 nucleus 子通道 ────────────────────────────────


@pytest.mark.asyncio
async def test_gated_mindflow_lists_and_mounts_nucleus_child():
    nuc = ListenerNucleus()
    mf = BaseMindflow(nuc, gate=True)
    async with mf:
        channel = mf.as_channel()
        async with channel.bootstrap() as runtime:
            await runtime.refresh_metas()
            # gate 开启: nucleus channel 默认关闭, 只在 notice 的 gate 目录里可见.
            notice = runtime.self_meta().notice
            assert "gated children" in notice
            assert "listener_nucleus (closed)" in notice
            assert "listener_nucleus" not in runtime.virtual_sub_channels()

            result = await runtime.mount_child("listener_nucleus")
            assert "mounted" in result
            assert "listener_nucleus" in runtime.virtual_sub_channels()
            assert "listener_nucleus (open)" in runtime.self_meta().notice


@pytest.mark.asyncio
async def test_mindflow_defaults_to_gate_on():
    nuc = ListenerNucleus()
    mf = BaseMindflow(nuc)  # gate 默认开启
    async with mf:
        channel = mf.as_channel()
        async with channel.bootstrap() as runtime:
            await runtime.refresh_metas()
            # nucleus 子通道默认关闭, 需 mount_child 披露.
            assert "listener_nucleus" not in runtime.virtual_sub_channels()
            assert "gated children" in runtime.self_meta().notice
