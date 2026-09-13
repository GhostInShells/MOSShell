"""Tests for the mindflow channel — 反身控制面 (reflexive mindflow control surface).

Covers ``build_mindflow_channel``:
- 常驻读面: ``status`` (always_observe 自省) 与 ``pull`` (主动 reach, 默认关) 留在顶层.
- 注意力治理面下移为 gated 虚拟子通道 ``attention``: ``set-priority`` /
  ``set-signal-bar`` / ``set-impulse-bar`` 不进顶层命令, 经子通道执行.
- nucleus channel 作为虚拟子通道; gate 开启时默认关闭, 由 mount_child 披露.
- instruction 是静态心智模型且不重复罗列命令; notice 列出 nucleus 目录.
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


def _context_text(runtime) -> str:
    return " | ".join(m.to_content_string() for m in runtime.self_meta().context)


# ── flag gating ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_attention_governance_lives_in_child():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        names = _command_names(runtime)
        assert "status" in names
        assert "pull" not in names  # default off
        # 治理面 (setter) 下移到 attention 子通道, 不在顶层.
        assert "set-priority" not in names
        assert "set-signal-bar" not in names
        assert "set-impulse-bar" not in names
        assert "attention" in runtime.virtual_sub_channels()


@pytest.mark.asyncio
async def test_pull_gated_by_flag():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf, enable_pull=True)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert "pull" in _command_names(runtime)

    chan_off = build_mindflow_channel(mf, enable_pull=False)
    async with chan_off.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert "pull" not in _command_names(runtime)


@pytest.mark.asyncio
async def test_attention_child_respects_priority_flag():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf, enable_priority=False)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        child = runtime.fetch_sub_runtime("attention")
        assert child is not None  # enable_bar 仍开, child 仍在
        child_names = {c.name for c in child.self_meta().commands}
        assert "set-priority" not in child_names
        assert "set-signal-bar" in child_names


# ── status / pull ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_status_reports_nucleus_and_peek():
    nuc = CachedNucleus()
    mf = BaseMindflow(nuc)
    nuc.set_impulse(_impulse())
    chan = build_mindflow_channel(mf)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("status")
        assert NUCLEUS_NAME in result
        assert "test cache nucleus" in result
        assert "hello" in result


@pytest.mark.asyncio
async def test_pull_consumes_impulse_and_returns_messages():
    nuc = CachedNucleus()
    mf = BaseMindflow(nuc)
    nuc.set_impulse(_impulse())
    chan = build_mindflow_channel(mf, enable_pull=True)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("pull", kwargs={"nucleus": NUCLEUS_NAME})
        assert "hello" in result
        assert nuc.peek() is None  # consumed via attended


@pytest.mark.asyncio
async def test_pull_on_empty_nucleus_is_clean_noop():
    nuc = CachedNucleus()
    mf = BaseMindflow(nuc)
    chan = build_mindflow_channel(mf, enable_pull=True)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("pull", kwargs={"nucleus": NUCLEUS_NAME})
        assert "nothing" in result.lower()


# ── priority bars (observable via context when enable_bar) ─────


@pytest.mark.asyncio
async def test_set_impulse_bar_reflected_in_context():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf, enable_bar=True)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        initial = _context_text(runtime)
        assert "impulse bar" in initial.lower()

        child = runtime.fetch_sub_runtime("attention")
        await child.execute_command("set-impulse-bar", kwargs={"priority": "CRITICAL"})
        await runtime.refresh_metas()
        assert "CRITICAL" in _context_text(runtime)


@pytest.mark.asyncio
async def test_set_signal_bar_reflected_in_context():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf, enable_bar=True)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        child = runtime.fetch_sub_runtime("attention")
        await child.execute_command("set-signal-bar", kwargs={"priority": "WARNING"})
        await runtime.refresh_metas()
        assert "WARNING" in _context_text(runtime)


@pytest.mark.asyncio
async def test_set_priority_operates_on_current_attention():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf, enable_priority=True)
    async with mf:
        mf.set_impulse(_impulse(priority=Priority.FATAL))
        await _wait_until(lambda: mf.attention() is not None)
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            child = runtime.fetch_sub_runtime("attention")
            result = await child.execute_command("set-priority", kwargs={"priority": "CRITICAL"})
            assert "CRITICAL" in result
            assert mf.attention().priority() == Priority.CRITICAL


@pytest.mark.asyncio
async def test_set_priority_without_attention_is_noop():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf, enable_priority=True)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        child = runtime.fetch_sub_runtime("attention")
        result = await child.execute_command("set-priority", kwargs={"priority": "CRITICAL"})
        assert "no active attention" in result.lower()


# ── instruction / help ────────────────────────────────────────


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
async def test_help_lists_nuclei():
    nuc = CachedNucleus()
    mf = BaseMindflow(nuc)
    chan = build_mindflow_channel(mf)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert NUCLEUS_NAME in runtime.self_meta().notice


async def _wait_until(cond, *, timeout: float = 1.0) -> None:
    deadline = asyncio.get_event_loop().time() + timeout
    while not cond():
        assert asyncio.get_event_loop().time() < deadline, "condition not met before timeout"
        await asyncio.sleep(0.01)


# ── gate: nucleus channel 渐进式披露 ──────────────────────────


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
async def test_default_mindflow_mounts_nucleus_child_directly():
    nuc = ListenerNucleus()
    mf = BaseMindflow(nuc)  # gate 默认 False
    async with mf:
        channel = mf.as_channel()
        async with channel.bootstrap() as runtime:
            await runtime.refresh_metas()
            assert "listener_nucleus" in runtime.virtual_sub_channels()
