"""Tests for the mindflow channel — 反身控制面 (reflexive mindflow control surface).

Covers ``build_mindflow_channel``:
- 常驻命令面: ``set-priority`` / ``set-signal-bar`` / ``set-impulse-bar`` (注意力治理)
  + ``nuclei`` / ``peek`` / ``claim`` / ``specification`` (感知读面).
- 无 instruction / notice / context: 注意力状态自解释, 不常驻展开; 需要语义时走
  ``specification`` 拉 blueprint 源码.
- ``set-priority`` 只在 attention 活跃时可见 (flag AND 运行时谓词).
- ``peek`` / ``claim`` 不把 message 内容带出 nucleus (内容只经 ``claim`` 进下一帧).
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


# ── 常驻命令面 ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_governance_and_perception_resident_on_parent():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        names = _command_names(runtime)
        # 注意力治理是常驻一等能力.
        assert "set-signal-bar" in names
        assert "set-impulse-bar" in names
        # 感知读面.
        assert "nuclei" in names
        assert "peek" in names
        assert "claim" in names
        assert "specification" in names
        # set-priority 只在 attention 活跃时可见 (无 attention → 不可见).
        assert "set-priority" not in names
        # 无 status / 无 attention 子通道.
        assert "status" not in names
        assert "attention" not in runtime.virtual_sub_channels()


@pytest.mark.asyncio
async def test_set_priority_visible_only_with_active_attention():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf, enable_priority=True)
    async with mf:
        mf.set_impulse(_impulse(priority=Priority.FATAL))
        await _wait_until(lambda: mf.attention() is not None)
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            assert "set-priority" in _command_names(runtime)


@pytest.mark.asyncio
async def test_set_priority_gated_by_flag():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf, enable_priority=False)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert "set-priority" not in _command_names(runtime)
        assert "set-signal-bar" in _command_names(runtime)  # enable_bar 仍开


# ── 无 instruction / notice / context ─────────────────────────


@pytest.mark.asyncio
async def test_no_instruction_notice_or_context():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        assert not runtime.self_meta().instruction
        assert not runtime.self_meta().notice
        assert not runtime.self_meta().context


# ── 感知读面: nuclei / peek / claim / specification ───────────


@pytest.mark.asyncio
async def test_nuclei_reports_topology_not_content():
    nuc = CachedNucleus()
    mf = BaseMindflow(nuc)
    nuc.set_impulse(_impulse(text="secret_payload"))
    chan = build_mindflow_channel(mf)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("nuclei")
        assert NUCLEUS_NAME in result
        assert "test cache nucleus" in result
        assert "secret_payload" not in result  # 拓扑, 不带 message 内容


@pytest.mark.asyncio
async def test_peek_reports_held_units_with_state_and_preview():
    nuc = CachedNucleus()
    mf = BaseMindflow(nuc)
    nuc.set_impulse(_impulse(text="secret_payload"))
    chan = build_mindflow_channel(mf)
    async with nuc:  # nucleus running, mindflow idle → impulse 不被 rank 消费.
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("peek")
            assert NUCLEUS_NAME in result
            assert "NOTICE" in result
            assert "strength=" in result
            assert "age=" in result
            assert "expires=" in result
            # message 只给 head 预览, 短消息原样出现.
            assert "secret_payload" in result


@pytest.mark.asyncio
async def test_peek_truncates_long_message_head():
    nuc = CachedNucleus()
    mf = BaseMindflow(nuc)
    long_text = "START_" + "m" * 200 + "_END"
    nuc.set_impulse(_impulse(text=long_text))
    chan = build_mindflow_channel(mf)
    async with nuc:
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("peek")
            assert long_text[:40] + "…" in result
            assert "_END" not in result  # 尾部被截断


@pytest.mark.asyncio
async def test_peek_empty_reports_nothing():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("peek")
        assert "nothing" in result.lower()


@pytest.mark.asyncio
async def test_claim_consumes_and_forces_observe():
    nuc = CachedNucleus()
    mf = BaseMindflow(nuc)
    nuc.set_impulse(_impulse(text="payload"))
    chan = build_mindflow_channel(mf)
    async with nuc:  # nucleus running, mindflow idle → impulse 不会被 rank 消费.
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("claim", kwargs={"nucleus": NUCLEUS_NAME})
            assert "claimed" in result
            assert nuc.peek() is None  # 消费 (attended)
            assert mf.moments.need_observe()  # 强制下一轮观察


@pytest.mark.asyncio
async def test_claim_nothing_held():
    nuc = CachedNucleus()
    mf = BaseMindflow(nuc)
    chan = build_mindflow_channel(mf)
    async with nuc:
        async with chan.bootstrap() as runtime:
            await runtime.refresh_metas()
            result = await runtime.execute_command("claim", kwargs={"nucleus": NUCLEUS_NAME})
            assert "nothing to claim" in result


@pytest.mark.asyncio
async def test_claim_no_such_nucleus():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("claim", kwargs={"nucleus": "missing"})
        assert "no nucleus" in result


@pytest.mark.asyncio
async def test_specification_returns_module_path():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("specification")
        assert "ghoshell_moss.core.blueprint.mindflow" in result


# ── 注意力治理 ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_impulse_bar_command():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf, enable_bar=True)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("set-impulse-bar", kwargs={"priority": "CRITICAL"})
        assert "CRITICAL" in result
        assert mf.impulse_priority_bar() == Priority.CRITICAL


@pytest.mark.asyncio
async def test_set_signal_bar_command():
    mf = BaseMindflow()
    chan = build_mindflow_channel(mf, enable_bar=True)
    async with chan.bootstrap() as runtime:
        await runtime.refresh_metas()
        result = await runtime.execute_command("set-signal-bar", kwargs={"priority": "WARNING"})
        assert "WARNING" in result
        assert mf.signal_priority_bar() == Priority.WARNING


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
            catalog = runtime.self_meta().named_notices["gated_children"]
            assert "listener_nucleus (closed)" in catalog
            assert "listener_nucleus" not in runtime.virtual_sub_channels()

            result = await runtime.mount_child("listener_nucleus")
            assert "mounted" in result
            assert "listener_nucleus" in runtime.virtual_sub_channels()
            assert "listener_nucleus (open)" in runtime.self_meta().named_notices["gated_children"]


@pytest.mark.asyncio
async def test_mindflow_defaults_to_gate_on():
    nuc = ListenerNucleus()
    mf = BaseMindflow(nuc)  # gate 默认开启
    async with mf:
        channel = mf.as_channel()
        async with channel.bootstrap() as runtime:
            await runtime.refresh_metas()
            assert "listener_nucleus" not in runtime.virtual_sub_channels()
            assert "gated_children" in runtime.self_meta().named_notices
