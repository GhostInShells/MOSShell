"""ContextMonitor 单元测试.

纯函数注入 metas, 零 shell 依赖, 全部可构造.
测试覆盖 V2.8 单测清单:
  投影确定性 / 无变更无delta / 字段变更delta / ADD/REMOVE
  / ACK 纪律 / 降级三态机 / hot 层渲染.
"""

import pytest

from ghoshell_moss.host.context_monitor import (
    ContextMonitor,
    ContextSnapshot,
    WarmDelta,
    WarmUnit,
)
from ghoshell_moss.core.concepts.channel import ChannelMeta
from ghoshell_moss.core.concepts.command import CommandMeta
from ghoshell_moss.message import Message


# ── helpers ─────────────────────────────────────────────


def _cmd(name: str, interface: str = "", **kw) -> CommandMeta:
    return CommandMeta(name=name, interface=interface, **kw)


def _meta(name: str = "test", **kw) -> ChannelMeta:
    return ChannelMeta(name=name, **kw)


def _snap_deltas(m: ContextMonitor, metas: dict) -> tuple[WarmDelta, ...]:
    snap = m.snapshot(metas)
    m.ack(snap)
    return snap.warm_deltas


def _snap_unacked(m: ContextMonitor, metas: dict) -> ContextSnapshot:
    return m.snapshot(metas)


# ── 投影确定性 ───────────────────────────────────────────


class TestProjectionDeterministic:
    def test_same_metas_no_deltas_after_ack(self):
        m = ContextMonitor()
        meta = _meta(description="d", commands=[_cmd("hello", "async def hello() -> None")])
        s1 = m.snapshot({"a.b": meta})
        assert len(s1.warm_deltas) == 1
        assert s1.warm_deltas[0].kind == "add"
        m.ack(s1)

        s2 = m.snapshot({"a.b": meta})
        assert len(s2.warm_deltas) == 0

    def test_render_deterministic_across_monitors(self):
        meta = _meta(description="d", commands=[_cmd("x"), _cmd("a")])
        s1 = ContextMonitor().snapshot({"x": meta})
        s2 = ContextMonitor().snapshot({"x": meta})
        assert s1.warm_deltas[0].block == s2.warm_deltas[0].block

    def test_hot_layer_in_snapshot(self):
        """hot 层渲染 failure 和 context, 每帧新鲜."""
        m = ContextMonitor()
        meta = _meta(failure="broke")
        s = m.snapshot({"ch": meta})
        assert len(s.hot_messages) == 1
        assert "broke" in s.hot_messages[0].to_content_string()
        assert "failure" in s.hot_messages[0].to_content_string()
        # warm 端不受影响.
        assert len(s.warm_deltas) == 0

    def test_hot_includes_context(self):
        m = ContextMonitor()
        meta = _meta(context=[Message.new(tag="").with_content("<context>data</context>")])
        s = m.snapshot({"ch": meta})
        assert len(s.hot_messages) == 1
        assert "data" in s.hot_messages[0].to_content_string()

    def test_hot_sorted_by_path(self):
        m = ContextMonitor()
        msgs = m.snapshot({"z": _meta(failure="z"), "a": _meta(failure="a")}).hot_messages
        texts = [m.to_content_string() for m in msgs]
        a_pos = next(i for i, t in enumerate(texts) if "channel" in t and "a" in t)
        z_pos = next(i for i, t in enumerate(texts) if "channel" in t and "z" in t)
        assert a_pos < z_pos


# ── 无变更无 delta ──────────────────────────────────────


class TestNoChangeNoDelta:
    def test_acked_stable_round_produces_nothing(self):
        m = ContextMonitor()
        meta = _meta(description="static")
        assert len(_snap_deltas(m, {"ch": meta})) == 1  # ADD
        assert len(_snap_deltas(m, {"ch": meta})) == 0  # stable

    def test_two_stable_rounds_both_empty(self):
        m = ContextMonitor()
        meta = _meta(description="static")
        _snap_deltas(m, {"ch": meta})
        assert len(_snap_deltas(m, {"ch": meta})) == 0
        assert len(_snap_deltas(m, {"ch": meta})) == 0


# ── 字段变更 delta ──────────────────────────────────────


class TestFieldChangeDelta:
    def test_interface_change_emits_update(self):
        m = ContextMonitor()
        _snap_deltas(m, {"ch": _meta(commands=[_cmd("hello", "async def hello() -> None")])})
        deltas = _snap_deltas(m, {"ch": _meta(commands=[_cmd("hello", "async def hello(name: str) -> None")])})
        assert len(deltas) == 1
        assert deltas[0].kind == "update"
        assert deltas[0].unit == WarmUnit.INTERFACE
        assert 'hello(name: str)' in deltas[0].block

    def test_states_change_emits_update(self):
        m = ContextMonitor()
        _snap_deltas(m, {"ch": _meta(states={"idle": "idle state"})})
        deltas = _snap_deltas(m, {"ch": _meta(states={"idle": "idle state"}, current_state="idle")})
        assert len(deltas) == 1
        assert deltas[0].unit == WarmUnit.STATES
        assert "Current state: idle" in deltas[0].block

    def test_description_change_emits_update(self):
        m = ContextMonitor()
        _snap_deltas(m, {"ch": _meta(description="old")})
        deltas = _snap_deltas(m, {"ch": _meta(description="new")})
        assert len(deltas) == 1
        assert deltas[0].unit == WarmUnit.DESC_INSTRUCTION

    def test_unit_absent_to_present_emits_add(self):
        m = ContextMonitor()
        _snap_deltas(m, {"ch": _meta()})  # 无 warm → 不发射
        deltas = _snap_deltas(m, {"ch": _meta(description="now exists")})
        assert len(deltas) == 1
        assert deltas[0].kind == "add"

    def test_unit_present_to_absent_emits_cleared_marker(self):
        m = ContextMonitor()
        _snap_deltas(m, {"ch": _meta(description="exists")})
        deltas = _snap_deltas(m, {"ch": _meta()})
        assert len(deltas) == 1
        assert deltas[0].unit == WarmUnit.DESC_INSTRUCTION
        assert any(tag in deltas[0].block for tag in ("<description/>", "<instruction/>"))

    def test_failure_does_not_trigger_warm_delta(self):
        m = ContextMonitor()
        _snap_deltas(m, {"ch": _meta(failure="broke")})
        assert len(_snap_deltas(m, {"ch": _meta(failure="broke")})) == 0
        # failure 变更 → 不进 warm delta (在 hot 层).
        s = m.snapshot({"ch": _meta(failure="recovered")})
        assert len(s.warm_deltas) == 0
        # hot 有渲染.
        assert any("recovered" in m.to_content_string() for m in s.hot_messages)

    def test_context_does_not_trigger_warm_delta(self):
        m = ContextMonitor()
        _snap_deltas(m, {"ch": _meta()})
        s = m.snapshot({"ch": _meta(context=[Message.new(tag="").with_content("fresh")])})
        assert len(s.warm_deltas) == 0


# ── ADD / REMOVE ────────────────────────────────────────


class TestAddRemove:
    def test_new_channel_emits_add_with_full_block(self):
        meta = _meta(description="d", instruction="i",
                     commands=[_cmd("hello", "async def hello()")],
                     states={"open": "open state"})
        deltas = _snap_deltas(ContextMonitor(), {"ch": meta})
        assert len(deltas) == 1
        d = deltas[0]
        assert d.kind == "add"
        assert d.path == "ch"
        assert d.unit is None
        assert "d" in d.block and "i" in d.block
        assert "async def hello()" in d.block and "open state" in d.block

    def test_empty_warm_channel_skips_add(self):
        s = ContextMonitor().snapshot({"ghost": _meta()})
        assert len(s.warm_deltas) == 0

    def test_remove_emits_tombstone(self):
        m = ContextMonitor()
        _snap_deltas(m, {"ch": _meta(description="d")})
        deltas = _snap_deltas(m, {})
        assert len(deltas) == 1
        assert deltas[0].kind == "remove"
        assert deltas[0].path == "ch"

    def test_remove_non_existent_is_nothing(self):
        assert len(ContextMonitor().snapshot({}).warm_deltas) == 0

    def test_re_add_after_remove(self):
        m = ContextMonitor()
        meta = _meta(description="d")
        _snap_deltas(m, {"ch": meta})
        _snap_deltas(m, {})
        d = _snap_deltas(m, {"ch": meta})
        assert len(d) == 1
        assert d[0].kind == "add"


# ── ACK 纪律 ─────────────────────────────────────────────


class TestAckDiscipline:
    def test_replay_without_ack(self):
        """ACK 前 delta 可重放: 两次 snapshot 不 ack, 各自独立冻帧."""
        m = ContextMonitor()
        meta = _meta(description="d")
        s1 = m.snapshot({"ch": meta})
        s2 = m.snapshot({"ch": meta})
        assert s1.warm_deltas[0].block == s2.warm_deltas[0].block

    def test_ack_advances_baseline(self):
        m = ContextMonitor()
        meta = _meta(description="d")
        s1 = m.snapshot({"ch": meta})
        m.ack(s1)
        s2 = m.snapshot({"ch": meta})
        assert len(s2.warm_deltas) == 0

    def test_double_ack_is_idempotent(self):
        """同一快照 ack 两次, 第二次 no-op."""
        m = ContextMonitor()
        s1 = m.snapshot({"ch": _meta(description="d")})
        m.ack(s1)
        m.ack(s1)  # no-op
        assert len(m.snapshot({"ch": _meta(description="d")}).warm_deltas) == 0

    def test_old_snapshot_ack_after_newer_is_superseded(self):
        """新快照已 ack 后, 旧快照 ack no-op."""
        m = ContextMonitor()
        meta = _meta(description="d")
        s1 = m.snapshot({"ch": meta})
        s2 = m.snapshot({"ch": _meta(description="d2")})
        m.ack(s2)  # ack 新帧
        # s1 现在是过时的, ack 它应该 no-op.
        old_baseline = m.baseline
        m.ack(s1)
        # baseline 保持 s2 的.
        assert m.baseline == old_baseline

    def test_ack_only_advances_emitted_units(self):
        """ACK 只推进实际发射的单元, 降级后未发射的版本不进基线."""
        m = ContextMonitor(demote_threshold=1)
        _snap_deltas(m, {"ch": _meta(description="d", commands=[_cmd("hi", "async def hi()")])})
        # demote_threshold=1 → 首轮变更即降级, 但仍发射末次 warm delta.
        _snap_deltas(m, {"ch": _meta(description="d", commands=[_cmd("hi", "async def hi(v: int)")])})
        # 降级后 interface 变更不入史 (mode=demoted).
        s = m.snapshot({"ch": _meta(description="d", commands=[_cmd("hi", "async def hi(v: float)")])})
        m.ack(s)
        assert not any(d.unit == WarmUnit.INTERFACE for d in s.warm_deltas)
        # 降级轮末发射的 v:int 已在基线, 但 v:float 不应推进.
        baseline_iface = m.baseline.get("ch", {}).get(WarmUnit.INTERFACE, "")
        assert "v: int" in baseline_iface
        assert "v: float" not in baseline_iface

    def test_roundtrip_without_ack_replays_delta(self):
        m = ContextMonitor()
        meta_a = {"ch": _meta(description="original")}
        s = m.snapshot(meta_a)
        # 不 ack, 切换 metas
        m.snapshot({"ch": _meta(description="other")})
        s3 = m.snapshot({"ch": _meta(description="original")})
        assert len(s3.warm_deltas) == 1
        assert "original" in s3.warm_deltas[0].block


# ── 降级三态机 ──────────────────────────────────────────


class TestDemote:
    def test_churn_triggers_demote(self):
        m = ContextMonitor(demote_threshold=2, repromote_stable=3)
        _snap_deltas(m, {"ch": _meta(commands=[_cmd("x", "async def x()")])})
        _snap_deltas(m, {"ch": _meta(commands=[_cmd("x", "async def x(v1: int)")])})
        _snap_deltas(m, {"ch": _meta(commands=[_cmd("x", "async def x(v2: str)")])})
        # 第 3 轮: 已 demoted, 不进历史.
        s = m.snapshot({"ch": _meta(commands=[_cmd("x", "async def x(v3: float)")])})
        m.ack(s)
        assert not any(d.unit == WarmUnit.INTERFACE for d in s.warm_deltas)

    def test_repromote_after_stable(self):
        m = ContextMonitor(demote_threshold=2, repromote_stable=3)
        # 触发降级.
        _snap_deltas(m, {"ch": _meta(commands=[_cmd("x", "async def x()")])})
        _snap_deltas(m, {"ch": _meta(commands=[_cmd("x", "async def x(v1: int)")])})
        _snap_deltas(m, {"ch": _meta(commands=[_cmd("x", "async def x(v2: str)")])})
        # 降级后变更 (不入史).
        _snap_deltas(m, {"ch": _meta(commands=[_cmd("x", "async def x(v3: float)")])})
        # 连续 3 轮稳定 (v3 不变) → re-promote + commit.
        stable = {"ch": _meta(commands=[_cmd("x", "async def x(v3: float)")])}
        _snap_deltas(m, stable)
        _snap_deltas(m, stable)
        d = _snap_deltas(m, stable)
        assert any(dd.unit == WarmUnit.INTERFACE for dd in d)
        assert "v3" in next(dd.block for dd in d if dd.unit == WarmUnit.INTERFACE)
        # 此后 warm 恢复: 变更发射.
        d_post = _snap_deltas(m, {"ch": _meta(commands=[_cmd("x", "async def x(v4: bytes)")])})
        assert any(dd.unit == WarmUnit.INTERFACE for dd in d_post)

    def test_demote_is_per_unit(self):
        m = ContextMonitor(demote_threshold=2, repromote_stable=3)
        _snap_deltas(m, {"ch": _meta(states={"a": "a"}, commands=[_cmd("x", "v1")])})
        _snap_deltas(m, {"ch": _meta(states={"a": "a"}, commands=[_cmd("x", "v2")])})
        _snap_deltas(m, {"ch": _meta(states={"a": "a"}, commands=[_cmd("x", "v3")])})
        # interface 已 demoted, states 未降级.
        s = m.snapshot({"ch": _meta(states={"a": "a"}, commands=[_cmd("x", "v4")])})
        m.ack(s)
        assert not any(d.unit == WarmUnit.INTERFACE for d in s.warm_deltas)
        # states 变更 → 有 delta (states 在 warm).
        s_states = m.snapshot({"ch": _meta(states={"a": "b"}, commands=[_cmd("x", "v4")])})
        assert any(d.unit == WarmUnit.STATES for d in s_states.warm_deltas)


# ── 多 channel 与排序 ────────────────────────────────────


class TestMultiChannelAndOrdering:
    def test_independent_channels(self):
        m = ContextMonitor()
        _snap_deltas(m, {"a": _meta(description="a"), "b": _meta(description="b")})
        deltas = _snap_deltas(m, {"a": _meta(description="a2"), "b": _meta(description="b")})
        assert len(deltas) == 1
        assert deltas[0].path == "a"

    def test_delta_order_is_sorted_by_path(self):
        meta = _meta(description="d")
        deltas = _snap_deltas(ContextMonitor(), {"z": meta, "a": meta, "m": meta})
        assert [d.path for d in deltas] == ["a", "m", "z"]

    def test_states_render_sorted_by_name(self):
        meta = _meta(states={"z": "last", "a": "first", "m": "mid"})
        block = _snap_deltas(ContextMonitor(), {"ch": meta})[0].block
        a, m, z = block.index("a: first"), block.index("m: mid"), block.index("z: last")
        assert a < m < z

    def test_commands_sorted_in_interface(self):
        meta = _meta(commands=[_cmd("z", "def z_fn() -> None"), _cmd("a", "def a_fn() -> None")])
        block = _snap_deltas(ContextMonitor(), {"ch": meta})[0].block
        a, z = block.index("def a_fn"), block.index("def z_fn")
        assert a < z

    def test_metas_dict_iteration_order_does_not_matter(self):
        meta = _meta(description="d")
        s1 = ContextMonitor().snapshot({"b": meta, "a": meta})
        s2 = ContextMonitor().snapshot({"a": meta, "b": meta})
        assert [d.path for d in s1.warm_deltas] == [d.path for d in s2.warm_deltas]
