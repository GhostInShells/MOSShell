"""ContextMonitor — 运行时上下文监控模块.

InterleavedThinkingToolset 游标体系的标准化超集: 在游标基座上增加 hot/warm
分层、单 channel 完整信息读取, 同时保留 static/dynamic 入口 (装线阶段落地).

核心抽象: ``ContextSnapshot`` 冻帧 — 一次 ``snapshot()`` 产出所有 warm deltas
与 hot 当前态, 调用方用快照拼装 moment, 落库后显式 ack(snapshot) 推进基线.
ACK 以快照为令牌, 终结无参 ack 的游标幻觉.

分类 (V2.2, 定义 vs 状态):
    warm = 能力定义: interface / states / description / instruction — 低频变更, 进历史.
    hot  = 能力状态: failure / connected / context — 每轮重绘, 尾部.

使用:
    monitor = ContextMonitor()
    snap = monitor.snapshot(metas)     # 产出一帧冻快照
    # snap.warm_deltas  → 拼入 moment 的 durable 段
    # snap.hot_messages → 拼入 moment 的 ephemeral 尾部
    monitor.ack(snap)                   # 确认入史, 推进基线
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

from ghoshell_moss.core.concepts.channel import ChannelMeta
from ghoshell_moss.core.ctml.v1_0.prompts import make_interfaces
from ghoshell_moss.message import Message

__all__ = [
    "ContextMonitor",
    "ContextSnapshot",
    "WarmDelta",
    "WarmUnit",
]


class WarmUnit(str, Enum):
    """warm 数据单元 — 字段组粒度 (V2.3). 每个单元独立 hash/delta/降级."""

    DESC_INSTRUCTION = "desc_instruction"
    STATES = "states"
    INTERFACE = "interface"


_UNIT_TAGS: dict[WarmUnit, tuple[str, ...]] = {
    WarmUnit.DESC_INSTRUCTION: ("description", "instruction"),
    WarmUnit.STATES: ("states",),
    WarmUnit.INTERFACE: ("interface",),
}


@dataclass(frozen=True)
class WarmDelta:
    """一次 warm 变更事件.

    kind: add=整 channel 出现 / update=某单元变更 / remove=channel 移除 (tombstone).
    block 是 channel 包裹的渲染文本. 历史里同 path 后块字段级覆盖前块.
    """

    kind: Literal["add", "update", "remove"]
    path: str
    unit: WarmUnit | None
    block: str

    def to_messages(self) -> list[Message]:
        return [Message.new(tag="", timestamp=False).with_content(self.block)]


@dataclass(frozen=True)
class ContextSnapshot:
    """一次 context monitor 快照 — 冻帧, ack 的显式令牌.

    调用方用 warm_deltas 拼入 durable 段, hot_messages 拼入 ephemeral 尾部,
    落库后把同一快照交给 ``monitor.ack(snapshot)`` 推进基线.

    warm_deltas 已按 path 排序, 可直接使用.
    """

    warm_deltas: tuple[WarmDelta, ...]
    hot_messages: tuple[Message, ...]

    # ack 验证与落基线所需的内态.
    _ack_id: int
    _frame_projection: dict[str, dict[WarmUnit, str]]


# ── warm 投影 (确定性渲染) ──────────────────────────────────


def _render_desc_instruction(meta: ChannelMeta) -> str:
    parts = []
    if meta.description:
        parts.append(f"<description>\n{meta.description}\n</description>")
    if meta.instruction:
        parts.append(f"<instruction>\n{meta.instruction}\n</instruction>")
    return "\n".join(parts)


def _render_states(meta: ChannelMeta) -> str:
    parts = []
    if meta.states:
        lines = [
            f"- {name}: {desc.replace(chr(10), ';')}"
            for name, desc in sorted(meta.states.items())
        ]
        parts.append("<states>\n" + "\n".join(lines) + "\n</states>")
    if meta.current_state:
        parts.append(f"Current state: {meta.current_state}")
    return "\n".join(parts)


def _render_interface(meta: ChannelMeta) -> str:
    if not meta.commands:
        return ""
    commands = sorted(meta.commands, key=lambda c: c.name)
    return make_interfaces(
        meta.model_copy(update={"commands": commands}),
        dynamic=True,
        sustain=True,
    )


def _project_units(meta: ChannelMeta) -> dict[WarmUnit, str]:
    """确定性投影: 只含 warm 单元, 排除 hot (failure/context). 空单元省略."""
    result: dict[WarmUnit, str] = {}
    if text := _render_desc_instruction(meta):
        result[WarmUnit.DESC_INSTRUCTION] = text
    if text := _render_states(meta):
        result[WarmUnit.STATES] = text
    if text := _render_interface(meta):
        result[WarmUnit.INTERFACE] = text
    return result


def _wrap_channel(path: str, inner: str) -> str:
    return f'<channel name="{path}">\n{inner}\n</channel>'


def _render_all_units(units: dict[WarmUnit, str]) -> str:
    return "\n".join(units[unit] for unit in WarmUnit if unit in units)


def _cleared_unit_marker(unit: WarmUnit) -> str:
    return "\n".join(f"<{tag}/>" for tag in _UNIT_TAGS[unit])


# ── hot 渲染 (当前状态层, 每轮新鲜) ──────────────────────────


def _render_hot(meta: ChannelMeta) -> str | None:
    """渲染 channel 状态层: failure / context. 不包含 warm 定义字段."""
    parts = []
    if meta.failure:
        parts.append(f"<failure>\n{meta.failure}\n</failure>")
    if meta.context:
        ctx_lines = ["<context>"]
        ctx_lines.extend(msg.to_content_string() for msg in meta.context)
        ctx_lines.append("</context>")
        parts.append("\n".join(ctx_lines))
    return "\n".join(parts) if parts else None


def _build_hot_messages(metas: dict[str, ChannelMeta]) -> list[Message]:
    """按 path 排序渲染所有 channel 的 hot 状态."""
    result = []
    for path in sorted(metas):
        hot = _render_hot(metas[path])
        if hot:
            block = _wrap_channel(path, hot)
            result.append(Message.new(tag="", timestamp=False).with_content(block))
    return result


# ── 降级三态机内态 ────────────────────────────────────────


@dataclass
class _UnitState:
    mode: Literal["warm", "demoted"] = "warm"
    churn: int = 0
    stable: int = 0
    last_text: str | None = None


class ContextMonitor:
    """帧级 compare-and-emit 的 warm/hot 上下文检测器.

    ``snapshot(metas)`` 产出一帧冻快照 (warm deltas + hot 当前态).
    ``ack(snapshot)`` 以快照为令牌推进基线 — 无参 ack 的游标幻觉由此消除.

    纯函数, 零 shell 依赖. 游标原语 (drain/status) 与 static/dynamic 入口
    在装线阶段接入.
    """

    def __init__(
            self,
            *,
            demote_threshold: int = 2,
            repromote_stable: int = 3,
    ) -> None:
        self._demote_threshold = demote_threshold
        self._repromote_stable = repromote_stable
        # baseline = 最后一次 ACK 的帧投影 = 当前历史里的 warm 版本.
        self._baseline: dict[str, dict[WarmUnit, str]] = {}
        # 降级三态机状态, 键 (path, unit).
        self._states: dict[tuple[str, WarmUnit], _UnitState] = {}
        # ack 游标: 单调递增帧号, 验证快照合法性.
        self._frame_counter: int = 0
        self._last_acked_frame: int = -1

    @property
    def baseline(self) -> dict[str, dict[WarmUnit, str]]:
        """最后一次 ACK 的帧投影 (调试 / 测试反射用)."""
        return {p: dict(units) for p, units in self._baseline.items()}

    # ── 核心 API ───────────────────────────────────────────

    def snapshot(self, metas: dict[str, ChannelMeta]) -> ContextSnapshot:
        """对当前 metas 帧做 compare-and-emit, 产出一帧冻快照.

        对比基准是最后一次 ACK 的投影. ACK 前多次 snapshot 会重算同一批 delta
        (每次都是独立冻帧, 各自 ack 时判定是否已经过时).
        """
        self._frame_counter += 1
        ack_id = self._frame_counter

        current = {path: _project_units(meta) for path, meta in metas.items()}
        deltas: list[WarmDelta] = []
        baseline = self._baseline

        # L0: 存在性.
        for path in sorted(current.keys() - baseline.keys()):
            units = current[path]
            if units:
                deltas.append(
                    WarmDelta("add", path, None, _wrap_channel(path, _render_all_units(units)))
                )
            self._init_states(path, units)
        for path in sorted(baseline.keys() - current.keys()):
            deltas.append(WarmDelta("remove", path, None, f'<channel name="{path}" state="removed"/>'))
            self._drop_states(path)

        # L1/L2: 存活 channel, 逐单元对比.
        for path in sorted(current.keys() & baseline.keys()):
            base_units = baseline[path]
            cur_units = current[path]
            for unit in WarmUnit:
                if unit in base_units or unit in cur_units:
                    self._tick_unit(path, unit, cur_units.get(unit), base_units.get(unit), deltas)

        hot_messages = tuple(_build_hot_messages(metas))

        return ContextSnapshot(
            warm_deltas=tuple(deltas),
            hot_messages=hot_messages,
            _ack_id=ack_id,
            _frame_projection=current,
        )

    def ack(self, snapshot: ContextSnapshot) -> None:
        """确认快照的 warm delta 已进入历史, 推进基线.

        - 快照的 ack_id 不大于上次已 ack 帧号时 no-op (已过时或重复 ack).
        - 只推进快照内实际发射的单元; 降级单元在尾部的版本不进基线.
        """
        if snapshot._ack_id <= self._last_acked_frame:
            return
        current = snapshot._frame_projection
        for delta in snapshot.warm_deltas:
            if delta.kind == "add":
                self._baseline[delta.path] = dict(current[delta.path])
            elif delta.kind == "remove":
                self._baseline.pop(delta.path, None)
            else:
                assert delta.unit is not None
                units = self._baseline.setdefault(delta.path, {})
                if delta.unit in current[delta.path]:
                    units[delta.unit] = current[delta.path][delta.unit]
                else:
                    units.pop(delta.unit, None)
        self._last_acked_frame = snapshot._ack_id

    # ── 内部 ───────────────────────────────────────────────

    def _init_states(self, path: str, units: dict[WarmUnit, str]) -> None:
        for unit in WarmUnit:
            self._states[(path, unit)] = _UnitState(last_text=units.get(unit))

    def _drop_states(self, path: str) -> None:
        for unit in WarmUnit:
            self._states.pop((path, unit), None)

    def _tick_unit(
            self,
            path: str,
            unit: WarmUnit,
            cur_text: str | None,
            hist_text: str | None,
            deltas: list[WarmDelta],
    ) -> None:
        st = self._states.setdefault((path, unit), _UnitState())
        changed_round = cur_text != st.last_text

        if st.mode == "warm":
            if cur_text != hist_text:
                deltas.append(self._emit_update(path, unit, cur_text))
                st.churn += 1
                if st.churn >= self._demote_threshold:
                    st.mode = "demoted"
                    st.stable = 0
            else:
                st.churn = 0
        else:  # demoted — 变更由尾部承载, 不进历史.
            if changed_round:
                st.stable = 0
            else:
                st.stable += 1
                if st.stable >= self._repromote_stable:
                    deltas.append(self._emit_update(path, unit, cur_text))
                    st.mode = "warm"
                    st.churn = 0
                    st.stable = 0

        st.last_text = cur_text

    def _emit_update(self, path: str, unit: WarmUnit, text: str | None) -> WarmDelta:
        if text is not None:
            block = _wrap_channel(path, text)
        else:
            block = _wrap_channel(path, _cleared_unit_marker(unit))
        return WarmDelta("update", path, unit, block)
