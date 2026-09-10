"""dsh session 引用 — 定位一个 session 的 turn 区间的轻量坐标.

DshSessionRef 是「坐标」不是「快照」: 只定位, 不承载历史数据. 还原 (read /
seed / fork) 全部从 live source session 的 log 重建, ref 里的自解释字段
(preset / title / cwd) 仅用于「不连 dsh 也能看懂这个 session 是什么」, 不是
还原输入.

定位是一个 **turn 区间** (span): ``start_turn`` / ``end_turn`` 是区间两端 (含端),
``start_seq`` / ``end_seq`` 是两端的 log 事件 seq, 均可缺省.

- **turn 是主坐标, seq 是派生加速器.** turn index 是「哪个 turn」的稳定语义锚,
  ref 长期持有它; seq 是某条 log 的内部位置, 没记录时从 turn 反查
  (``turn/start`` / ``turn/end`` 事件各带 ``data.turn``).
- 覆盖两阶段时序: turn 中间产出 ref (只有 turn, seq 待补) 与 turn/end 后 commit
  (补 end_seq).

还原语义 (见 ``trajectory.seed_from_log`` / ``session.fork``):
- ``end_turn`` / ``end_seq`` 决定 seed 切点 (turn/end 边界).
- ``start_turn`` / ``start_seq`` 只决定 read 窗口左端, 不参与 seed.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

__all__ = ["DshSessionRef"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DshSessionRef(BaseModel):
    """指向一个 dsh session 某 turn 区间的坐标.

    字段分两类:
    - 定位 (turn 主 / seq 辅, 区间两端含端, seq 均可缺省).
    - 自解释 (preset / title / cwd / created), 仅用于可读性, 不参与还原.
    """

    session_id: str = Field(description="定位: 哪个 dsh session.")

    start_turn: int = Field(description="定位 (主): 区间左端 turn index, 含端.")
    end_turn: int = Field(description="定位 (主): 区间右端 turn index, 含端.")

    start_seq: int | None = Field(
        default=None,
        description="定位 (派生): start_turn 的 turn/start 事件 log seq, 缺省可从 turn 反查.",
    )
    end_seq: int | None = Field(
        default=None,
        description="定位 (派生): end_turn 的 turn/end 事件 log seq, 缺省可从 turn 反查.",
    )

    preset: str | None = Field(default=None, description="自解释: agentPreset, 供参考.")
    title: str | None = Field(default=None, description="自解释: session title, 供参考.")
    cwd: str | None = Field(default=None, description="自解释: session cwd, 供参考.")
    created: datetime = Field(default_factory=_utc_now, description="ref 生成时间.")
