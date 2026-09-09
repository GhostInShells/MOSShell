"""dsh session 引用 — 定位一个 session 的某 turn 的轻量坐标.

DshSessionRef 是「坐标」不是「快照」: 只定位, 不承载历史数据. 还原 (fork /
read) 全部从 live source session 重建, ref 里的自解释字段 (preset / title /
cwd) 仅用于「不连 dsh 也能看懂这个 session 是什么」, 不是还原输入.

定位靠三个字段, 按优先级取用: ``turn`` > ``end_seq`` > ``start_seq``.

- ``turn`` — turn index (data.turn), turn 中间就能确定. 语义锚.
- ``end_seq`` — 该 turn 的 turn/end 事件在 log 里的 seq, completed turn 边界.
- ``start_seq`` — 该 turn 的 turn/start 事件在 log 里的 seq, turn 中间就能确定.

官方 fork 的 ``atSeq`` 需要合法 seq (end_seq / start_seq); 我们自己的接口可
以 turn index 反查 seq. 三者都允许缺省, 覆盖「turn 中间产出 ref (只有 turn +
start_seq)」与「turn/end 后 commit (补 end_seq)」两阶段时序.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

__all__ = ["DshSessionRef"]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class DshSessionRef(BaseModel):
    """指向一个 dsh session 某 turn 的坐标.

    字段分两类:
    - 定位 (按优先级 turn > end_seq > start_seq 取用, 均允许缺省).
    - 自解释 (preset / title / cwd / created), 仅用于可读性, 不参与还原.
    """

    session_id: str = Field(description="定位: 哪个 dsh session.")

    turn: int | None = Field(
        default=None,
        description="定位 (最高优先级): turn index (data.turn). turn 中间即可确定.",
    )
    end_seq: int | None = Field(
        default=None,
        description="定位: 该 turn 的 turn/end 事件 log seq (completed turn 边界).",
    )
    start_seq: int | None = Field(
        default=None,
        description="定位: 该 turn 的 turn/start 事件 log seq (turn 中间即可确定).",
    )

    preset: str | None = Field(default=None, description="自解释: agentPreset, 供参考.")
    title: str | None = Field(default=None, description="自解释: session title, 供参考.")
    cwd: str | None = Field(default=None, description="自解释: session cwd, 供参考.")
    created: datetime = Field(default_factory=_utc_now, description="ref 生成时间.")
