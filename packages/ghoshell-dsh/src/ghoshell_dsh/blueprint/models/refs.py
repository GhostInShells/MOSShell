"""dsh session 引用 — 定位一个 session 的 turn 区间的轻量坐标.

DshSessionRef 是「坐标」不是「快照」: 只定位, 不承载历史数据. 还原 (read /
seed / fork) 全部从 live source session 的 log 重建, ref 里的自解释字段
(preset / title / cwd) 仅用于「不连 dsh 也能看懂这个 session 是什么」, 不是
还原输入.

定位是一个 **半开 turn 区间** (span): ``(start_turn, end_turn]`` ——
**下界开、上界闭**: turn ``start_turn`` 归**上一个**区间 (它就是上一个 ref 的
``end_turn``), 本区间不重复覆盖它; turn ``end_turn`` 含端. 因此相邻区间严丝合缝
(``(0,1] (1,2] (2,3]``), 新区间的下界逐字抄旧区间的 ``end_turn``, 不需要 +1;
``start_turn == end_turn`` 即空区间. ``start_seq`` / ``end_seq`` 是两端的 log 事件
seq, 均可缺省.

- **turn 是主坐标, seq 是派生加速器.** turn index 是「哪个 turn」的稳定语义锚,
  ref 长期持有它; seq 是某条 log 的内部位置, 没记录时从 turn 反查
  (``turn/start`` / ``turn/end`` 事件各带 ``data.turn``).
- 覆盖两阶段时序: turn 中间产出 ref (只有 turn, seq 待补) 与 turn/end 后 commit
  (补 end_seq).

还原语义 (见 ``trajectory.seed_from_log`` / ``session.fork``):
- ``end_turn`` / ``end_seq`` 决定 seed 切点 (切在 ``end_turn`` 的 ``turn/end`` 之后,
  ``end_turn`` 含端进 seed).
- ``start_turn`` / ``start_seq`` 只决定 read 窗口左端 (开), 不参与 seed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ghoshell_dsh.blueprint.anchor import DshSessionAnchor

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

    start_turn: int = Field(
        description="定位 (主): 区间左端 turn index, **开** (该 turn 归上一个区间).",
    )
    end_turn: int = Field(
        description="定位 (主): 区间右端 turn index, **闭** (含端). start_turn == end_turn 即空区间.",
    )

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

    def to_anchor(
        self,
        *,
        at: Literal["start", "end"] = "end",
        note: str | None = None,
    ) -> DshSessionAnchor:
        """把这段区间折成一个锚点 —— 取它的某一端.

        为什么需要这个转换: ref 记的是**区间** (memento 的"这一段时间发生了什么"),
        anchor 记的是**一个点** (交给另一个进程"从这里往后看"). 两者是同一坐标的两种粒度,
        不是两个概念, 所以转换必须显式、且必须承认它是**有损的**:

        - 一个区间折成一个点, **左端就没了** —— 想再读回同一段, 得重新拿 ref, 不能从 anchor 反推;
        - ``seq`` 是主坐标: ref 只记了 turn 而 ``end_seq`` 为空时, 折出来的 anchor 会带着
          ``seq=None``, 而 None 在 anchor 里意为"会话起点" —— 那是另一个意思. 所以本方法
          **要求那一端有 seq**, 缺了就报错, 不替你猜.

        实现在 anchor 侧 (``from_ref``), 这里只是让"我有一段, 给我一个点"有地方可写.
        延迟导入是为了不把 models 反向绑到 blueprint 的上一层.
        """
        from ghoshell_dsh.blueprint.anchor import DshSessionAnchor

        seq = self.end_seq if at == "end" else self.start_seq
        if seq is None:
            raise ValueError(
                f"ref has no {at}_seq: a turn-only ref cannot become an anchor without guessing "
                f"(resolve the seq first, or keep the ref)"
            )
        return DshSessionAnchor.from_ref(self, at=at, note=note)
