from ghoshell_moss.core.blueprint.mindflow import Attention, Impulse, Priority
from ghoshell_moss.core.mindflow.base_attention import AbsAttention
import time

__all__ = ['PriorityProtectionAttention']


class PriorityProtectionAttention(AbsAttention):
    """
    极简仲裁: 纯优先级 + 固定保护期 + 简单强度比较. 无衰减曲线, 无同源提权.

    规则 (按顺序命中):
    1. challenger 过期        → False  压制
    2. DEBUG                  → None   吸收
    3. 同 ID                  → None   吸收 (更新 complete)
    4. FATAL                  → True   抢占
    5. challenger prio > 当前 → True   抢占
    6. challenger prio < 当前 → False  压制
    7. prio 同级:
        保护期内         → False  压制 (防震荡)
        保护期外 + 强度高 → True   抢占
        保护期外 + 强度低 → False  压制

    两个可调 knob: priority (粗调) + strength (微调).
    保护期从 attention 创建时刻算起, _escalation_on_active 为 no-op.
    """

    def __init__(
            self,
            *,
            protection_seconds: float = 2.5,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self._protection_seconds: float = protection_seconds

    def _escalation_on_active(self) -> None:
        # no-op: 保护期基于创建时刻, 不在每轮迭代中刷新.
        pass

    def challenge(self, challenger: Impulse) -> bool | None:
        if challenger.is_stale():
            return False

        if challenger.priority == Priority.DEBUG:
            return None

        if challenger.id == self._init_impulse.id:
            self._update_current_impulse(challenger)
            return None

        challenger_prio = challenger.priority
        current_prio = self._init_impulse.priority

        if challenger.priority == Priority.FATAL or challenger_prio > current_prio:
            return True

        if challenger_prio < current_prio:
            return False

        # 同级: 保护期内压制, 保护期外比强度
        elapsed = time.monotonic() - self._strength_refreshed_at
        if elapsed < self._protection_seconds:
            return False
        return challenger.strength > self._initial_strength

    def current_strength(self) -> int:
        # 永不自行衰减退出, 只被抢占或手动 abort 结束.
        return int(self._initial_strength)
