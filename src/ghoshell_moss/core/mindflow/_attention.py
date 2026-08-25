"""
Attention 基础结构实现.

重构自历史文件 ./base_attention.py
"""

from abc import abstractmethod, ABC
from typing import Literal
from ghoshell_moss.core.blueprint.mindflow import (
    Attention, Impulse, Priority
)
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.contracts import LoggerItf, get_moss_logger
import time
import asyncio

__all__ = ['AbsAttention', 'BaseAttention']


class AbsAttention(Attention, ABC):
    """
    Attention 的抽象基类 (数据化 / 去生命周期化).
    只持有仲裁所需状态: 当前 impulse, 强度/衰减/保护期, 优先级, abort 生命周期.
    不管理 think/action 的生成与 observe 循环 — 那些移到 Mindflow 与 Think/Action.

    current_strength() 和 arbit_challenge_by_strength() 留给子类实现仲裁策略.
    """

    def __init__(
            self,
            *,
            impulse: Impulse,
            logger: LoggerItf | None = None,
            system_floor_strength: float = 0.0,  # 强度 floor — 预留, 当前实现未消费
    ):
        self._init_impulse: Impulse = impulse
        self._priority = impulse.priority
        self._has_any_completed_impulse = ThreadSafeEvent()
        self._logger = logger or get_moss_logger()
        self._protected_until: float = time.monotonic() + impulse.protection_time

        # 关键的 flags.
        self._aborted_event = ThreadSafeEvent()
        # 运行时.
        self._event_loop: asyncio.AbstractEventLoop | None = None

        # 这三个值通过 absorb_impulse 播种/更新, 子类只读.
        self._strength_start_value: float = 0.0
        self._strength_refreshed_at: float = 0.0
        self._strength_decay_time: float = 0.0

        # 强度 floor — 预留的构造参数, 供子类判断衰减自尽/可抢占边界 (当前实现未消费).
        self._system_floor_strength: float = system_floor_strength
        self._attention_level_priority: Priority | None = None

        self._started: bool = False
        self._closing: bool = False
        self._closed_event = ThreadSafeEvent()
        self._abort_reason = ''
        # update the impulse
        self._log_prefix = f"<Attention id={self._init_impulse.id}>"

        # 播种强度/保护期: 去生命周期化后, 强度不再由构造函数字段默认 0, 而是对初始 impulse
        # 调用 absorb_impulse 播种, 否则 current_strength() 走 decay_duration=0 会除零.
        # 同时让首个 complete impulse 置位 _has_any_completed_impulse, wait_ready() 得以及时返回.
        self.absorb_impulse(self._init_impulse)

    def __repr__(self):
        return self._log_prefix

    def absorb_impulse(self, impulse: Impulse) -> Impulse | None:
        """
        仅由 Mindflow 调用: challenge() 返回 'absorb' 后, 把同 id impulse 折进当前 attention.

        无条件刷新强度/优先级/保护期, 为下一帧预订衰减起点 (moments 已 inject 该 impulse).
        返回 None = 同 id 已吸收为 _init_impulse; 返回 impulse = 异 id 未吸收, 留给调用方路由
        (如 ghost 主动 pull 的 command result, 此时同样要提前更新当前 attention).
        """
        # 起始强度, 用于计算当前强度.
        self._strength_start_value = impulse.strength
        # 强度最后更新时间.
        self._strength_refreshed_at = time.monotonic()
        self._priority = impulse.priority
        # 保护期所在时间点.
        self._protected_until = self._strength_refreshed_at + impulse.protection_time
        self._strength_decay_time = self._init_impulse.strength_decay_seconds
        if self._strength_decay_time <= 0:
            # 不要让它为0.
            self._strength_decay_time = 1
        if impulse.id == self._init_impulse.id:
            self._init_impulse = impulse
            if impulse.complete:
                # 只有 complete 类型的 impulse 才会进入 buffer, 其它的只是占据注意力.
                self._has_any_completed_impulse.set()
            return None
        else:
            return impulse

    @property
    def strength_refreshed_at(self) -> float:
        """测试专用参数, 避免取私有值. """
        return self._strength_refreshed_at

    @property
    def strength_start_value(self) -> float:
        """强度衰减曲线起点. 由最新进入 buffer 的 impulse 刷新."""
        return self._strength_start_value

    @property
    def strength_decay_time(self) -> float:
        """attention 衰减总时长, 钉在 init impulse 的 strength_decay_seconds (clamp 到 >=1)."""
        return self._strength_decay_time

    @property
    def protected_until(self) -> float:
        """同优先级保护期截止 monotonic 时间."""
        return self._protected_until

    def draw_from(self) -> Impulse:
        return self._init_impulse

    async def wait_ready(self) -> Impulse:
        """等第一个 complete impulse; 若 attention 被终止 (abort/exit) 也返回当前 impulse.
        返回后用 is_aborted() 区分正常就绪 vs 被终止."""
        await self._has_any_completed_impulse.wait()
        return self.draw_from()

    def set_priority(self, priority: Priority | None) -> None:
        self._attention_level_priority = priority

    def escalate(self) -> None:
        self._escalation_on_active()

    def _escalation_on_active(self) -> None:
        # 刷新活跃时间, 阻止强度衰减误触发自尽; 同时在 challenge 时维持当前强度.
        self._strength_refreshed_at = time.monotonic()
        self._logger.debug("%s _escalation_on_active: strength_refreshed_at=%.3f",
                           self._log_prefix, self._strength_refreshed_at)

    @abstractmethod
    def current_strength(self) -> int:
        """基于剩余生存权重的线性衰减模型."""
        pass

    async def challenge(self, challenger: Impulse) -> Literal['win', 'lose', 'absorb']:
        """
        仲裁新 impulse. 结果交由 Mindflow 处置:
          win     — 抢占成功, 当前 attention 被 abort, 新 impulse 接管.
          lose    — 压制, 挑战失败, 挑战者被 suppress.
          absorb  — 吸收, 被内部消化 (同源更新/buffer/DEBUG), 不影响当前 attention;
                    随后 Mindflow 调 absorb_impulse.
        """
        if challenger.is_stale():
            return 'lose'
        if challenger.id == self._init_impulse.id and not self._init_impulse.complete:
            # 相同 id 的永远可以 buffer. 但只 buffer 一次.
            return 'absorb'
        elif challenger.priority == Priority.FATAL or challenger.priority > self.priority():
            return 'win'
        elif challenger.priority < self.priority():
            return 'lose'
        # 相同优先级保护期.
        elif self._protected_until > time.monotonic():
            return 'lose'
        if self.arbit_challenge_by_strength(challenger):
            return 'win'
        else:
            return 'lose'

    def is_protected(self) -> bool:
        return self._protected_until > time.monotonic()

    def priority(self) -> Priority:
        if self._attention_level_priority is not None:
            return self._attention_level_priority
        return self._priority

    @abstractmethod
    def arbit_challenge_by_strength(self, challenger: Impulse) -> bool:
        """
        同级仲裁: 按强度比较决定 challenger 是否抢占当前 attention.
        True: 发起强于当前, 抢占成功.  False: 发起弱于当前, 被压制.
        """
        pass

    def is_closed(self) -> bool:
        return self._aborted_event.is_set()

    def abort(self, error: str | Exception | None) -> None:
        if self._aborted_event.is_set():
            return
        self._aborted_event.set()
        # 唤醒 wait_ready: 终止也算"不必再等 complete 尾包". 调用方用 is_aborted() 区分正常 vs 被终止.
        self._has_any_completed_impulse.set()
        if error is not None:
            if isinstance(error, Exception):
                self._abort_reason = f"Error: {error}"
            elif isinstance(error, str):
                self._abort_reason = error

    def abort_reason(self) -> str:
        return self._abort_reason

    async def wait_abort(self) -> None:
        await self._aborted_event.wait()

    def stop(self) -> None:
        self.abort(None)

    def is_running(self) -> bool:
        return not self._aborted_event.is_set() and self._started and not self._closing

    def is_aborted(self) -> bool:
        return self._aborted_event.is_set()

    async def __aenter__(self):
        if self._started:
            return self
        self._started = True
        self._event_loop = asyncio.get_running_loop()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        关键是哪些异常是需要对外抛出的.
        """
        if self._closing:
            return None
        self._closing = True
        try:
            # 取消 inner task.
            self.abort(self._abort_reason)
        finally:
            # 两个确保能够退出的标记.
            self._aborted_event.set()
            self._closed_event.set()
            # 同样唤醒 wait_ready, 兜底 abort() 之外的退出路径.
            self._has_any_completed_impulse.set()


class BaseAttention(AbsAttention):
    """
    Beta 版本的强度衰减仲裁实现.
    提供默认强度衰减曲线与同源提权仲裁 (current_strength / arbit_challenge_by_strength).
    """

    def __init__(
            self,
            *,
            impulse: Impulse,
            logger: LoggerItf | None = None,
            system_floor_strength: float = 0.0,
            source_escalation: float = 1.1,
            max_protection_time: float = 3.0,
            protection_duration_ratio: float = 0.2,
    ):
        super().__init__(
            impulse=impulse,
            logger=logger,
            system_floor_strength=system_floor_strength,
        )
        self._source_escalation: float = source_escalation
        self._max_protection_time: float = max_protection_time
        self._protection_duration_ratio: float = min(max(protection_duration_ratio, 0.0), 1.0)

    def arbit_challenge_by_strength(self, challenger: Impulse) -> bool:
        challenger_strength = challenger.strength
        if challenger.source == self._init_impulse.source:
            challenger_strength = int(challenger_strength * self._source_escalation)
        current_strength = self.current_strength()
        return current_strength < challenger_strength

    def current_strength(self) -> int:
        """基于剩余生存权重的线性衰减模型. 强度在不提权情况下, 默认要衰减到 0"""
        now = time.monotonic()
        elapsed = now - self._strength_refreshed_at

        protection_time = min(
            self._strength_decay_time * self._protection_duration_ratio,
            self._max_protection_time,
        )
        if elapsed < protection_time:
            return int(self._strength_start_value * self._source_escalation)

        decay_elapsed = elapsed - protection_time
        decay_duration = self._strength_decay_time - protection_time
        progress = min(decay_elapsed / decay_duration, 1.0)
        decay_factor = 1.0 if self._init_impulse.complete else 1.5
        current = self._strength_start_value * (1.0 - (progress * decay_factor))
        return int(max(current, 0))
