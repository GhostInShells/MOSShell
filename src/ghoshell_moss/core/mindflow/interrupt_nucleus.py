"""InterruptNucleus — turns ``interrupt`` signals into interrupt-mode impulses.

四元 nucleus (扩为五元) 之一: 配对 ``ImpulsePrimitive.interrupt`` 的"中断动作" 通道.
监听 ``InterruptSignalMeta`` (signal name = ``"interrupt"``), 把 signal 包装成
``FATAL + notify + thinking_effort='none' + interrupt=True`` 的 impulse.

FATAL 保证抢占成功, notify 走 default 成功路径创建新 attention,
effort='none' 让 ghost.articulate 提前返回, ``interrupt=True`` 让
``ghost_runtime._run_articulator`` 在新 attention 起步时调
``shell.stop_interpretation()`` 清干净旧 logos.

结构对比:
- 与 ``CommandNucleus`` 同构 (fire-and-forget, 无 buffer)
- 与 ``BroadcastNucleus`` (未实现, 由 ImpulsePrimitive.broadcast 单原语承载) 对偶:
  broadcast 用 silent 不接管, interrupt 用 notify 接管但立即放手

反向 suppress 模型:
- 与 ``InputSignalNucleus`` 的"失败 suppress" (输方反复试) 相反
- interrupt 是 "胜利 suppress" — 仲裁胜利后进入冷静期, 防止短时间内反复 interrupt
  导致 shell churn (反复 stop_interpretation + 重建 attention 的 DDOS-like 抖动)
- 冷静期内 add_signal 静默丢 — interrupt 没有"累积" 语义, 多个等价于一个
"""
import time
from typing import Callable, Iterable
from typing_extensions import Self

from ghoshell_container import IoCContainer

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.message import ContextType
from ghoshell_moss.core.blueprint import Impulse
from ghoshell_moss.core.blueprint.mindflow import (
    SignalMeta, SignalName, Priority, Signal,
    Nucleus, NucleusMeta, ImpulsePrimitive,
)

__all__ = [
    'InterruptNucleus', 'InterruptSignalMeta', 'InterruptNucleusMeta',
    'new_interrupt_signal',
]


class InterruptSignalMeta(SignalMeta):
    """Signal meta for ``interrupt`` — must-deliver, must-interrupt.

    与 ``BroadcastSignalMeta`` (由 ImpulsePrimitive.broadcast 离散使用)
    形成对偶: broadcast 不接管 attention 只 buffer messages;
    interrupt 接管 attention 并打断 shell 执行, 但立即放手不思考.

    priority 锁 FATAL — interrupt 没有"低优中断" 的语义.
    """

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'interrupt'

    @classmethod
    def priority(cls) -> Priority:
        return Priority.FATAL


class InterruptNucleus(Nucleus):
    """Interrupt channel — last-impulse cache with victory-side cooldown.

    Cache 模式: ``add_signal`` 写入 ``_impulse``, mindflow 通过 ``peek/pop_impulse``
    拉取. interrupt 是离散事件, last-wins (新的覆盖旧的); 即便多个 interrupt
    在 mindflow 消费前抵达, 仲裁结果都等价 — 都是 FATAL 抢占成功, 都触发
    shell.stop_interpretation.

    反向 suppress (与 InputSignalNucleus 等"失败侧 suppress" 对偶):
    - pop_impulse 触发时启动冷静期 (impulse 被仲裁取出且执行, 视为胜利)
    - 冷静期内 add_signal 静默丢 (不进 cache, 不通知)
    - 冷静期到 → 自然恢复

    Why 反向: FATAL 仲裁只有 same-id absorb 或 stale 才会"失败", 这两种都不
    需要冷静期; 真实 DOS 风险是反向 — 反复成功 interrupt 导致 shell churn.

    Why 不聚合: 多个 interrupt 合并无语义价值, 第一个就够了.
    """

    NAME = 'interrupt_nucleus'

    def __init__(
            self,
            *,
            name: str = NAME,
            suppress_seconds: float = 0.5,
            logger: LoggerItf | None = None,
    ):
        self._name = name
        self._suppress_seconds = suppress_seconds
        self._impulse_notify: Callable[[Impulse], None] | None = None
        self._is_running = False
        self._logger = logger or get_moss_logger()
        # 反向 suppress: 胜利后才设, 失败侧不动.
        self._suppress_until: float = 0.0
        self._impulse: Impulse | None = None

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return 'interrupt channel — must-deliver, takes attention then drops it without thinking'

    def status(self) -> str:
        return ''

    def signals(self) -> list[SignalName]:
        return [InterruptSignalMeta.signal_name()]

    def clear(self) -> None:
        self._suppress_until = 0.0
        self._impulse = None

    def add_signal(self, signal: Signal) -> None:
        if not self._is_running:
            return
        # 反向 suppress: 上一次中断刚胜利, 冷静期内静默丢.
        if time.monotonic() < self._suppress_until:
            return
        impulse = self.build_impulse(signal)
        if impulse is None:
            return
        self._impulse = impulse
        if self._impulse_notify:
            self._impulse_notify(impulse)

    def build_impulse(self, signal: Signal) -> Impulse | None:
        if not InterruptSignalMeta.match(signal):
            return None
        impulse = Impulse.from_signal(signal, source=self.name())
        # interrupt primitive 强制 FATAL + notify + effort='none' + interrupt=True.
        # Signal.priority 被覆盖 — interrupt 的语义承诺不可降级.
        return ImpulsePrimitive.interrupt(impulse)

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            impulse_notify: Callable[[Impulse], None],
    ) -> None:
        self._impulse_notify = impulse_notify

    def suppress(self, suppress_by: Impulse) -> None:
        # 失败侧不进冷静期 — FATAL 仲裁失败只可能是 same-id absorb 或 stale,
        # 这两种都不需要冷静期 (absorb 已被内部处理, stale 在入口丢).
        # 但 cache 仍要清, 让 nucleus 状态正确反映 "没有 pending impulse".
        self._impulse = None

    def pop_impulse(self, impulse: Impulse) -> None:
        # 反向 suppress: 仲裁胜利后启动冷静期, 防止 shell churn.
        if not self._is_running:
            return
        if self._impulse is impulse:
            self._impulse = None
        self._suppress_until = time.monotonic() + self._suppress_seconds

    def peek(self, no_stale: bool = True) -> Impulse | None:
        if self._impulse is None:
            return None
        if no_stale and self._impulse.is_stale():
            self._impulse = None
            return None
        return self._impulse

    def is_running(self) -> bool:
        return self._is_running

    async def __aenter__(self) -> Self:
        self._is_running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self._is_running = False
        self._impulse = None


class InterruptNucleusMeta(NucleusMeta):
    """Factory meta — lets ``moss manifests nuclei`` discover InterruptNucleus."""

    def __init__(self, *, suppress_seconds: float = 0.5):
        self._suppress_seconds = suppress_seconds

    def name(self) -> str:
        return InterruptNucleus.NAME

    def description(self) -> str:
        return 'interrupt channel that turns interrupt signals into FATAL+notify+effort=none+interrupt impulses'

    def signals(self) -> Iterable[type[SignalMeta]]:
        yield InterruptSignalMeta

    def factory(self, container: IoCContainer) -> Nucleus:
        logger = container.get(LoggerItf)
        return InterruptNucleus(suppress_seconds=self._suppress_seconds, logger=logger)


def new_interrupt_signal(
        *messages: ContextType,
        description: str = '',
        stale_timeout: float = 0,
        hint: str = '',
) -> Signal:
    """Helper — construct an ``interrupt`` signal in one call.

    priority is not exposed — interrupt is always FATAL by contract.
    Use ``new_notify_signal`` for soft preemption attempts.
    """
    return InterruptSignalMeta().to_signal(
        *messages,
        description=description,
        stale_timeout=stale_timeout,
        hint=hint,
    )
