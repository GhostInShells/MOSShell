"""DoloresEgoNucleus — ego 自醒 nucleus (最小流程切片).

特化 nucleus: 接收 Dolores Ego 的 turn/start 监听打来的自醒 signal, 每封产出一个
``Priority.INFO`` / 默认 mode (正常仲裁) / 空 message body 的 impulse, 唤醒静默的
mindflow 走一轮正常挑战.

当前只验证「signal → nucleus → impulse」链路, 消息体留空, 后续再织入轨迹影像等物料.
与 silent_nucleus 的区别: silent 是低污染 buffer 聚合 (mode=silent, 不接管 attention),
本 nucleus 是一次性自醒 (mode 默认, 走正常仲裁) — 都起"唤醒"作用, 但语义不同.
"""

from collections.abc import Iterable
from typing import Callable

from ghoshell_container import IoCContainer
from typing_extensions import Self

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.blueprint.mindflow import (
    Impulse,
    Nucleus,
    NucleusMeta,
    Priority,
    Signal,
    SignalMeta,
    SignalName,
)
from ghoshell_moss.message import ContextType

__all__ = [
    "NAME",
    "SIGNAL_NAME",
    "DoloresEgoSignalMeta",
    "DoloresEgoNucleus",
    "DoloresEgoNucleusMeta",
    "new_dolores_ego_signal",
]

NAME = "dolores_ego_nucleus"
"""nucleus 名 — 也作为 impulse 的 source 名."""

SIGNAL_NAME = "dolores/ego"
"""自醒通道 signal 名 — ego 的 turn/start 监听发出该 signal."""


class DoloresEgoSignalMeta(SignalMeta):
    """自醒通道 signal meta — 一次 turn/start 观察即一封自醒 signal."""

    @classmethod
    def signal_name(cls) -> SignalName:
        return SIGNAL_NAME

    @classmethod
    def priority(cls) -> Priority:
        return Priority.INFO


class DoloresEgoNucleus(Nucleus):
    """自醒 nucleus — 发 BACKGROUND 挑战包 (发完丢), attended 加工成 INFO 运行包唤醒 attention."""

    NAME = NAME

    def __init__(
        self,
        *,
        name: str = NAME,
        description: str = "wake moss when the dolores ego agent is running",
        logger: LoggerItf | None = None,
    ):
        self._name = name
        self._description = description
        self._target_signal = SIGNAL_NAME
        self._logger = logger or get_moss_logger()
        self._impulse: Impulse | None = None
        self._broadcast_cb: Callable[[Signal], None] | None = None
        self._notify_cb: Callable[[Impulse], None] | None = None
        self._index = 0
        self._running = False

    # -- Nucleus ABC --

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def status(self) -> str:
        return ""

    def signals(self) -> list[SignalName]:
        return [self._target_signal]

    def clear(self) -> None:
        self._impulse = None

    def with_bus(
        self,
        signal_broadcast: Callable[[Signal], None],
        fire_impulse: Callable[[Impulse], None],
    ) -> None:
        self._broadcast_cb = signal_broadcast
        self._notify_cb = fire_impulse

    def add_signal(self, signal: Signal) -> None:
        if not self.is_running():
            return
        if signal.name != self._target_signal:
            return
        # 挑战包: BACKGROUND — 低调, 只在 mindflow idle 时 initial 成功; 有 attention 时 suppress.
        self._index += 1
        self._impulse = Impulse(
            source=self._name,
            source_idx=self._index,
            id=signal.id,
            priority=Priority.BACKGROUND,
            messages=[],  # 纯空 body — 处理逻辑在 thinking/enter 层定义.
            description="",  # 自醒通道, 无摘要
            complete=True,  # mode 默认空 = 正常仲裁 (非 silent)
        )
        if self._notify_cb is not None:
            self._notify_cb(self._impulse)

    def suppress(self, suppress_by: Impulse, suppressed: Impulse | None = None) -> None:
        # 发完丢: 抢占失败即丢, 不 reraise, 不做冷静期.
        return

    def attended(self, impulse: Impulse) -> Impulse | None:
        # 运行包: 抬到 INFO (正常运行强度), 区别于挑战用的 BACKGROUND. 纯空 body 保留.
        if not self.is_running():
            return None
        return impulse.model_copy(update={"priority": Priority.INFO})

    def peek(self, no_stale: bool = True) -> Impulse | None:
        if self._impulse is None:
            return None
        impulse = self._impulse
        self._impulse = None  # 发完丢: peek 即清, 一次消费不保留 cache
        if no_stale and impulse.is_stale():
            return None
        return impulse

    def is_running(self) -> bool:
        return self._running

    async def __aenter__(self) -> Self:
        self._running = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        self._running = False


class DoloresEgoNucleusMeta(NucleusMeta):
    """Factory meta — 让 ``moss manifests nuclei`` 可发现 dolores ego 自醒 nucleus."""

    def __init__(
        self,
        *,
        name: str = NAME,
        description: str = "wake moss when the dolores ego agent is running",
    ):
        self._name = name
        self._description = description

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return self._description

    def signals(self) -> Iterable[type[SignalMeta]]:
        yield DoloresEgoSignalMeta

    def factory(self, container: IoCContainer) -> Nucleus:
        logger = container.get(LoggerItf)
        return DoloresEgoNucleus(name=self._name, description=self._description, logger=logger)


def new_dolores_ego_signal(
    *messages: ContextType,
    priority: Priority = Priority.INFO,
    description: str = "",
    hint: str = "",
) -> Signal:
    """Helper — 构造一封自醒 signal, 供 ego 的 turn/start 监听发出."""
    return DoloresEgoSignalMeta().to_signal(
        *messages,
        description=description,
        priority=priority,
        hint=hint,
    )
