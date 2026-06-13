"""CommandNucleus — 把 ``command`` signal 直接转成 ``command_only`` impulse.

四元 nucleus 之一: 配对 ``ImpulsePrimitive.command_only`` 的反射弧入口.
监听 ``CommandSignalMeta`` (signal name = ``"command"``), 把 signal 携带的 logos
直接卸载为 impulse, 走 mindflow 的 ``thinking_effort='none'`` 提前返回路径,
在不调用 ``ghost.articulate()`` 的情况下让 shell 执行指令.

priority 完全继承 ``Signal.priority`` — nucleus 不强制 floor, 调用方用
``Priority.FATAL`` 表达 "强制指令" (等价于 ``ImpulsePrimitive.fatal_command``),
用 ``Priority.NOTICE`` 表达 "普通命令". 单一 nucleus 覆盖两种.
"""
from typing import Self, Callable, Iterable

from ghoshell_container import IoCContainer
from pydantic import Field

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.message import ContextType
from ghoshell_moss.core.blueprint import Impulse
from ghoshell_moss.core.blueprint.mindflow import (
    SignalMeta, SignalName, Priority, Signal,
    Nucleus, NucleusMeta, ImpulsePrimitive,
)

__all__ = ['CommandNucleus', 'CommandSignalMeta', 'CommandNucleusMeta']


class CommandSignalMeta(SignalMeta):
    """Signal meta for ``command`` — carries the logos to be executed.

    使用 ``signal.metadata`` 携带 ``logos`` 字段, 反序列化通过 SignalMeta
    标准路径 (``from_signal`` -> ``model_validate(signal.metadata)``).
    """

    logos: str = Field(
        description="待 shell 直接执行的 logos (通常是 CTML). "
                    "由 CommandNucleus 把它从 signal.metadata 抽出, "
                    "卸载到 Impulse.logos 字段, 经 attention.command_logos 送入 shell.",
    )

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'command'

    @classmethod
    def priority(cls) -> Priority:
        return Priority.NOTICE


class CommandNucleus(Nucleus):
    """Reflex-arc nucleus — turns each ``command`` signal into a ``command_only``
    impulse, caches it as last-impulse for mindflow rank/challenge pull.

    Last-impulse cache 模式 (与 ``_DirectImpulseNucleus`` 同构):
    - ``add_signal`` 立即构造 impulse, 写入 ``_impulse`` cache 并通知 mindflow
    - mindflow ``_rank_nuclei`` 通过 ``peek()`` 拉取, 仲裁后调 ``pop_impulse``
      清 cache
    - 连续两条 signal 进入但 mindflow 未消费时, 后者覆盖前者 (last-wins) —
      command 的语义是"最新指令为准", 旧的过时

    与 ``InputSignalNucleus`` / ``SilentNucleus`` 的区别: 不聚合, 不保留历史,
    每次 add 都覆盖. command 视为离散事件, 多个未消费的 command 合并无意义.

    priority 完全继承 ``Signal.priority`` (不设 floor) — 调用方用
    Priority.FATAL 等价于 ``ImpulsePrimitive.fatal_command``.
    """

    NAME = 'command_nucleus'

    def __init__(self, *, name: str = NAME, logger: LoggerItf | None = None):
        self._name = name
        self._impulse_notify: Callable[[Impulse], None] | None = None
        self._is_running = False
        self._logger = logger or get_moss_logger()
        # Last-impulse cache: 满足 mindflow pull-based 协议.
        self._impulse: Impulse | None = None

    def name(self) -> str:
        return self._name

    def description(self) -> str:
        return 'send logos directly to the shell, bypassing ghost thinking'

    def status(self) -> str:
        return ''

    def signals(self) -> list[SignalName]:
        return [CommandSignalMeta.signal_name()]

    def clear(self) -> None:
        self._impulse = None

    def add_signal(self, signal: Signal) -> None:
        if not self._is_running:
            return
        impulse = self.build_impulse(signal)
        if impulse is None:
            return
        # Last-wins cache: 覆盖未消费的旧 impulse.
        self._impulse = impulse
        if self._impulse_notify:
            self._impulse_notify(impulse)

    def build_impulse(self, signal: Signal) -> Impulse | None:
        meta = CommandSignalMeta.from_signal(signal)
        if meta is None or not meta.logos:
            return None
        impulse = Impulse.from_signal(signal, source=self.name())
        # 继承 signal.priority 不再强制 floor — 调用方用 Priority.FATAL
        # 等价于 ImpulsePrimitive.fatal_command.
        return ImpulsePrimitive.command_only(impulse, meta.logos)

    def with_bus(
            self,
            signal_broadcast: Callable[[Signal], None],
            impulse_notify: Callable[[Impulse], None],
    ) -> None:
        self._impulse_notify = impulse_notify

    def suppress(self, suppress_by: Impulse) -> None:
        # command 抢占失败 (priority 不够) → 清 cache, 让位.
        # command 没有"重试" 语义 — 失败就丢, 等下一条新 command.
        self._impulse = None

    def pop_impulse(self, impulse: Impulse) -> None:
        if self._impulse is impulse:
            self._impulse = None

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


class CommandNucleusMeta(NucleusMeta):
    """Factory meta — lets ``moss manifests nuclei`` discover CommandNucleus."""

    def name(self) -> str:
        return CommandNucleus.NAME

    def description(self) -> str:
        return 'reflex-arc nucleus that turns command signals into command_only impulses'

    def signals(self) -> Iterable[type[SignalMeta]]:
        yield CommandSignalMeta

    def factory(self, container: IoCContainer) -> Nucleus:
        logger = container.get(LoggerItf)
        return CommandNucleus(logger=logger)


def new_command_signal(
        logos: str,
        *messages: ContextType,
        priority: Priority = Priority.NOTICE,
        description: str = '',
        stale_timeout: float = 0,
) -> Signal:
    """Helper — construct a ``command`` signal from raw logos.

    Convenience over ``CommandSignalMeta(logos=...).to_signal(...)`` for
    one-liner injection in tests / scripts.
    """
    return CommandSignalMeta(logos=logos).to_signal(
        *messages,
        description=description,
        stale_timeout=stale_timeout,
        priority=priority,
    )
