"""
Memory transport for Parameters — 单进程内存总线.

用于参考实现与测试: host 与 worker 在同一进程内, 通过一根共享的 ``MemoryBus`` 互通.
没有真实网络, 也没有跨线程 — 但复用与 zenoh 相同的 ``ParametersBroadcaster`` 接口,
所以收敛逻辑 (host 采纳 / worker 收真值 / host 化身收敛) 可以离线断言.
"""

from typing import Callable

from ghoshell_moss.core.blueprint.parameter import ParameterData
from ghoshell_moss.core.parameter._base import ParametersBroadcaster

__all__ = ["MemoryBus", "MemoryParametersBroadcaster"]

_Key = str
_Address = str
_Epoch = str


class MemoryBus:
    """进程内共享总线 — 相当于 zenoh session. 无生命周期, 只承载回调路由."""

    def __init__(self):
        self._truth_subscribers: dict[_Key, set[Callable[[ParameterData], None]]] = {}
        self._declaration_subscribers: set[Callable[[ParameterData], None]] = set()
        self._host_alive_subscribers: set[Callable[[_Address, _Epoch], None]] = set()
        self._host: tuple[_Address, _Epoch] | None = None


class MemoryParametersBroadcaster(ParametersBroadcaster):
    """单节点到 MemoryBus 的 transport handle. host 与 worker 各持一个实例, 共享一根总线."""

    def __init__(self, bus: MemoryBus):
        self._bus = bus

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None

    # -- 真值通道 --------------------------------------------------------

    async def subscribe_host_truth(
            self, key: _Key, callback: Callable[[ParameterData], None],
    ) -> Callable[[], None]:
        subscribers = self._bus._truth_subscribers.setdefault(key, set())
        subscribers.add(callback)

        def dispose() -> None:
            subscribers.discard(callback)

        return dispose

    async def publish_host_truth(self, parameter: ParameterData) -> None:
        for callback in list(self._bus._truth_subscribers.get(parameter.key, ())):
            callback(parameter)

    # -- 声明通道 --------------------------------------------------------

    async def publish_declaration(self, parameter: ParameterData) -> None:
        for callback in list(self._bus._declaration_subscribers):
            callback(parameter)

    async def subscribe_declarations(
            self, callback: Callable[[ParameterData], None],
    ) -> Callable[[], None]:
        self._bus._declaration_subscribers.add(callback)

        def dispose() -> None:
            self._bus._declaration_subscribers.discard(callback)

        return dispose

    # -- host 化身 --------------------------------------------------------

    def set_host(self, address: _Address, epoch: _Epoch) -> None:
        self._bus._host = (address, epoch)
        for callback in list(self._bus._host_alive_subscribers):
            callback(address, epoch)

    async def on_host_alive(self, callback: Callable[[_Address, _Epoch], None]) -> None:
        self._bus._host_alive_subscribers.add(callback)
        # 晚启动的 worker 也要立刻知道当前 host (回放当前化身).
        if self._bus._host is not None:
            callback(*self._bus._host)
