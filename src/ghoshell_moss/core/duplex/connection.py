from abc import ABC, abstractmethod

from .protocol import ChannelEvent

__all__ = ["ChannelEvent", "Connection", "ConnectionClosedError", "ConnectionNotAvailable"]


# --- errors --- #


class ConnectionClosedError(Exception):
    """表示 connection 已经连接失败."""

    pass


class ConnectionNotAvailable(Exception):
    pass


class Connection(ABC):
    """
    provider 与 proxy 之间的通讯连接, 用来接受和发布事件.
    provider 持有的应该是 proxyConnection
    而 proxy 持有的应该是 providerConnection.
    但两者的接口目前看起来应该是相似的.
    """

    @abstractmethod
    async def recv(self, timeout: float | None = None) -> ChannelEvent:
        """从通讯事件循环中获取一个事件. proxy 获取的是 provider event, provider 获取的是 proxy event"""
        pass

    @abstractmethod
    async def send(self, event: ChannelEvent) -> None:
        """发送一个事件给远端, proxy 发送的是 proxy event, provider 发送的是 provider event."""
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """判断 connection 是否已经彻底关闭了."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """判断 connection 是否还可以用."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭这个 connection"""
        pass

    @abstractmethod
    async def start(self) -> None:
        """启动这个 connection."""
        pass
