from abc import ABC, abstractmethod
from .protocol import ChannelEvent

__all__ = ['ConnectionClosedError', 'Connection']


# --- errors --- #

class ConnectionClosedError(Exception):
    """表示 connection 已经连接失败. """
    pass


class Connection(ABC):
    """
    Server 与 client 之间的通讯连接, 用来接受和发布事件.
    Server 持有的应该是 ClientConnection
    而 Client 持有的应该是 ServerConnection.
    但两者的接口目前看起来应该是相似的.
    """

    @abstractmethod
    async def recv(self, timeout: float | None = None) -> ChannelEvent:
        """从通讯事件循环中获取一个事件. client 获取的是 server event, server 获取的是 client event"""
        pass

    @abstractmethod
    async def send(self, event: ChannelEvent) -> None:
        """发送一个事件给远端, client 发送的是 client event, server 发送的是 server event."""
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """判断 connection 是否已经彻底关闭了. """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """判断 connection 是否还可以用. """
        pass

    @abstractmethod
    async def close(self) -> None:
        """关闭这个 connection """
        pass

    @abstractmethod
    async def start(self) -> None:
        """启动这个 connection. """
        pass
