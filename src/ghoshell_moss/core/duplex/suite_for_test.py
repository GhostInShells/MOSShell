from abc import ABC, abstractmethod
from ghoshell_moss.core.concepts.channel import ChannelProxy, ChannelProvider
from .thread_channel import create_thread_channel

__all__ = ['BridgeTestSuite', 'ThreadBridgeTestSuite']


class BridgeTestSuite(ABC):

    @abstractmethod
    def create(self, proxy_name: str = "proxy") -> tuple[ChannelProvider, ChannelProxy]:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass


class ThreadBridgeTestSuite(BridgeTestSuite):

    def create(self, proxy_name: str = "proxy") -> tuple[ChannelProvider, ChannelProxy]:
        return create_thread_channel(proxy_name)

    def cleanup(self) -> None:
        pass
