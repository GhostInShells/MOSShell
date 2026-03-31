from abc import abstractmethod
from typing import Callable, Protocol
from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.blueprint.builder import new_channel

__all__ = ['ChannelInterface', 'AppChannel']

class ChannelInterface(Protocol):

    @abstractmethod
    def as_channel(self, name: str, description: str) -> Channel:
        channel = new_channel(name=name, description=description)
        ...  # build it with self methods
        return channel


class AppChannel(Protocol):
    """
    定义 Channel 的一种范式.
    将共享的状态, 函数用面向对象的方式来定义.
    同时这个 Channel 提供一个独立的进程运行时, 可以用于渲染图形界面或其它持续性的工作.
    它通过协议自动发现和 Shell 进程的通讯方式.

    本处设计只是开发范式的提示. 具体用法可以发挥想象.
    """

    @abstractmethod
    def as_channel(self) -> Channel:
        channel = new_channel(name='name', description='description')
        # register self method for building
        # channel.build.command(self.method)
        return channel

    @abstractmethod
    def main(self) -> None:
        """
        run the channel in the process
        """
        # start the channel in thread
        cancel = provide_in_thread(self.as_channel())
        # run until process closed
        ...


_CancelFunc = Callable[[], None]


def provide_in_thread(channel: Channel) -> _CancelFunc:
    # todo
    pass
