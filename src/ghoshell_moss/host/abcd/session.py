from typing import Callable
from abc import ABC, abstractmethod
from ghoshell_moss.contracts.workspace import Storage
from ghoshell_moss.message import Message
from PIL.Image import Image
from .mindflow import Signal, SignalMeta, InputSignal
from .conversation import ConversationItem


class Session(ABC):
    """
    MOSS 运行时当前的连接状态.
    """

    @property
    @abstractmethod
    def session_id(self) -> str:
        """
        所属的会话 id
        """
        pass

    @abstractmethod
    def input(self, signal: Signal) -> None:
        """
        input a signal to the MOSS session.
        """
        pass

    def add_input(
            self,
            *values: str | Image | Message,
            description: str = '',
            priority: int | None = None,
            meta: SignalMeta | None = None,
            stale_timeout: float = 0,
    ) -> None:
        """
        easy way to add a signal to the MOSS session.
        """
        meta = meta or InputSignal()
        signal = meta.to_signal(
            *values,
            description=description,
            priority=priority,
            stale_timeout=stale_timeout,
        )
        self.input(signal)

    @abstractmethod
    def on_input(self, callback: Callable[[Signal], None]) -> None:
        """
        listen to the MOSS input signal
        """
        pass

    @property
    @abstractmethod
    def storage(self) -> Storage:
        """
        session 专属的 storage.
        """
        pass

    @abstractmethod
    def output(self, *items: ConversationItem) -> None:
        """
        输出消息给 moss 共享 session 的终端.
        """
        pass

    @abstractmethod
    def on_output(self, callback: Callable[[ConversationItem], None]) -> None:
        """
        输出回调监听 conversation item.
        可以用来做个什么渲染.
        """
        pass
