from typing import Callable
from typing_extensions import Self
from ghoshell_moss.contracts.workspace import Storage
from .mindflow import Signal, SignalMeta, InputSignal
from typing import Iterable, Literal
from abc import ABC, abstractmethod
from ghoshell_moss.message import Message, Addition
from pydantic import BaseModel, Field
from PIL.Image import Image

Role = Literal['system', 'logos', 'log', 'error', 'task']


class OutputItem(BaseModel):
    """
    可以用于输出的数据结构.
    以 Message 为基础.
    """
    role: str | Role = Field(
        default='log',
        description="消息的类型.",
    )
    log: str = Field(
        default="",
        description="some log information.",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description='messages',
    )

    @classmethod
    def new(cls, role: Role | str, *messages: Message, log: str = '') -> Self:
        if isinstance(role, str):
            return cls.model_construct(role=role, messages=[], log=log).with_messages(*messages)
        else:
            return cls(role=role).with_messages(*messages)

    def with_messages(self, *messages: Message | str) -> Self:
        for msg in messages:
            # 接受字符串处理后的消息.
            if isinstance(msg, str):
                self.messages.append(Message.new().with_content(msg))
            else:
                self.messages.append(msg.compact())
        return self


class OutputBuffer(ABC):

    @abstractmethod
    def close(self) -> None:
        """关闭 buffer"""
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """是否关闭"""
        pass

    @abstractmethod
    def add_output(self, item: OutputItem) -> None:
        """添加 item, 需要实现线程安全. """
        pass

    @abstractmethod
    def values(self) -> Iterable[OutputItem]:
        """返回所有的 items. 会生成一个线程安全的快照. """
        pass

    @abstractmethod
    def updated_at(self) -> float:
        """最后更新的 timestamp"""
        pass


class Session(ABC):
    """
    MOSS 运行时当前的连接状态.
    """

    @property
    @abstractmethod
    def session_scope(self) -> str:
        """
        所属的会话 scope
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
    def output(self, role: str | Role, *messages: Message | str) -> None:
        """
        输出消息给 moss 共享 session 的终端.
        """
        pass

    @abstractmethod
    def on_output(self, callback: Callable[[OutputItem], None]) -> None:
        """
        输出回调监听 conversation item.
        可以用来做个什么渲染.
        """
        pass

    @abstractmethod
    def output_buffer(
            self,
            maxsize: int = 100,
    ) -> OutputBuffer:
        """生产一个 OutputBuffer"""
        pass
