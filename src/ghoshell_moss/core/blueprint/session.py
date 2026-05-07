from typing import Callable
from typing_extensions import Self
from ghoshell_moss.contracts.workspace import Storage
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.blueprint.mindflow import Signal, SignalMeta, InputSignal
from typing import Iterable, Literal
from abc import ABC, abstractmethod
from ghoshell_moss.message import Message
from pydantic import BaseModel, Field
from PIL.Image import Image

Role = Literal['system', 'logos', 'log', 'error', 'task']


class OutputItem(BaseModel):
    """
    可以用于输出的原子化数据结构.

    是整个系统的对外输出.
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
    MOSS 运行时当前的通讯总线.

    todo:
        1. 实现一个 stream[bytes] 发送不同 key 的首包/间包/尾包
        2. 实现系统级的 logos 流监听?
        3. 实现共享的 parameters
        4. 实现可注册的基于 key 的函数.
    """

    @property
    @abstractmethod
    def session_scope(self) -> str:
        """
        所属的会话 scope
        """
        pass

    @property
    @abstractmethod
    def session_id(self) -> str:
        """
        session id
        """
        pass

    @abstractmethod
    def add_signal(self, signal: Signal) -> None:
        """
        input a mindflow signal to the Session
        """
        pass

    def add_input_signal(
            self,
            *values: str | Image | Message,
            description: str = '',
            priority: int | None = None,
            meta: SignalMeta | None = None,
            stale_timeout: float = 0,
    ) -> None:
        """
        easy way to add a default input signal to the Mindflow
        """
        meta = meta or InputSignal()
        signal = meta.to_signal(
            *values,
            description=description,
            priority=priority,
            stale_timeout=stale_timeout,
        )
        self.add_signal(signal)

    @abstractmethod
    def on_signal(self, callback: Callable[[Signal], None]) -> None:
        """
        listen to the MOSS input signal
        """
        pass

    @property
    @abstractmethod
    def topics(self) -> TopicService:
        """
        基于 Topic 概念的服务.
        """
        pass


    @property
    @abstractmethod
    def storage(self) -> Storage:
        """
        session 专属的 storage.
        需要的话可以将文件作为通讯方式.
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
