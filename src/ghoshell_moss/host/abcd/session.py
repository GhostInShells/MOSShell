from typing import Generic, TypeVar, Any, Callable
from typing_extensions import Self
from abc import ABC, abstractmethod
from ghoshell_moss.contracts.workspace import Storage
from ghoshell_moss.message import Message
from pydantic import BaseModel, Field
from ghoshell_common.helpers import uuid


class ConversationItem(BaseModel):
    """
    可以用于输出的某种数据结构.
    暂时不与 AI 模型强耦合. 仅仅用于做 MOSS 命令行交互界面的输出.
    """
    id: str = Field(
        default_factory=uuid,
        description="conversation unique id",
    )
    role: str = Field(description="描述消息的角色")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="关于这个 item 的元信息.",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="一组消息体"
    )

    def with_message(self, *messages: Message | str) -> Self:
        for msg in messages:
            if isinstance(msg, Message):
                self.messages.append(msg)
            elif isinstance(msg, str):
                self.messages.append(Message.new().with_content(msg))
        return self


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
