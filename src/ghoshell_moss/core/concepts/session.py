from typing import Callable
from ghoshell_moss.contracts.workspace import Storage
from .mindflow import Signal, SignalMeta, InputSignal
import asyncio
from typing import Any, Iterable, Literal
from typing_extensions import Self
from abc import ABC, abstractmethod
from ghoshell_moss.message import Message, WithAdditional
from pydantic import BaseModel, Field, AwareDatetime
from ghoshell_common.helpers import uuid
from datetime import datetime
from dateutil import tz
from PIL.Image import Image

Role = Literal['perception', 'logos', 'log']


class ConversationItem(BaseModel, WithAdditional):
    """
    可以用于输出的某种数据结构.
    暂时不与 AI 模型强耦合. 仅仅用于做 MOSS 命令行交互界面的输出.
    """
    id: str = Field(
        default_factory=uuid,
        description="conversation unique id",
    )
    role: Role = Field(
        default='log',
        description="消息的类型.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="关于这个 item 的元信息.",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="一组消息体"
    )

    @classmethod
    def new(cls, role: Role, **metadata: dict) -> Self:
        return cls(role=role, metadata=metadata)

    def to_json(self) -> str:
        return self.model_dump_json(indent=0, ensure_ascii=False, exclude_defaults=True, exclude_none=True)

    def with_message(self, *messages: Message | str | Image) -> Self:
        for msg in messages:
            if isinstance(msg, Message):
                self.messages.append(msg)
            else:
                self.messages.append(Message.new().with_content(msg))
        return self


class ConversationMeta(BaseModel):
    id: str = Field(
        default_factory=uuid,
        description="conversation unique id",
    )
    session_id: str = Field(
        default='',
        description="conversation created in which session",
    )
    root_id: str = Field(
        default='',
        description="the root id of the conversation tree",
    )
    fork_from: str = Field(
        default='',
        description="the parent conversation id that the current one fork from",
    )
    recap: str = Field(
        default='',
        description="the recap info of the parent conversation",
    )
    title: str = Field(
        default='',
        description="the title of the conversation",
    )
    description: str = Field(
        default='',
        description="the short description of the conversation",
    )
    items_total: int = Field(
        default=0,
        description="the total number of items in the conversation",
    )
    created: AwareDatetime = Field(
        default_factory=lambda: datetime.now(tz.gettz()),
        description="the time when the conversation was created",
    )
    updated: AwareDatetime = Field(
        default_factory=lambda: datetime.now(tz.gettz()),
        description="the time when the conversation was updated",
    )


class Conversation(ABC):

    @property
    @abstractmethod
    def id(self) -> str:
        """
        记录 id.
        """
        pass

    @abstractmethod
    def meta(self) -> ConversationMeta:
        pass

    @abstractmethod
    def items(self) -> Iterable[ConversationItem]:
        """
        返回所有的 Items, 并且合并同类型的 Items.
        """
        pass

    @abstractmethod
    def append(self, *items: ConversationItem) -> asyncio.Future[None]:
        """
        保存当前的 items.
        底层逻辑实现要考虑异步安全性.
        """
        pass

    @abstractmethod
    async def compact(self) -> Self:
        """
        压缩上下文, 同时会 fork 一个新的 conversation.
        """
        pass


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
