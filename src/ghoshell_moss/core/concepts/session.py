from typing import Callable
from ghoshell_moss.contracts.workspace import Storage
from .mindflow import Signal, SignalMeta, InputSignal
import asyncio
from typing import Any, Iterable, Literal
from typing_extensions import Self
from abc import ABC, abstractmethod
from ghoshell_moss.message import Message, WithAdditional, Addition
from pydantic import BaseModel, Field, AwareDatetime
from ghoshell_common.helpers import uuid
from datetime import datetime
from dateutil import tz
from PIL.Image import Image

Role = Literal['perception', 'logos', 'log']


class OutputItem(Addition):
    """
    可以用于输出的某种数据结构.
    暂时不与 AI 模型强耦合. 仅仅用于做 MOSS 命令行交互界面的输出.
    """
    role: str = Field(
        default='log',
        description="消息的类型.",
    )
    session_id: str = Field(

    )

    @classmethod
    def keyword(cls) -> str:
        return 'session/output'


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
    def output(self, *items: OutputItem) -> None:
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
