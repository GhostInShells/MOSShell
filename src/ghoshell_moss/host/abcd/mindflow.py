import asyncio
import time
from abc import ABC, abstractmethod
from typing import Callable, Any
from typing_extensions import Self
from pydantic import BaseModel, Field, AwareDatetime, ValidationError
from ghoshell_moss.message import Message
from ghoshell_common.helpers import uuid
from PIL.Image import Image
import datetime
import dateutil

Priority = int
SignalName = str


class Signal(BaseModel):
    name: SignalName = Field(
        description="the signal name, if not match any mind pulse, the signal will be ignore"
    )
    priority: int = Field(
        default=0,
        description="信号的优先级, 越大优先级越高. 用于做抢占式调度. 来自边缘系统的输入本身应包含第一轮优先级"
    )
    trace: str = Field(
        default_factory=uuid,
        description="trace of the signal name",
    )
    description: str = Field(
        default='',
        description="short description of the signal",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="被处理过的消息体.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="meta data of the signal follow the protocol of the name",
    )
    stale_timeout: float = Field(
        default=0,
    )
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.datetime.now(dateutil.tz.gettz()),
    )

    @classmethod
    def new(
            cls,
            name: SignalName,
            *messages: Message,
            priority: int = 0,
            description: str = '',
            metadata: dict[str, Any] | None = None,
            stale_timeout: float = 0,
    ) -> Self:
        return cls(
            name=name,
            messages=list(messages),
            priority=priority,
            description=description,
            metadata=metadata or {},
            stale_timeout=stale_timeout,
        )

    def is_stale(self) -> bool:
        if self.stale_timeout <= 0:
            return False
        delta = time.time() - self.created_at.timestamp()
        return delta > self.stale_timeout

    def to_json(self) -> str:
        return self.model_dump_json(indent=0, exclude_none=True, exclude_defaults=True, ensure_ascii=False)


class SignalMeta(BaseModel, ABC):
    """
    to define a signal protocol.
    """

    @classmethod
    @abstractmethod
    def signal_name(cls) -> SignalName:
        pass

    @classmethod
    @abstractmethod
    def priority(cls) -> Priority:
        pass

    @classmethod
    def from_signal(cls, signal: Signal) -> Self | None:
        if cls.signal_name() != signal.name:
            return None
        try:
            metadata = signal.metadata
            return cls.model_validate(metadata)
        except ValidationError:
            return None

    def to_signal(
            self,
            *messages: Message | str | Image,
            description: str = '',
            stale_timeout: float = 0,
            priority: int | None = None,
    ) -> Signal:
        name = self.signal_name()
        wrapped_messages = []
        for msg in messages:
            if isinstance(msg, Image):
                wrapped_messages.append(Message.new().with_content(msg))
            elif isinstance(msg, str):
                wrapped_messages.append(Message.new().with_content(msg))
            elif isinstance(msg, Message):
                wrapped_messages.append(msg)
        priority = self.priority() if priority is None else priority
        # 使用 model construct 不做类型校验.
        return Signal.model_construct(
            name=name,
            messages=wrapped_messages,
            metadata=self.model_dump(exclude_defaults=True, exclude_none=True),
            description=description,
            stale_timeout=stale_timeout,
            priority=priority,
        )


class InputSignal(SignalMeta):
    """
    basic input.
    """

    @classmethod
    def signal_name(cls) -> SignalName:
        return 'moss/input'

    @classmethod
    def priority(cls) -> Priority:
        return 0


class Impulse(BaseModel):
    """
    priority impulse for Superior AI mindflow to handle.
    """
    id: str = Field(
        default_factory=uuid,
        description="the impulse id",
    )
    belongs_to: str = Field(
        description="belongs to which MindPulse"
    )
    trace: str = Field(
        default='',
        description="trace of the impulse name",
    )
    priority: int = Field(
        default=0,
        description="the impulse priority",
    )
    description: str = Field(
        default='',
        description="shot description of the newest impulse",
    )
    messages: list[Message] = Field(
        default_factory=list,
        description="the impulse messages",
    )
    instruction: str = Field(
        default='',
        description="the instruction to handle the impulse",
    )
    created_at: AwareDatetime = Field(
        default_factory=lambda: datetime.datetime.now(dateutil.tz.gettz()),
        description="the creation time of the impulse",
    )
    stale_timeout: float = Field(
        default=0,
        description="stale timeout of the impulse",
    )

    def is_stale(self) -> bool:
        if self.stale_timeout <= 0:
            return False
        delta = time.time() - self.created_at.timestamp()
        return delta > self.stale_timeout

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, Impulse):
            raise TypeError('Comparing value must be of type Impulse')
        if self.priority < other.priority:
            return True
        elif self.priority == other.priority:
            return self.created_at > other.created_at
        return False

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, Impulse):
            raise TypeError('Comparing value must be of type Impulse')
        if self.priority > other.priority:
            return True
        elif self.priority == other.priority:
            return self.created_at < other.created_at
        return False


class MindPulse(ABC):
    """
    并行思维的单一节点. 处理输入信号, 同时调整状态.
    """

    @abstractmethod
    def name(self) -> str:
        """
        identity of the mind pulse
        """
        pass

    @abstractmethod
    def description(self) -> str:
        """
        simple description of the mind pulse
        """
        pass

    @abstractmethod
    def on_signal(self, signal: Signal) -> None:
        """
        receive new signal from Mindflow signal bus.
        quickly receive, handle signa asynchronously
        """
        pass

    @abstractmethod
    def receiving(self) -> list[SignalName]:
        """
        manifest the receiving signals of this mind pulse.
        """
        pass

    @abstractmethod
    def peek(self) -> Impulse | None:
        """
        peek the current impulse, useful for mindflow to:
        1. rank priority
        2. list the mindflow status
        """
        pass

    @abstractmethod
    def pop_impulse(self) -> Impulse | None:
        """
        pop last impulse from MindPulse
        """
        pass

    @abstractmethod
    def with_bus(self, impulse_notify: Callable[[Impulse], None], signal_bus: Callable[[Signal], None]) -> None:
        """
        Register mindflow signal and impulse bus,
        When MindPulse emerge a new Impulse, shall notify mindflow, but not pop yet.
        """
        pass

    @abstractmethod
    def supress(self, other_impulse: Impulse) -> None:
        """
        Suppress the current impulse due to other prior mind impulse by Mindflow.
        Shall remove the impulse first, then decide to decay or escalation by mind pulse itself asynchronously.
        """
        pass

    @abstractmethod
    async def start(self) -> None:
        """
        start the MindPulse with it inner async loop task to handle signal.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """
        stop he MindPulse.
        """
        pass


class Mindflow(ABC):
    """
    The parallel think unit to handle outside input signal.
    """

    @abstractmethod
    def with_pulse(self, pulse: MindPulse) -> Self:
        """
        注册必要的节点.
        """
        pass

    @abstractmethod
    def pulses(self) -> dict[str, MindPulse]:
        """
        mapping the MindPulse by name
        """
        pass

    @abstractmethod
    def context(self) -> str:
        """
        the context message of all MindPulse
        """
        pass

    @abstractmethod
    def on_signal(self, signal: Signal) -> None:
        """
        接受信号, 调度或者丢弃.
        """
        pass

    @abstractmethod
    def set_impulse(self, impulse: Impulse) -> None:
        """
        receive new impulse, trigger the AI thinking or action.
        """
        pass

    @abstractmethod
    def wait_impulse(self, *, priority: int = -1, wait_new: bool = False) -> asyncio.Future[Impulse]:
        """
        wait any new impulse prior than the priority. will notify when:
        0. Some Unhandled prior impulse already exists.
        1. new impulse emerge from a MindPulse

        the method will not pop the impulse, just peek it.
        the wait process is always cancellable
        """
        pass

    @abstractmethod
    def pop_impulse(self, pulse_name: str | None) -> Impulse | None:
        """
        pop an impulse from all the mind pulse or the specific one.
        """
        pass

    @abstractmethod
    async def __aenter__(self):
        """
        启动 mindflow, 并行运行 MindPulse 节点.
        """
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        关闭 Mindflow 和所有的 MindPulse.
        """
        pass
