"""
Parameter — typed, retained state shared across the network.

A parameter has one **truth** (hosted by the network's host node) and many
readers.  Workers report local writes as *declarations*; the host adopts-or-
rejects each declaration and always broadcasts a truth in response.  When no
host is present, a worker's local value is the truth.

Keys are address-free — sender identity travels in the payload's
``ParameterMeta`` (like ``TopicMeta.sender``), never in the key.  The logical
ordering lives in ``ParameterData.version`` (three values, see below).
"""

import time
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable, Type, Any

from pydantic import BaseModel, Field

__all__ = [
    "ParameterModel",
    "ParameterSchema",
    "ParameterDeclaration",
    "ParameterSubscriber",
    "ParameterData",
    "ParameterMeta",
    "Parameters",
    "T_PARAM",
    "REQUIRE_ONLY",
    "FIRST_PACKET",
    "ExampleParameter",
]

T_PARAM = TypeVar("T_PARAM", bound="ParameterModel")

# ParameterData.version 三值 — 全系统的真值序.
#   -1  仅要求广播一次 (require): host 不采纳, 回播当前真值一次
#    0  首包: 仅在 host 尚无真值时被采纳 (铸为 >=1); 否则等同于 require
#   >=1 真值声明: host 在版本不低时采纳, 否则回播当前真值一次
REQUIRE_ONLY = -1
FIRST_PACKET = 0


class ParameterSchema(BaseModel):
    """自描述声明 — 用于发现 / 内省."""

    name: str = Field(description="parameter name")
    description: str = Field(description="parameter description")
    json_schema: dict = Field(description="parameter json schema")


class ParameterMeta(BaseModel):
    """Wire identity of a parameter datum.

    ``address`` identifies the sender — provenance only, never used for routing.
    ``host`` marks whether the datum is a truth emitted by the host node.
    """

    address: str = Field(description="address of the node that emitted this datum")
    host: bool = Field(description="True when emitted by the host node (truth)")


class ParameterData(BaseModel):
    """A single parameter datum on the wire — the unit of declaration and truth broadcast.

    ``version`` is the logical ordering (not a wall-clock):
      - ``-1`` (REQUIRE_ONLY)  request one truth broadcast; never adopted
      - ``0``  (FIRST_PACKET)  offer as truth only if none exists yet
      - ``>=1``                 a truth declaration; adopted when not older
    """

    meta: ParameterMeta = Field(description="sender identity / truth flag")
    version: int = Field(default=FIRST_PACKET, description="logical version; see class docstring")
    epoch: str = Field(default="", description="host address that stamped this truth (unique per incarnation)")
    key: str = Field(description="parameter key")
    payload: dict[str, Any] = Field(description="parameter data")
    created: float = Field(default_factory=lambda: round(time.time(), 4), description="when this datum was created")


class ParameterModel(BaseModel, ABC):
    """
    自描述 parameter 声明.

    子类定义一个 typed parameter.  ``parameter_key()`` 是默认 key, 也是跨进程
    对齐"同一个 parameter"的协议标识.

    Usage::

        class GhostPersona(ParameterModel):
            name: str = "Echo"

            @classmethod
            def parameter_key(cls) -> str:
                return "ghost_persona"
    """

    @classmethod
    @abstractmethod
    def parameter_key(cls) -> str:
        """默认 key — 每子类唯一, 声明 / 订阅都靠它对齐."""
        pass

    @classmethod
    def default(cls) -> "ParameterModel":
        """
        从类还原一个默认实例 — 订阅侧构建初始值 / schema 的契约点.

        默认 ``cls()``; 子类需要非默认构造时覆盖. 这是 code as prompt 的强制点,
        避免订阅方隐式假设"无参可构造".
        """
        return cls()

    @classmethod
    def to_parameter_schema(cls) -> ParameterSchema:
        return ParameterSchema(
            name=cls.parameter_key(),
            description=cls.__doc__ or '',
            json_schema=cls.model_json_schema(),
        )

    def to_parameter_data(self, meta: ParameterMeta) -> ParameterData:
        return ParameterData(
            meta=meta,
            payload=self.model_dump(mode='json', exclude_none=True),
            key=self.parameter_key(),
            version=FIRST_PACKET,
        )


class ExampleParameter(ParameterModel):
    example: str = 'hello world'

    @classmethod
    def parameter_key(cls) -> str:
        return "example"


class ParameterSubscriber(Generic[T_PARAM], ABC):
    """读者 — subscribe 的产物, 收网络真值 + 可退订."""

    @property
    @abstractmethod
    def value(self) -> T_PARAM:
        """最新本地值 — 乐观, 未必已被 host 确认."""
        ...

    @abstractmethod
    def on_change(
            self, callback: Callable[[T_PARAM], None],
    ) -> Callable[[], None]:
        """网络真值到达并通过闸口时触发. 本地乐观 set 不触发."""
        ...

    @abstractmethod
    def get_truth(self, use_default: bool = False) -> T_PARAM | None:
        """已确认的网络真值. 无真值且 ``use_default=False`` 时返回 None."""
        ...

    @abstractmethod
    def get_truth_data(self, use_default: bool = False) -> ParameterData | None:
        """已确认真值的原始数据 (含 version / epoch)."""
        ...

    @abstractmethod
    def on_truth(self, callback: Callable[[ParameterData], None]) -> Callable[[], None]:
        """网络确认后的真值到达时回调 (携带原始 ParameterData)."""
        ...

    @abstractmethod
    async def wait_first_truth(self) -> T_PARAM:
        """阻塞等待到拿到第一个真值; host 不存在则遥遥无期 (调用方以 wait_for 负超时).
        在拿到真值前被 close() 则抛异常."""
        ...

    @abstractmethod
    def parameter_schema(self) -> ParameterSchema:
        """返回监听的 parameter schema."""
        ...

    @abstractmethod
    def is_closed(self) -> bool:
        ...

    @abstractmethod
    def close(self) -> None:
        """退订 — 停止 transport 层的 push."""
        ...


class ParameterDeclaration(ParameterSubscriber[T_PARAM], ABC):
    """写者 handle — declare 的产物. 上报本地写, 由 host 定序后广播回真值."""

    @property
    @abstractmethod
    def key(self) -> str:
        """本 parameter 的 key."""
        pass

    @property
    @abstractmethod
    def meta(self) -> ParameterMeta:
        """自身所处运行时的身份声明."""
        ...

    @abstractmethod
    def set(self, value: T_PARAM) -> None:
        """本地立即生效, 并上报 host (fire-and-forget). 不触发 on_change."""
        ...


class Parameters(ABC):
    """
    网络通讯中共享状态的服务.

    单点 host 持有真值并广播变更; worker 读真值、上报本地写. host 不在线时,
    以本地值为真相. key 与 address 无关 — 来源身份走 payload 的 ``ParameterMeta``.
    """

    @abstractmethod
    def is_host(self) -> bool:
        """本节点是否为真相宿主."""
        ...

    @abstractmethod
    async def declare(
            self,
            model: T_PARAM,
            *,
            key: str | None = None,
    ) -> ParameterDeclaration[T_PARAM]:
        """声明本 parameter.  ``model`` 携带默认值."""
        ...

    @abstractmethod
    def declared(self) -> list[ParameterSchema]:
        """已声明的 parameter schema 列表 (内省)."""
        ...

    @abstractmethod
    async def subscribe(
            self,
            model: Type[T_PARAM],
            *,
            key: str | None = None,
    ) -> ParameterSubscriber[T_PARAM]:
        """订阅本 key 的网络真值."""
        ...

    @abstractmethod
    def is_running(self) -> bool:
        ...

    @abstractmethod
    async def __aenter__(self) -> 'Parameters':
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...
