"""
Parameter — typed, retained state on the matrix.

A parameter has one declarer (writer) and many subscribers (readers).
The declarer retains the current value and pushes updates; a subscriber
gets the current value once on subscribe, then ongoing pushes.

This supplements Topic: topic is ephemeral broadcast, parameter is retained
state with a protocolized declaration (push over pull).
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable, Type

from pydantic import BaseModel, Field

__all__ = [
    "ParameterModel",
    "ParameterSchema",
    "ParameterDeclaration",
    "ParameterSubscriber",
    "Parameters",
    "T_PARAM",
    "ExampleParameter",
]

T_PARAM = TypeVar("T_PARAM", bound="ParameterModel")


class ParameterSchema(BaseModel):
    """自描述声明 — 用于发现 / 内省."""

    name: str = Field(description="parameter name")
    description: str = Field(description="parameter description")
    json_schema: dict = Field(description="parameter json schema")


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
    def to_parameter_schema(cls) -> ParameterSchema:
        return ParameterSchema(
            name=cls.parameter_key(),
            description=cls.__doc__ or '',
            json_schema=cls.model_json_schema(),
        )


class ExampleParameter(ParameterModel):
    example: str = 'hello world'

    @classmethod
    def parameter_key(cls) -> str:
        return "example"


class ParameterDeclaration(Generic[T_PARAM], ABC):
    """写者 handle — declare 的产物, 持有当前值并 push 给订阅者."""

    @property
    @abstractmethod
    def key(self) -> str:
        """本 parameter 的 key."""
        pass

    @property
    @abstractmethod
    def value(self) -> T_PARAM:
        """当前值 (default 或最近一次 set), 零 IO 本地读."""
        ...

    @abstractmethod
    def set(self, value: T_PARAM) -> None:
        """本地立即生效, 并异步 push 给订阅者 (fire-and-forget)."""
        ...


class ParameterSubscriber(Generic[T_PARAM], ABC):
    """读者 handle — subscribe 的产物, 收推 + 可退订."""

    @property
    @abstractmethod
    def value(self) -> T_PARAM | None:
        """当前订阅值.  None = 声明者不存在 / 尚未推送."""
        ...

    @abstractmethod
    def on_change(
            self, callback: Callable[[T_PARAM], None],
    ) -> Callable[[], None]:
        """每次值到达触发 (含订阅时拉到的初始值). 返回取消该回调."""
        ...

    @abstractmethod
    def close(self) -> None:
        """退订 — 停止 transport 层的 push."""
        ...


class Parameters(ABC):
    """
    matrix 面 parameter 服务: 声明 (成为写者) 与订阅 (成为读者).

    单写者由 declare 构造 — 谁 declare 谁就是唯一源, 无需仲裁.
    subscribe 是点对点: 向某 ``address`` 声明的 parameter 拉一次当前值 + 持续收推.

    Usage::

        decl = await parameters.declare(GhostPersona())
        decl.set(GhostPersona(name="Nova"))

        sub = await parameters.subscribe(GhostPersona, address="...")
        sub.value
        sub.on_change(lambda new: ...)
    """

    @abstractmethod
    async def declare(
            self,
            model: T_PARAM,
            *,
            key: str | None = None,
    ) -> ParameterDeclaration[T_PARAM]:
        """声明本 parameter (成为唯一写者).  ``model`` 携带 default 值."""
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
            address: str | None = None,
    ) -> ParameterSubscriber[T_PARAM]:
        """订阅某 ``address`` 声明的 parameter (点对点).  订阅时拉一次当前值, 之后收推."""
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
