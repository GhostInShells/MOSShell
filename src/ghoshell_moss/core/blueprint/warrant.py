"""Warrant — Matrix 级通用授权机制抽象.

三层职责分离: qa 是交互协议, warrant 是存储 + 装线, permission 是业务逻辑.
设计见 `.ai_partners/features/workstreams/2026/08/warrant/FEATURE.md`.
"""

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from typing_extensions import Self

from pydantic import BaseModel

from ghoshell_moss.core.concepts.qa import Answer, Question


PermissionState = TypeVar("PermissionState", bound=BaseModel)


class AuthorizationResult(BaseModel):
    """授权结果. 最小对象, 只有拒绝的自然语言描述.

    reason = None 表示通过 (放行); str 表示拒绝理由.
    """

    reason: str | None = None


class Permission(ABC, Generic[PermissionState]):
    """授权场景的业务逻辑. 纯逻辑, 无 IO.

    静态授权参数在 __init__ 里配置; state 是 warrant 读回的动态授权状态.
    permission 只决定逻辑, 不做持久化.
    """

    @property
    @abstractmethod
    def namespace(self) -> str:
        """审批问题发往的 qa namespace."""
        ...

    @abstractmethod
    def default(self) -> PermissionState:
        """无存储时的初始授权状态."""
        ...

    @abstractmethod
    def check(self, state: PermissionState) -> Question | None:
        """根据当前 state 判断是否需要授权.

        None = 无需授权; Question = 构造好的完整审批问题 (由 warrant 接线发出).
        """
        ...

    @abstractmethod
    def replied(self, answer: Answer) -> tuple[PermissionState, AuthorizationResult, bool]:
        """解释应答, 返回 (新 state, 结果, 是否更新存储).

        save 标志由 permission 的业务判断决定 (如 grant 持久化, deny 不持久化).
        """
        ...


class Warrant(ABC):
    """执行门: 存储 + 装线. 唯一 IO 面.

    可选能力, 从 IoC 取; 拿不到视为放行 (fail-open).
    """

    @abstractmethod
    async def __aenter__(self) -> Self:
        """创建生命周期对象 (存储/QA 协调), 存储动作在其异步 task 里执行."""
        ...

    @abstractmethod
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: Any,
    ) -> None:
        """协调/取消生命周期; 取消随调用方 scope 传播 (cancel question 同路径)."""
        ...

    @abstractmethod
    async def require(self, permission: Permission[Any]) -> AuthorizationResult | None:
        """要求一项授权通过, 返回 None 或拒绝结果.

        闭环: 读 state → permission.check → 无 Question 放行; 有则自 issuer 发到
        permission.namespace → 等待应答 (取消随 scope) → permission.replied →
        按 save flag 落盘 → 返回结果.
        """
        ...
