"""Warrant — Matrix 级通用授权机制抽象.

三层职责分离: qa 是交互协议, warrant 是存储 + 装线, permission 是业务逻辑.
qa 交互协议见 `core/concepts/qa.py`; 本模块是 warrant + permission 抽象面.

职责边界 (显著提示):
- permission 是纯逻辑, 无 IO: 不碰持久化、不发问题、不知道 namespace.
- warrant 是唯一 IO 面, 保持哑: 不解释 state、不派生 key/type、不参与业务判断.
- warrant 是授权闭环专用, 不是通用 QA 客户端 — command 要走 QA 拿实参,
  直接自己调 QA, 不走 warrant.

模板方法: require 是授权闭环的默认实现, concrete 填四个原材料 — states
(读缓存), ask_question (发问题等答案), store (入队落盘), list_states (枚举).
取消沿调用方 scope 传播; 存储时序由 __aenter__ 创建的落盘 task + 有序队列
保证.

软授权边界 (非安全机制, 见 warrant FEATURE.md KD14): warrant 是交互式审批,
不是安全边界 — MOSS 允许模型自迭代, 模型可自写 node 做 QA watcher 给自己授权.
若需硬化为真授权: ① `.moss` 移出 project dir; ② QA namespace 改秘密 UUID;
③ UUID 用 credential 而非环境变量 (或 qa/warrant 改走第三方工业级 provider).
当前 pre-1.0 有意不硬化, 仅声明此边界.
"""

# 设计决策见 warrant FEATURE.md

from abc import ABC, abstractmethod
from typing import Any, Callable, Generic, TypeVar
from typing_extensions import Self

from pydantic import BaseModel, Field, ValidationError, AwareDatetime
from ghoshell_moss.core.concepts.qa import Answer, Question

from datetime import datetime, timezone
import asyncio

__all__ = [
    'AuthorizationResult',
    'Permission',
    'PermissionStateData',
    'Warrant',
    'StateT',
]

StateT = TypeVar('StateT', bound=BaseModel)


class AuthorizationResult(BaseModel, Generic[StateT]):
    """授权结果. allowed 单通道表达通过/拒绝.

    只承载执行与否 + 变更后的授权状态. 需要结构化扩展的 concrete
    继承本类加字段.
    """

    allowed: bool
    reason: str | None = None
    state: StateT | None = None


class Permission(ABC, Generic[StateT]):
    """授权场景业务逻辑. 纯逻辑, 无 IO.

    key/type 是语言无关的约定路径字符串 (人工声明, 形如 a.b.c, 类似
    topic_name/topic_type), 不依赖模块结构.

    参数化 permission 的实体隔离由 concrete 在 state 文档内部处理,
    warrant 保持哑.
    """

    @classmethod
    @abstractmethod
    def key(cls) -> str:
        """语言无关唯一键, 寻址每份授权状态. 形如 a.b.c."""
        ...

    @classmethod
    @abstractmethod
    def type(cls) -> str:
        """语言无关类型标识, 约定 permission 类型. 形如 a.b.c."""
        ...

    @abstractmethod
    def default(self) -> StateT:
        """无存储时的初始状态. 返回实例的类型是 StateT 的权威来源."""
        ...

    @abstractmethod
    def check(self, state: StateT) -> Question | None:
        """None = 无需授权直接放行; Question = 构造好的完整审批问题."""
        ...

    @abstractmethod
    def replied(self, answer: Answer) -> AuthorizationResult[StateT]:
        """解释应答, 返回结果 (含新 state). 是否存储由 state 是否有值决定."""
        ...


class PermissionStateData(BaseModel):
    """可存储的 permission state data. 弱类型载体, 强类型还原靠 permission.

    对齐 Topic/QA 的 Model + Meta 模式: data 是 StateT 的序列化本体,
    key 是语言无关寻址.
    """

    key: str = Field(
        description="语言无关唯一键, 对应 permission.key()",
    )
    created: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="记录创建时间",
    )
    seq: int | None = Field(
        default=None,
        description="每 key 单调序号 (语言无关 version/index). host-only 单写模式留 None, "
                    "过 topic 时填; host 只接受 seq == current + 1, 其余 reject-retry (见 v8)",
    )
    data: dict[str, Any] = Field(
        description="StateT 序列化本体 (model_dump)",
    )

    @classmethod
    def from_state(cls, permission: Permission[StateT], state: StateT) -> Self:
        """从 permission + 具体 state 构造存储载体."""
        return cls(
            key=permission.key(),
            data=state.model_dump(exclude_none=True, mode='json'),
        )

    @classmethod
    def from_permission(cls, permission: Permission[StateT]) -> Self:
        """从 permission 的默认 state 构造存储载体 (初始记录用)."""
        return cls(
            key=permission.key(),
            data=permission.default().model_dump(exclude_none=True, mode='json'),
        )

    def to_permission_state(self, permission: Permission[StateT]) -> StateT | None:
        """按 permission 还原强类型 state; key 不匹配返回 None.

        返回实例的类型是 StateT 权威来源; 还原失败 (ValidationError) 由调用方处理.
        """
        if permission.key() != self.key:
            return None
        return type(permission.default()).model_validate(self.data)


class Warrant(ABC):
    """交互式鉴权授权模块.

    将授权/审批/交互能力单元化封装. 模板方法: require 是授权闭环的默认实现,
    concrete 填 4 个原材料 — states (读缓存), ask_question (发问题等答案),
    store (同步入队落盘), list_states (枚举).

    存储时序: store 同步更新内存缓存并推入有序队列, 落盘 IO 由 __aenter__
    创建的生命周期 task 消费队列执行, 保证写序. require 读 state 走内存缓存
    (get_permission_state). store 的实现不应抛出影响授权结果的异常, 落盘失败
    自行记录.
    """

    @abstractmethod
    async def __aenter__(self) -> Self:
        """创建生命周期对象: 加载存储进内存缓存, spawn 落盘 task + 有序队列."""
        ...

    @abstractmethod
    async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: Any,
    ) -> None:
        """flush 落盘队列 + cancel 落盘 task; 取消随调用方 scope 传播."""
        ...

    @abstractmethod
    def is_running(self) -> bool:
        """生命周期内 true."""
        ...

    @abstractmethod
    def states(self) -> dict[str, PermissionStateData]:
        """当前内存缓存的全部授权状态 (key → data). 同步, 读缓存."""
        ...

    @abstractmethod
    async def ask_question(self, question: Question) -> Answer:
        """通过 warrant 约定的 namespace 发问题并等待应答.

        :raises asyncio.CancelledError: 调用方 scope 取消时, 必须沿此传播.
        """
        ...

    @abstractmethod
    def store(self, state: PermissionStateData) -> None:
        """主动存一份授权状态. 同步入队 (更新内存缓存 + 推入有序队列),
        落盘 IO 由生命周期 task 异步执行, 保序."""
        ...

    @abstractmethod
    def list_states(self) -> list[PermissionStateData]:
        """枚举全部授权状态. 同步, 读内存缓存."""
        ...

    @abstractmethod
    def on_flushed(
            self,
            callback: Callable[[PermissionStateData], None],
    ) -> Callable[[], None]:
        """登记"某份 state 真实落盘"后的回调, 返回注销句柄.

        "真实落盘发生"是存储时序通用契约, 触发方式是 concrete 差异:
        host 版写盘后触发, topic 版收 truth 后触发 (见 v8). 回调收到已落盘的
        PermissionStateData.
        """
        ...

    def get_permission_state(self, permission: Permission[StateT]) -> StateT:
        """读该 permission 的授权状态并还原强类型; 无记录或校验失败 fallback default."""
        state_data = self.states().get(permission.key())
        state = permission.default()
        if state_data:
            try:
                saved_state = state_data.to_permission_state(permission)
                if saved_state:
                    state = saved_state
            except ValidationError:
                # concrete 可覆盖本方法以处理校验失败, 默认 fallback 初始状态.
                pass
        return state

    async def require(self, permission: Permission[StateT]) -> AuthorizationResult[StateT]:
        """要求一项授权通过. 统一返回结果, allowed 表达放行 (无 None).

        闭环: 读 state → permission.check → 无 Question 放行; 有则自 issuer
        发到 warrant 约定的 namespace → 等待应答 (取消沿 scope 传播) →
        permission.replied → 有 state 则落盘 → 返回结果.
        """
        state = self.get_permission_state(permission)
        result: AuthorizationResult[StateT] | None = None
        try:
            question = permission.check(state)
            if question is None:
                result = AuthorizationResult(allowed=True)
            else:
                answer = await self.ask_question(question)
                result = permission.replied(answer)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            result = AuthorizationResult(allowed=False, reason=f"Error: {e}")
        finally:
            if result is not None and result.state is not None:
                self.store(PermissionStateData.from_state(permission, result.state))
        return result
