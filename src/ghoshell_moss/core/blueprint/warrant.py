from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar
from typing_extensions import Self
from pydantic import BaseModel
from ghoshell_container import IoCContainer
from ghoshell_moss.core.concepts.qa import Question, Answer


class ItemMeta(BaseModel):
    """model 可列表化的元数据. """
    data: dict[str, Any]


class ItemModel(BaseModel, ABC):
    """一种类型的授权数据. """
    def to_meta(self) -> ItemMeta:
        ...

ITEM_MODEL = TypeVar("ITEM_MODEL", bound=ItemModel)

class Item(Generic[ITEM_MODEL], ABC):
    """一个授权检查的场景. """

    @classmethod
    def factory(cls, container: IoCContainer) -> Self:
        """实例化自身."""
        ...

    @property
    @abstractmethod
    def default(self) -> ITEM_MODEL:
        """默认的授权状态"""
        ...

    @abstractmethod
    def check(self, config: ItemMeta) -> Question | None:
        """检查存储的授权状态, 返回 question 表示需要授权; 否则返回 None"""
        ...

    @abstractmethod
    async def replied(self, answer: Answer) -> tuple[ItemModel, bool, str | None]:
        """基于返回的 answer, 判断后续的配置变更, 和是否通过. 不通过返回报错."""
        ...



class Warrant(ABC):

    @abstractmethod
    async def check(self, item: Item) -> str | None:
        """返回是否授权成功. """
        # 1. 对应 session-scope 里的 storage 找到配置文件.
        # 2. 如果没有 item 的配置文件. 使用默认的配置文件.
        # 3. 检查配置文件, 判断是否要发起申请. 申请的条件参数应该在 __init__ 里构建.
        # 4. 如果产生 question, 通过 qa 发送.
        # 5. 拿到 qa 的结果, 用 item 校验, 返回新的配置文件, 是否成功, 拒绝理由等等.
        ...

    @abstractmethod
    async def __aenter__(self) -> Self:
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc, tb):
        ...
