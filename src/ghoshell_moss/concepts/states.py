from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Dict, ClassVar, Any, Type, Iterable
from ghoshell_common.helpers import generate_import_path


class State(BaseModel):
    version: str = Field(default="", description="state version, Optimistic Lock")
    name: str = Field(description="The name of the state object.")
    description: str = Field(default="", description="The description of the state object.")
    schema: Dict[str, Any] = Field(description="the json schema of the state")
    data: Dict[str, Any] = Field(description="the default value of the state")


class StateModel(BaseModel, ABC):
    """
    通过强类型的方式对 State 进行建模.
    """
    state_desc: ClassVar[str] = ""
    state_name: ClassVar[str] = ""

    version: str = Field(default="", description="state version, Optimistic Lock")

    @classmethod
    def to_state(cls) -> State:
        name = cls.state_name or generate_import_path(cls)
        description = cls.state_desc or cls.__doc__ or ""
        default = cls().model_dump()
        schema = cls.model_json_schema()
        return State(name=name, description=description, schema=schema, default=default)


class StateStore(ABC):

    @abstractmethod
    def register(self, states: Iterable[State | StateModel], *, share: bool = False) -> None:
        """
        注册一个状态. 并且决定是否与整个系统共享.
        """
        pass

    def set(self, state: State | StateModel) -> None:
        """
        强制设置一个 State 到本地.
        """
        pass

    @abstractmethod
    async def get(self, state_name: str) -> Dict[str, Any]:
        """
        获取当前状态. 只有注册过的状态才会返回值.
        :raise AttributeError: 如果调用了没注册过的 State, 会抛出异常.
        """
        pass

    @abstractmethod
    async def get_model(self, default: StateModel) -> StateModel:
        """
        获取一个强类型的 StateModel. 如果目标不存在, 或者数据结构有冲突, 会返回 default 值.
        """
        pass

    @abstractmethod
    async def save(self, state: StateModel | State) -> bool:
        """
        保存一个 State. 其中的 Version 是乐观锁.
        Save 会触发广播和更新.
        """
        pass
