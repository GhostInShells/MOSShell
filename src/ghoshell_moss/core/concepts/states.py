import asyncio
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any, ClassVar, Optional

from ghoshell_common.helpers import generate_import_path, uuid
from pydantic import BaseModel, Field
from typing_extensions import Self

__all__ = ["MemoryStateStore", "State", "StateBaseModel", "StateModel", "StateStore"]


class State(BaseModel):
    version: str = Field(default="", description="state version, Optimistic Lock")
    name: str = Field(description="The name of the state object.")
    changed_by: str = Field(default="", description="who change the state object.")
    description: str = Field(default="", description="The description of the state object.")
    data: dict[str, Any] = Field(description="the default value of the state")


class StateModel(ABC):
    @classmethod
    @abstractmethod
    def to_state(cls) -> State:
        pass

    @abstractmethod
    def to_state_data(self) -> dict[str, Any]:
        pass

    @classmethod
    @abstractmethod
    def from_state(cls, state: State) -> Self:
        pass

    @classmethod
    @abstractmethod
    def get_state_name(cls) -> str:
        pass


class StateBaseModel(BaseModel, StateModel, ABC):
    """
    通过强类型的方式对 State 进行建模.
    """

    state_desc: ClassVar[str] = ""
    state_name: ClassVar[str] = ""

    version: str = Field(default="", description="state version, Optimistic Lock")

    def to_state(self) -> State:
        name = self.state_name or generate_import_path(self.__class__)
        description = self.state_desc or self.__doc__ or ""
        data = self.model_dump()
        version = self.version
        return State(name=name, description=description, data=data, version=version)

    def to_state_data(self) -> dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_state(cls, state: State) -> Self:
        new_one = cls(**state.data)
        new_one.version = state.version
        return new_one

    @classmethod
    def get_state_name(cls) -> str:
        # 最好定义 state name, 否则引用路径经常会根据 python 的路径不同而变化.
        return cls.state_name or generate_import_path(cls)


class StateStore(ABC):
    @abstractmethod
    def register(self, *states: State | StateModel) -> None:
        """
        注册一个状态. 并且决定是否与整个系统共享.
        """
        pass

    @abstractmethod
    def set(self, state: State | StateModel) -> None:
        """
        强制设置一个 State 到本地.
        """
        raise NotImplementedError

    @abstractmethod
    async def get(self, state_name: str) -> dict[str, Any] | None:
        """
        获取当前状态. 只有注册过的状态才会返回值.
        :raise AttributeError: 如果调用了没注册过的 State, 会抛出异常.
        """
        pass

    @abstractmethod
    async def get_model(self, default: StateModel | type[StateModel]) -> StateModel:
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

    @abstractmethod
    async def on_change(
        self,
        callback: Callable[[State], Coroutine[None, None, None]],
        state_name: Optional[str] = None,
    ) -> None:
        """
        记录 change.
        """
        pass


class MemoryStateStore(StateStore):
    def __init__(self, owner: str):
        self._owner = owner
        self._states: dict[str, State] = {}
        self._on_change_callbacks: list[Callable[[State], Coroutine[None, None, None]]] = []
        self._on_state_name_change_callbacks: dict[str, list[Callable[[State], Coroutine[None, None, None]]]] = {}

    def register(self, *states: State | StateModel) -> None:
        for state in states:
            saving = state
            if isinstance(state, StateModel):
                saving = state.to_state()
            if saving.name in self._states:
                # 不重复注册, 按顺序.
                continue
            self._states[saving.name] = saving

    def set(self, state: State | StateModel) -> None:
        state_value = state
        if isinstance(state, StateModel):
            state_value = state.to_state()

        state_value.version = uuid()
        state_value.changed_by = self._owner
        self._states[state_value.name] = state_value

        callbacks = [*self._on_change_callbacks]
        callbacks.extend(self._on_state_name_change_callbacks.get(state_value.name, []))
        if not callbacks:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

        async def _run_callbacks() -> None:
            await asyncio.gather(*(callback(state_value) for callback in callbacks))

        loop.create_task(_run_callbacks())

    async def get(self, state_name: str) -> dict[str, Any] | None:
        state = self._states.get(state_name)
        if state is None:
            return None
        return state.data

    async def get_model(self, default: StateModel | type[StateModel]) -> StateModel:
        state_name = default.get_state_name()
        result = None
        if not isinstance(default, StateModel) and issubclass(default, StateModel):
            state_cls = default
        else:
            state_cls = type(default)
            result = default
        value = self._states.get(state_name, None)
        if value is None:
            if result is not None:
                return result
            else:
                raise LookupError(f"Cannot find state {state_name}")
        else:
            return state_cls.from_state(value)

    async def save(self, state: StateModel | State) -> bool:
        state_value = state
        if isinstance(state, StateModel):
            state_value = state.to_state()
        exists = self._states.get(state_value.name, None)
        if exists is not None:
            if state_value.version != exists.version:
                # 乐观锁不匹配.
                return False
        state_value.version = uuid()
        state_value.changed_by = self._owner
        self._states[state_value.name] = state_value
        callbacks = [*self._on_change_callbacks]
        callbacks.extend(self._on_state_name_change_callbacks.get(state_value.name, []))
        # todo: 考虑用全异步.
        await asyncio.gather(*(callback(state_value) for callback in callbacks))
        return True

    async def on_change(
        self,
        callback: Callable[[State], Coroutine[None, None, None]],
        state_name: Optional[str] = None,
    ) -> None:
        if state_name is None:
            self._on_change_callbacks.append(callback)
        else:
            registered = self._on_state_name_change_callbacks.get(state_name, [])
            registered.append(callback)
            self._on_state_name_change_callbacks[state_name] = registered
