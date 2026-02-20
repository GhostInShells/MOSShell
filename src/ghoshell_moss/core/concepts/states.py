import asyncio
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import Any, ClassVar, Optional, TypeVar

from ghoshell_common.helpers import generate_import_path, uuid
from pydantic import BaseModel, Field
from typing_extensions import Self

__all__ = ["BaseStateStore", "State", "StateBaseModel", "StateModel", "StateStore"]


class State(BaseModel):
    """
    State 是在 Shell 和 Channel 之间共享的状态数据.
    State 本身是可传输的数据结构.
    """

    name: str = Field(description="The name of the state object.")
    uid: str = Field(default_factory=uuid, description="The unique identifier for the state.")
    issuer: str = Field(default="", description="who change the state object.")
    data: dict[str, Any] = Field(description="the default value of the state")


class StateModel(ABC):
    """
    State Model 是对 State 的强类型建模.
    """

    @classmethod
    @abstractmethod
    def to_state(cls) -> State:
        """
        从强类型转化为弱类型.
        """
        pass

    @classmethod
    @abstractmethod
    def from_state(cls, state: State) -> Self:
        """
        通过 state 对象重建.
        """
        pass

    @classmethod
    @abstractmethod
    def get_state_name(cls) -> str:
        """
        返回 state 的唯一命名.
        """
        pass


class StateBaseModel(BaseModel, StateModel, ABC):
    """
    通过强类型的方式对 State 进行建模.
    基于 pydantic BaseModel 实现.
    """

    uid: str = Field(default="", description="The unique identifier for the state.")
    issuer: str = Field(default="", description="who change the state object.")

    def to_state(self) -> State:
        name = self.get_state_name()
        data = self.model_dump(exclude={"uid", "issuer"})
        uid = self.uid or uuid()
        issuer = self.issuer
        return State(name=name, data=data, uid=uid, issuer=issuer)

    @classmethod
    def from_state(cls, state: State) -> Self:
        new_one = cls(**state.data)
        new_one.uid = state.uid
        new_one.issuer = state.issuer
        return new_one

    @classmethod
    def get_state_name(cls) -> str:
        return generate_import_path(cls)


STATE_MODEL = TypeVar("STATE_MODEL", bound=StateModel)


class StateStore(ABC):
    """
    State 存储和通讯的中枢.
    """

    @abstractmethod
    def id(self) -> str:
        pass

    @abstractmethod
    def register(self, *states: State | StateModel) -> None:
        """
        注册一系列的状态值.
        """
        pass

    @abstractmethod
    def all(self) -> dict[str, State]:
        pass

    @abstractmethod
    def is_listening(self) -> bool:
        pass

    @abstractmethod
    def listening(self) -> set[str]:
        pass

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    async def register_child(self, store: Self) -> None:
        pass

    @abstractmethod
    def get(self, state_name: str) -> State | None:
        """
        获取当前状态. 只有注册过的状态才会返回值.
        """
        pass

    def get_model(self, default: STATE_MODEL | type[STATE_MODEL]) -> STATE_MODEL | None:
        """
        获取一个强类型的 StateModel. 如果目标不存在, 或者数据结构有冲突, 会返回 default 值.
        """
        name = default.get_state_name()
        state_value = self.get(name)
        if state_value is None:
            if isinstance(default, StateModel):
                return default
            else:
                return None
        return default.from_state(state_value)

    @abstractmethod
    async def save(self, state: StateModel | State) -> None:
        """
        保存一个 State. 会校验乐观锁.
        Save 会触发上行广播.
        """
        pass

    @abstractmethod
    async def on_sync(self, state: StateModel | State) -> None:
        pass

    async def __aenter__(self):
        await self.start()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class BaseStateStore(StateStore):
    """
    基线的 StateStore 实现.
    """

    def __init__(self, owner: str, *, parent: StateStore | None = None):
        self._owner = owner
        self._states: dict[str, State] = {}
        self._register_child_lock = asyncio.Lock()
        self._save_lock = asyncio.Lock()
        self._on_sync_lock = asyncio.Lock()
        self._parent = parent
        self._children: dict[str, StateStore] = {}
        self._closed = asyncio.Event()
        self._started = asyncio.Event()

    async def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._parent = None
        self._children.clear()

    async def start(self) -> None:
        if self._started.is_set():
            return
        self._started.set()
        if len(self._children) > 0:
            for child in self._children.values():
                for state_name, value in child.all().items():
                    if state_name not in self._states:
                        self._states[state_name] = value

        if self._parent:
            # 同时会完成同步.
            await self._parent.register_child(self)

    def id(self) -> str:
        return self._owner

    def all(self) -> dict[str, State]:
        return self._states

    def is_listening(self) -> bool:
        return self._started.is_set() and not self._closed.is_set()

    def listening(self) -> list[str]:
        if not self.is_listening():
            return []
        return list(self._states.keys())

    async def register_child(self, store: Self) -> None:
        try:
            await self._register_child_lock.acquire()
            child_id = store.id()
            if child_id in self._children:
                return
            # 注册子节点.
            self._children[child_id] = store
            all_states = store.all()
            for state_name, value in all_states.items():
                if state_name not in self._states:
                    self._states[state_name] = value
                    # 不需要广播给子孙.
                else:
                    # 如果已经注册过了, 用注册过的值来更新孩子的值.
                    exists = self._states[state_name]
                    exists.issuer = self._owner
                    await store.on_sync(exists)
        finally:
            self._register_child_lock.release()

    def register(self, *states: State | StateModel) -> None:
        for state in states:
            saving = state
            if isinstance(state, StateModel):
                saving = state.to_state()

            if saving.name in self._states:
                # 不重复注册, 按顺序.
                continue
            saving.issuer = self._owner
            self._states[saving.name] = saving

    def get(self, state_name: str) -> State | None:
        state = self._states.get(state_name)
        if state is None:
            return None
        state = state.model_copy()
        state.uid = uuid()
        return state

    async def _do_saving(self, state_value: State):
        exists = self._states.get(state_value.name)
        if exists and exists.uid == state_value.uid:
            # 已经存储过.
            return

        state_value = state_value.model_copy()
        try:
            await self._save_lock.acquire()
            self._states[state_value.name] = state_value
            state_name = state_value.name
            # 改成自己发布的 state.
            saving_by_self = state_value.model_copy()
            saving_by_self.issuer = self._owner

            saving_tasks = []
            removing_child = []
            for child in self._children.values():
                child_id = child.id()
                if not child.is_listening():
                    removing_child.append(child_id)
                    continue
                if state_name not in child.listening():
                    continue
                saving_tasks.append(asyncio.create_task(child.on_sync(saving_by_self)))

            _ = await asyncio.gather(*saving_tasks)
            # 删除掉不听话的小孩.
            for child_id in removing_child:
                del self._children[child_id]
        finally:
            self._save_lock.release()

    async def on_sync(self, state: StateModel | State) -> None:
        if not self._started.is_set() or self._closed.is_set():
            # 直接忽略掉.
            return
        await self._do_saving(state)

    async def save(self, state: StateModel | State) -> None:
        if not self._started.is_set() or self._closed.is_set():
            # 直接忽略掉.
            return
        # 先类型转换, 确保 state 是 State 对象.
        state_value = state
        if isinstance(state, StateModel):
            state_value = state.to_state()

        if not isinstance(state_value, State):
            raise ValueError("Cannot save state of type {} to state of type {}".format(type(state), type(state)))

        if state_value.name not in self._states:
            # 忽略未监听的.
            return

        # 标记是自己的修改.
        state_value.issuer = self._owner
        if self._parent is None:
            await self._do_saving(state_value)
        else:
            await self._parent.save(state_value)
