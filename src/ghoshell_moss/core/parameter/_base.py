"""AbsParameters — Parameters 的抽象基类, 收敛队列 / task / 生命周期, transport 留给子类."""

import asyncio
import contextlib
from abc import ABC, abstractmethod
from typing import Callable, Type

import janus

from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.core.blueprint.parameter import (
    Parameters,
    ParameterModel,
    ParameterSchema,
    ParameterDeclaration,
    ParameterSubscriber,
    T_PARAM,
)

__all__ = [
    "BaseParameterDeclaration",
    "BaseParameterSubscriber",
    "AbsParameters",
]


class BaseParameterDeclaration(ParameterDeclaration):
    def __init__(
            self,
            key: str,
            value: T_PARAM,
            set_callback: Callable[[str, ParameterModel], None],
    ):
        self._key = key
        self._value = value
        self._callback = set_callback

    @property
    def key(self) -> str:
        return self._key

    @property
    def value(self) -> T_PARAM:
        return self._value

    def set(self, value: T_PARAM) -> None:
        self._value = value
        self._callback(self._key, value)

    def parameter_schema(self) -> ParameterSchema:
        return self._value.to_parameter_schema()


class BaseParameterSubscriber(ParameterSubscriber[T_PARAM]):
    def __init__(
            self,
            value: T_PARAM | None,
            model: Type[T_PARAM],
            logger: LoggerItf | None = None,
    ):
        self._model = model
        self._value = value
        self._on_change_callbacks: set[Callable[[T_PARAM], None]] = set()
        self._disposer: Callable[[], None] | None = None
        self._logger = logger or get_moss_logger()

    @property
    def value(self) -> T_PARAM | None:
        return self._value

    def on_change(self, callback: Callable[[T_PARAM], None]) -> Callable[[], None]:
        self._on_change_callbacks.add(callback)

        def _disposer() -> None:
            self._on_change_callbacks.discard(callback)

        return _disposer

    def set_disposer(self, disposer: Callable[[], None]) -> None:
        self._disposer = disposer

    def close(self) -> None:
        if self._disposer is not None:
            self._disposer()
            self._disposer = None

    def update(self, value: T_PARAM) -> None:
        """transport 回调 — 收推 (含订阅时拉到的初始值) 时覆写本地并触发 on_change."""
        self._value = value
        for callback in list(self._on_change_callbacks):
            try:
                callback(value)
            except Exception:
                self._logger.exception("Parameter subscriber callback failed: %r", callback)


class AbsParameters(Parameters, ABC):
    def __init__(self, logger: LoggerItf | None = None):
        self._logger = logger or get_moss_logger()
        self._declarations: dict[str, BaseParameterDeclaration] = {}
        self._subscribers: dict[str, BaseParameterSubscriber] = {}
        self._pub_parameter_queue: janus.Queue[tuple[str, ParameterModel]] = janus.Queue()
        self._publish_parameter_loop_task: asyncio.Task | None = None
        self._started = False
        self._stopped = False

    # -- Parameters ----------------------------------------------------

    async def declare(
            self, model: T_PARAM, *, key: str | None = None,
    ) -> ParameterDeclaration[T_PARAM]:
        key = key or model.parameter_key()
        if key not in self._declarations:
            self._declarations[key] = BaseParameterDeclaration(
                key=key,
                value=model,
                set_callback=self._set_parameter,
            )
        return self._declarations[key]

    async def subscribe(
            self,
            model: Type[T_PARAM],
            *,
            key: str | None = None,
            address: str | None = None,
    ) -> ParameterSubscriber[T_PARAM]:
        key = key or model.parameter_key()
        if key not in self._subscribers:
            subscriber = BaseParameterSubscriber(value=None, model=model, logger=self._logger)
            self._subscribers[key] = subscriber
            disposer = await self._subscribe_parameter(
                key=key, model=model, address=address, callback=subscriber.update,
            )
            subscriber.set_disposer(disposer)
        return self._subscribers[key]

    def declared(self) -> list[ParameterSchema]:
        return [d.parameter_schema() for d in self._declarations.values()]

    # -- transport (子类实现) ------------------------------------------

    def _set_parameter(self, key: str, parameter: ParameterModel) -> None:
        try:
            self._pub_parameter_queue.sync_q.put_nowait((key, parameter))
        except janus.SyncQueueShutDown:
            pass

    @abstractmethod
    async def _publish_declaration(self, key: str, parameter: ParameterModel) -> None:
        """声明者 push 一次值 (key + model).  子类决定 transport."""
        ...

    @abstractmethod
    async def _subscribe_parameter(
            self,
            *,
            key: str,
            model: Type[T_PARAM],
            address: str | None,
            callback: Callable[[T_PARAM], None],
    ) -> Callable[[], None]:
        """订阅某 ``address`` 声明的 parameter.  先拉一次当前值走 callback, 再持续收推.
        返回退订 disposer."""
        ...

    # -- 生命周期 -------------------------------------------------------

    async def _publish_declaration_loop(self) -> None:
        while self.is_running():
            try:
                key, parameter = await self._pub_parameter_queue.async_q.get()
                await self._publish_declaration(key, parameter)
            except asyncio.CancelledError:
                raise
            except janus.AsyncQueueShutDown:
                break
            except Exception:
                self._logger.exception("failed publishing declaration")

    def is_running(self) -> bool:
        return self._started and not self._stopped

    async def __aenter__(self) -> 'Parameters':
        self._started = True
        self._publish_parameter_loop_task = asyncio.create_task(self._publish_declaration_loop())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._stopped:
            return
        self._stopped = True
        if self._publish_parameter_loop_task is not None and not self._publish_parameter_loop_task.done():
            self._publish_parameter_loop_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._publish_parameter_loop_task
            self._publish_parameter_loop_task = None
        for subscriber in self._subscribers.values():
            subscriber.close()
        self._subscribers.clear()
