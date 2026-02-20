import contextlib

import asyncio
import contextvars
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import (
    Optional, Iterable, Any, TypeVar, Generic
)
from typing_extensions import Self

from ghoshell_container import IoCContainer, Container

from ghoshell_moss.core.concepts.command import (
    CommandTask, CommandResultStack, CommandUniqueName, Command, CommandTaskState,
)
from ghoshell_moss.core.concepts.states import StateStore, BaseStateStore, State
from ghoshell_moss.core.concepts.channel import (
    ChannelCtx, Channel, ChannelMeta, TaskDoneCallback, RefreshMetaCallback, ChannelBroker,
    ChannelFullPath, ChannelPaths,
)
from ghoshell_moss.core.concepts.errors import CommandErrorCode
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_common.contracts import LoggerItf
import logging

__all__ = ['AbsChannelBroker', 'ChannelImportLib', 'AbsChannelTreeBroker']

_ChannelId = str
_TaskWithPaths = tuple[ChannelPaths, CommandTask]


class ChannelImportLib:
    """
    唯一的 lib 用来管理所有可以被 import 的 channel broker
    """

    def __init__(self, main: ChannelBroker, container: IoCContainer | None = None):
        self._main = main
        self._name = "MossChannelImportLib/{}/{}".format(main.name, main.id)
        self._container = Container(
            name=self._name,
            parent=container,
        )
        # 绑定自身到容器中. 凡是用这个容器启动的 broker, 都可以拿到 ChannelImportLib 并获取子 channel broker.
        self._container.set(ChannelImportLib, self)
        self._logger: Optional[LoggerItf] = None
        self._brokers: dict[_ChannelId, ChannelBroker] = {}
        self._brokers_lock: asyncio.Lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._start: bool = False
        self._close: bool = False

    def get_channel_broker(self, channel: Channel) -> ChannelBroker | None:
        if channel is self._main.channel:
            # 根节点不启动.
            return self._main

        if not self.is_running():
            return None

        channel_id = channel.id()
        return self._brokers.get(channel_id)

    async def get_or_create_channel_broker(self, channel: Channel) -> ChannelBroker | None:
        if broker := self.get_channel_broker(channel):
            await broker.wait_started()
            if broker.is_running():
                return broker
            else:
                return None
        # 第一次创建.
        broker = await self._build_channel_broker(channel)
        await broker.wait_started()
        return broker

    async def _build_channel_broker(self, channel: Channel) -> ChannelBroker | None:
        # 只有创建这一段需要上锁.
        if not self.is_running():
            return None
        await self._brokers_lock.acquire()
        try:
            channel_id = channel.id()
            broker = self._brokers.get(channel_id)
            # 只要 broker 存在就立刻返回.
            if broker is not None:
                return broker
            # 用自身的容器启动 ChannelImportLib.
            broker = channel.bootstrap(self._container)
            # 避免抢锁嵌套成环.
            self._brokers[channel_id] = broker
            _ = asyncio.create_task(broker.start())
            return broker
        except Exception as e:
            self.logger.exception(
                "%s failed to build channel %s, id=%s: %s",
                self._name, channel.name(), channel.id(), e
            )
            return None
        finally:
            self._brokers_lock.release()

    @property
    def main(self) -> ChannelBroker:
        return self._main

    @property
    def logger(self):
        if self._logger is None:
            self._logger = self._container.get(LoggerItf)
            if self._logger is None:
                logger = logging.getLogger('moss')
                self._logger = logger
                self._container.set(LoggerItf, logger)
        return self._logger

    def is_running(self) -> bool:
        return self._start and not self._close

    async def start(self) -> None:
        if self._start:
            return
        self._start = True
        self._loop = asyncio.get_event_loop()
        await asyncio.to_thread(self._container.bootstrap)

    def find_descendants(self, channel: Channel) -> dict[ChannelFullPath, ChannelBroker]:
        result = {}
        broker = self.get_channel_broker(channel)
        if broker is None or not broker.is_running():
            return result
        for name, child in broker.imported().items():
            child_broker = self.get_channel_broker(child)
            result[name] = child_broker
            if child_broker is not None and child_broker.is_running():
                descendants = self.find_descendants(child)
                for path, descendant in descendants.items():
                    real_path = Channel.join_channel_path(name, path)
                    result[real_path] = descendant
        return result

    def recursively_find_broker(self, broker: ChannelBroker, path: ChannelFullPath) -> ChannelBroker | None:
        if path == "":
            return broker
        paths = Channel.split_channel_path_to_names(path, 1)
        child_name = paths[0]
        further_path = paths[1] if len(paths) > 1 else ""
        if child_name == "":
            return broker
        child_channel = broker.imported().get(child_name)
        if child_channel is None:
            return None
        child_broker = self.get_channel_broker(child_channel)
        if child_broker is None:
            return None
        return self.recursively_find_broker(child_broker, further_path)

    async def recursively_fetch_broker(self, root: ChannelBroker, paths: ChannelPaths) -> ChannelBroker | None:
        if len(paths) == 0:
            return root
        child_name = paths[0]
        further_path = paths[1:]
        child = root.imported().get(child_name)
        if child is None:
            return None
        child_broker = await self.get_or_create_channel_broker(child)
        return await self.recursively_fetch_broker(child_broker, further_path)

    async def close(self) -> None:
        if self._close:
            return
        self._close = True
        await self._brokers_lock.acquire()
        try:
            clear_brokers = []
            clear_broker_tasks = []
            closing_broker_ids = set()
            for broker in self._brokers.values():
                if broker.is_running():
                    if broker.id in closing_broker_ids:
                        continue
                    closing_broker_ids.add(broker.id)
                    clear_task = self._loop.create_task(broker.close())
                    clear_brokers.append(broker)
                    clear_broker_tasks.append(clear_task)
            done = await asyncio.gather(*clear_broker_tasks, return_exceptions=True)
            idx = 0
            self._brokers.clear()
            for t in done:
                if isinstance(t, Exception):
                    broker = clear_brokers[idx]
                    self.logger.exception(
                        "%s close broker %s, id=%s failed: %s",
                        self._name, broker.name, broker.id, t)
                idx += 1
        finally:
            self._brokers_lock.release()
            if self._loop:
                self._loop.run_in_executor(None, self._container.shutdown)


CHANNEL = TypeVar('CHANNEL', bound=Channel)


class AbsChannelBroker(Generic[CHANNEL], ChannelBroker, ABC):
    """
    实现基础的 Channel Broker, 用来给所有的 Broker 提供基准的生命周期.
    """

    def __init__(
            self,
            *,
            channel: CHANNEL,
            container: IoCContainer | None = None,
            logger: LoggerItf | None = None,
            state_store: StateStore | None = None,
    ):
        self._channel: CHANNEL = channel
        self._name = channel.name()
        self._uid = channel.id()
        # 用不同的容器隔离依赖. 经过 prepare container 才进行封装.
        container = Container(
            name=f'MossChannelBroker/{self._name}/{self._uid}',
            parent=container,
        )
        self._container: IoCContainer = container
        self._logger: LoggerItf | None = logger
        self._state_store: StateStore | None = state_store
        # import lib 是最重要的.
        self._importlib: ChannelImportLib | None = None

        self._logger: LoggerItf | None = logger

        self._starting = False
        self._started = asyncio.Event()
        self._running_task: Optional[asyncio.Task] = None
        # 用线程安全的事件. 考虑到 broker 未来可能会跨线程被使用.
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()

        self._cached_metas: dict[ChannelFullPath, ChannelMeta] = {}
        # 可以注册监听, 监听 refresh meta 动作.
        self._on_refresh_meta_callbacks: list[Callable[[ChannelMeta], Coroutine[None, None, None]]] = []
        self._refresh_meta_lock = asyncio.Lock()

        self._defer_clear_mark = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._main_loop_task: Optional[asyncio.Task] = None
        self._task_done_callbacks: list[TaskDoneCallback] = []
        self._exit_stack = contextlib.AsyncExitStack()

        # log_prefix
        self.log_prefix = "[Channel %s %s][%s]" % (self._name, self._uid, self.__class__.__name__)

    @property
    def channel(self) -> CHANNEL:
        return self._channel

    @property
    def states(self) -> StateStore:
        """
        返回当前 Channel 的状态存储.
        """
        if self._state_store is None:
            # 必须依赖一个 state store.
            self._state_store = self._container.force_fetch(StateStore)
        return self._state_store

    @property
    def logger(self) -> LoggerItf:
        if self._logger is None:
            # 日志总要有吧.
            self._logger = self.container.get(LoggerItf) or logging.getLogger('moss')
        return self._logger

    @property
    def importlib(self) -> ChannelImportLib:
        if not self._importlib:
            raise RuntimeError(f"channel is not running")
        return self._importlib

    @property
    def container(self) -> IoCContainer:
        """
        broker 所持有的 ioc 容器.
        """
        return self._container

    def prepare_container(self, container: IoCContainer) -> IoCContainer:
        # 重写这个函数完成自定义.
        return container

    async def fetch_sub_broker(self, path: ChannelFullPath) -> ChannelBroker | None:
        paths = Channel.split_channel_path_to_names(path)
        return await self.importlib.recursively_fetch_broker(self, paths)

    @property
    def id(self) -> str:
        """
        broker 的唯一 id.
        """
        return self._uid

    @property
    def name(self) -> str:
        """
        对应的 channel name.
        """
        return self._name

    # --- abstract -- #

    @abstractmethod
    async def on_start_up(self) -> None:
        pass

    # --- interface --- #

    def metas(self) -> dict[ChannelFullPath, ChannelMeta]:
        """
        返回 Channel 自身的 Meta.
        """
        if not self.is_running() or not self.is_connected():
            return {"": ChannelMeta.new_empty(self._uid, self.channel)}
        # 还是复制一份.
        if "" not in self._cached_metas:
            return {"": ChannelMeta.new_empty(self._uid, self.channel)}
        return self._get_cached_meta()

    def _get_cached_meta(self) -> dict[ChannelFullPath, ChannelMeta]:
        return {name: meta.model_copy() for name, meta in self._cached_metas.items()}

    def on_refresh_meta(self, callback: RefreshMetaCallback) -> None:
        self._on_refresh_meta_callbacks.append(callback)

    @abstractmethod
    async def _generate_metas(self, force: bool) -> dict[ChannelFullPath, ChannelMeta]:
        """
        重新生成 meta 数据对象.
        """
        pass

    async def refresh_metas(
            self,
            force: bool = True,
            callback: bool = True,
    ) -> None:
        """
        更新当前的 Channel Meta 信息. 递归创建所有子节点的 metas.
        """
        await self._refresh_meta_lock.acquire()
        try:
            if not self._starting or self._closing_event.is_set():
                return
            if not force and '' in self._cached_metas:
                # 完成过刷新.
                return
            # 生成时添加 ctx.
            ctx = ChannelCtx(self)
            metas = await ctx.run(self._generate_metas, force)
            self._cached_metas = metas
            # 创建异步的回调.
            if callback and self._on_refresh_meta_callbacks:
                for callback_fn in self._on_refresh_meta_callbacks:
                    if inspect.iscoroutinefunction(callback_fn):
                        _ = asyncio.create_task(callback_fn(metas))
                    else:
                        self._loop.run_in_executor(None, callback_fn, metas)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.logger.exception("%s refresh self meta failed %s", self.log_prefix, exc)
            # 出现异常后, 刷新一个异常的 meta.
        finally:
            self._refresh_meta_lock.release()
            self.logger.info(
                "%s refreshed meta", self.log_prefix,
            )

    # --- status --- #

    def is_running(self) -> bool:
        """
        是否已经启动了. 如果 Broker 被 close, is_running 为 false.
        """
        return self._started.is_set() and not self._closing_event.is_set()

    def is_available(self) -> bool:
        """
        当前 Channel 对于使用者而言, 是否可用.
        当一个 Broker 是 running & connected 状态下, 仍然可能会因为种种原因临时被禁用.
        """
        return self.is_running() and self.is_connected() and self._is_available()

    @abstractmethod
    def _is_available(self) -> bool:
        pass

    # --- on task done --- #

    def _parse_task(self, task: CommandTask) -> CommandTask | None:
        if task.done():
            return
        elif not self.is_running():
            self.logger.error(
                "%s failed task %s: not running", self.log_prefix, task.cid,
            )
            task.fail(CommandErrorCode.NOT_RUNNING.error(f"channel {self.name} not running"))
            return
        elif not self.is_connected():
            self.logger.info(
                "%s failed task %s: not connected", self.log_prefix, task.cid,
            )
            task.fail(CommandErrorCode.NOT_CONNECTED.error(f"channel {self.name} not connected"))
            return
        elif not self.is_available():
            self.logger.info(
                "%s failed task %s: not available", self.log_prefix, task.cid,
            )
            task.fail(CommandErrorCode.NOT_AVAILABLE.error(f"channel {self.name} not available"))
            return
        return task

    async def push_task_with_paths(self, paths: ChannelPaths, task: CommandTask) -> None:
        """
        基于路径将任务入栈.
        """
        task = self._parse_task(task)
        if task is None:
            return
        # 设置运行通道记录.
        # 设置 task id 到 pending map 里.
        self._add_task_done_callback(task)
        try:
            if self._defer_clear_mark:
                self._defer_clear_mark = False
                await self._clear()
            await self._push_task_with_paths(paths, task)
        except Exception as exc:
            self.logger.exception(exc)
            if not task.done():
                task.fail(exc)
            raise exc

    @abstractmethod
    async def _push_task_with_paths(self, paths: ChannelPaths, task: CommandTask) -> None:
        pass

    def on_task_done(self, callback: TaskDoneCallback) -> None:
        # 注册 task 回调.
        self._task_done_callbacks.append(callback)

    def _add_task_done_callback(self, task: CommandTask) -> None:
        if len(self._task_done_callbacks) > 0:
            task.add_done_callback(self._task_done_callback)

    def _task_done_callback(self, task: CommandTask) -> None:
        import inspect
        if not self.is_running():
            return
        if len(self._task_done_callbacks) == 0:
            return
        for callback in self._task_done_callbacks:
            if inspect.iscoroutinefunction(callback):
                # todo: 似乎要考虑线程安全.
                self._loop.create_task(callback(task))
            else:
                # 同步运行.
                self._loop.run_in_executor(None, callback, task)

    async def clear(self) -> None:
        self._defer_clear_mark = False
        await self._clear()

    @abstractmethod
    async def _clear(self) -> None:
        pass

    def defer_clear(self) -> None:
        self._defer_clear_mark = True

    # --- 开始与结束 --- #

    @contextlib.asynccontextmanager
    async def _container_ctx(self):
        self._container = self.prepare_container(self._container)
        await self._loop.run_in_executor(None, self._container.bootstrap)
        yield
        self._loop.run_in_executor(None, self._container.shutdown)

    @contextlib.asynccontextmanager
    async def _importlib_ctx(self):
        if self._importlib is None:
            _importlib = self._container.get(ChannelImportLib)
            if _importlib is None:
                _importlib = ChannelImportLib(self, self._container)
                self.container.set(ChannelImportLib, _importlib)
            self._importlib = _importlib
        if self._importlib.main is self:
            await self._importlib.start()
        yield
        if self._importlib.main is self:
            await self._importlib.close()

    @contextlib.asynccontextmanager
    async def _states_ctx(self):
        if self._state_store is None:
            state_store = self.container.get(StateStore)
            if state_store is None:
                state_store = BaseStateStore(owner=self._uid)
            self._state_store = state_store
        self._state_store.register(*self.default_states())
        await self._state_store.start()
        yield
        await self._state_store.close()

    @abstractmethod
    def default_states(self) -> list[State]:
        pass

    @contextlib.asynccontextmanager
    async def _start_and_close_ctx(self):
        ctx = ChannelCtx(self)
        cor = ctx.run(self.on_start_up)
        self.logger.info(
            "%s started", self.log_prefix,
        )
        await cor
        yield
        try:
            ctx = ChannelCtx(self)
            on_close_cor = ctx.run(self.on_close)
            await on_close_cor
        except Exception as e:
            self.logger.exception("%s close failed: %s", self.log_prefix, e)

    @abstractmethod
    async def on_close(self) -> None:
        pass

    @contextlib.asynccontextmanager
    async def _running_task_ctx(self):
        ctx = ChannelCtx(self)
        self._running_task = asyncio.create_task(ctx.run(self._execute_running_task))
        yield
        if self._running_task and not self._running_task.done():
            self._running_task.cancel()
            try:
                await self._running_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.exception("%s close running task failed %s", self.log_prefix, e)

    @abstractmethod
    async def on_running(self) -> None:
        pass

    async def _execute_running_task(self) -> None:
        try:
            await self.on_running()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception("%s keep_running_task failed: %s", self.log_prefix, e)
        finally:
            self.logger.info("%s keep_running_task finished", self.log_prefix)

    @contextlib.asynccontextmanager
    async def _main_loop_ctx(self):
        self._main_loop_task = asyncio.create_task(self._main_loop())
        yield
        try:
            await self.clear()
            if self._main_loop_task and not self._main_loop_task.done():
                self._main_loop_task.cancel()
                try:
                    await self._main_loop_task
                except asyncio.CancelledError:
                    pass
            self._main_loop_task = None
        except Exception as e:
            self.logger.exception(e)
            raise

    @abstractmethod
    async def _main_loop(self) -> None:
        pass

    def _async_exit_ctx_funcs(self) -> Iterable[Callable]:
        yield self._container_ctx
        yield self._importlib_ctx
        yield self._states_ctx
        yield self._start_and_close_ctx
        yield self._running_task_ctx
        yield self._main_loop_ctx

    async def start(self):
        """
        启动 Channel Broker.
        通常用 with statement 或 async exit stack 去启动.
        只会启动当前 channel 自身.
        """
        if self._starting:
            return
        self._starting = True
        self._loop = asyncio.get_running_loop()
        await self._exit_stack.__aenter__()
        for ctx_func in self._async_exit_ctx_funcs():
            await self._exit_stack.enter_async_context(ctx_func())
        if self.is_connected():
            await self.refresh_metas(force=False)
        self._started.set()
        return self

    async def wait_started(self) -> None:
        if self._closing_event.is_set():
            return
        await self._started.wait()

    async def wait_closed(self) -> None:
        await self._closed_event.wait()

    def close_sync(self) -> None:
        if not self.is_running():
            return
        # 运行关闭逻辑.
        self._loop.create_task(self.close())

    async def close(self):
        """
        关闭当前 broker. 同时阻塞销毁资源直到结束.
        只会关闭当前 channel 的 broker.
        """
        if self._closing_event.is_set():
            return
        self._closing_event.set()
        try:
            self.logger.info(
                "%s start to close", self.log_prefix,
            )
            # 停止所有行为.
            await self._exit_stack.aclose()
        finally:
            self._closed_event.set()
            if self._logger:
                self._logger.info(
                    "%s closed", self.log_prefix,
                )
            # 做必要的清空.
            self.destroy()

    def destroy(self) -> None:
        self._container = None
        # 防止互相持有.
        self._channel = None
        self._state_store = None
        self._logger = None
        self._on_refresh_meta_callbacks.clear()
        self._task_done_callbacks.clear()
        self._importlib = None

    # --- execute tasks --- #


class AbsChannelTreeBroker(AbsChannelBroker, ABC):

    # --- main loop --- #

    def __init__(
            self,
            *,
            channel: CHANNEL,
            container: IoCContainer | None = None,
            logger: LoggerItf | None = None
    ):
        super().__init__(
            channel=channel,
            container=container,
            logger=logger,
        )
        self._blocking_action_lock = asyncio.Lock()
        self._lifecycle_task: asyncio.Task | None = None
        self._pending_task_queue: asyncio.Queue[_TaskWithPaths | None] = asyncio.Queue()

        # 运行执行的并行任务.
        self._consuming_command_task: CommandTask | None = None
        self._executing_command_task: CommandTask | None = None
        self._executing_cmd_tasks: set[CommandTask] = set()
        self._idled_event = asyncio.Event()
        self._has_task_queued = asyncio.Event()

    def get_children_brokers(self) -> dict[str, ChannelBroker]:
        children = self.imported()
        result = {}
        for name, child in children.items():
            broker = self.importlib.get_channel_broker(child)
            if broker is not None and broker.is_running():
                result[name] = broker
        return result

    @abstractmethod
    def imported(self) -> dict[str, Channel]:
        """
        当前持有的子 Channel.
        """
        pass

    def get_child_broker(self, name: str) -> ChannelBroker | None:
        child = self.imported().get(name)
        if child is None:
            return None
        return self.importlib.get_channel_broker(child)

    def descendants(self) -> dict[ChannelFullPath, ChannelBroker]:
        return self.importlib.find_descendants(self.channel)

    def all_brokers(self) -> dict[ChannelFullPath, Self]:
        result: dict[ChannelFullPath, ChannelBroker] = {"": self}
        descendants = self.descendants()
        result.update(descendants)
        return result

    @abstractmethod
    async def _generate_self_meta(self) -> ChannelMeta:
        pass

    async def _generate_metas(self, force: bool) -> dict[ChannelFullPath, ChannelMeta]:
        self_meta = await self._generate_self_meta()
        new_cached_metas: dict[ChannelFullPath, ChannelMeta] = {"": self_meta}
        children_names, children_metas = await self._generate_children_metas(force)
        new_cached_metas.update(children_metas)
        # 终于完成更新.
        self_meta.children = children_names
        return new_cached_metas

    async def _generate_children_metas(self, force: bool) -> tuple[list[str], dict[ChannelFullPath, ChannelMeta]]:

        async def create_child_interfaces(
                _child_name: str,
                _child: Channel,
        ) -> tuple[str, dict[ChannelFullPath, ChannelMeta]] | None:
            try:
                child_broker = await self.importlib.get_or_create_channel_broker(_child)
                if not child_broker or not child_broker.is_running():
                    return None
                # 不强制生成.
                await child_broker.refresh_metas(callback=False, force=force)
                _interfaces = child_broker.metas()
                _result = {}
                for channel_path, _meta in _interfaces.items():
                    new_channel_path = Channel.join_channel_path(_child_name, channel_path)
                    _result[new_channel_path] = _meta
                return _child_name, _result

            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                raise
            except Exception as e:
                self._logger.exception(
                    "%s failed to create child %s interface: %s",
                    self.log_prefix, _child_name, e
                )
                raise

        children = self.imported()
        result = {}
        children_names = []
        if len(children) > 0:
            gathering = []
            for child_name, child in children.items():
                child_task = self._loop.create_task(create_child_interfaces(child_name, child))
                gathering.append(child_task)
            # 按顺序更新.
            if len(gathering) > 0:
                done = await asyncio.gather(*gathering)
                for r in done:
                    if isinstance(r, Exception):
                        self._logger.exception(
                            "%s failed to create child interface: %s",
                            self.log_prefix, r
                        )
                    elif r is None:
                        continue
                    else:
                        child_name, child_metas = r
                        children_names.append(child_name)
                        for _path, _descendant_meta in child_metas.items():
                            result[_path] = _descendant_meta
        return children_names, result

    def commands(self, available_only: bool = True) -> dict[ChannelFullPath, dict[str, Command]]:
        commands = self.self_commands(available_only).copy()
        result = {'': commands}
        for name, child in self.imported().items():
            child_broker = self.importlib.get_channel_broker(child)
            if child_broker and child_broker.is_running():
                child_commands = child_broker.commands(available_only)
                for further_path, command_map in child_commands.items():
                    new_full_path = Channel.join_channel_path(name, further_path)
                    result[new_full_path] = command_map
        return result

    def get_command(self, name: CommandUniqueName) -> Optional[Command]:
        chan, command_name = Command.split_uniquename(name)
        if chan == "":
            return self.get_self_command(command_name)
        broker = self.importlib.recursively_find_broker(self, chan)
        if broker is None:
            return None
        return broker.get_self_command(command_name)

    async def wait_idle(self) -> None:
        """
        阻塞等待到闲时.
        """
        if not self.is_running():
            return
        wait_1 = asyncio.create_task(self._idled_event.wait())
        wait_2 = asyncio.create_task(self._closing_event.wait())
        done, pending = await asyncio.wait([wait_1, wait_2], return_when=asyncio.FIRST_COMPLETED)
        for t in pending:
            t.cancel()

    # --- lifecycle --- #

    async def idle(self) -> None:
        """
        进入闲时状态.
        闲时状态指当前 Broker 及其 子 Channel 都没有 CommandTask 在运行的时候.
        """
        if not self.is_running():
            return
        await self._clear_lifecycle_task()
        await self._blocking_action_lock.acquire()
        try:
            await asyncio.sleep(0.0)
            ctx = ChannelCtx(self)
            on_idle_cor = ctx.run(self.on_idle)
            # idle 是一个在生命周期中单独执行的函数.
            task = asyncio.create_task(on_idle_cor)
            self._lifecycle_task = task
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._logger.exception(
                "%s idle task failed %s", self.log_prefix, exc
            )
            # 不返回.
        finally:
            self._blocking_action_lock.release()
            self.logger.info("%s idling", self.log_prefix)

    @abstractmethod
    async def on_idle(self) -> None:
        """
        进入闲时状态.
        闲时状态指当前 Broker 及其 子 Channel 都没有 CommandTask 在运行的时候.
        """
        pass

    async def _clear_lifecycle_task(self) -> None:
        # 终止阻塞中的任务.
        await self._blocking_action_lock.acquire()
        try:
            self._idled_event.clear()
            if self._lifecycle_task and not self._lifecycle_task.done():
                self._lifecycle_task.cancel()
                try:
                    await self._lifecycle_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.exception("%s clear lifecycle task failed: %s", self.log_prefix, e)
            self._lifecycle_task = None
        finally:
            self._blocking_action_lock.release()

    async def _wait_children_idled(self) -> None:
        async def wait_child_empty(_child: Channel):
            broker = await self._importlib.get_or_create_channel_broker(_child)
            if broker and broker.is_running():
                await broker.wait_idle()
            return

        wait_all = []
        children = self.imported()
        if len(children) > 0:
            for child in children.values():
                wait_all.append(wait_child_empty(child))
            _ = await asyncio.gather(*wait_all)

    def _is_children_idled(self) -> bool:
        children = self.imported()
        if len(children) > 0:
            for child in children.values():
                broker = self.importlib.get_channel_broker(child)
                if not broker.is_running():
                    continue
                elif not broker.is_idle():
                    return False
        return True

    def is_idle(self) -> bool:
        return self.is_running() and self._idled_event.is_set()

    async def _main_loop(self) -> None:
        try:
            await self.wait_started()
            while not self._closing_event.is_set():
                _pending_queue = self._pending_task_queue
                # 如果队列是空的, 则要看看是否能够启动 idle.
                if _pending_queue.empty() and not self._idled_event.is_set():
                    await asyncio.sleep(0)
                    if self._is_children_idled():
                        # 这种情况下就真的可以 idle 了.
                        await self.idle()
                        self._idled_event.set()
                        continue
                # 阻塞等待下一个结果.
                try:
                    item = await asyncio.wait_for(_pending_queue.get(), timeout=0.2)
                except asyncio.TimeoutError:
                    continue

                # 可能拿到了 clear 清空后的毒丸.
                if item is None:
                    self.logger.info("%s receive none from pending task queue", self.log_prefix)
                    continue
                # 拿到新命令后, 就清空生命周期函数.
                paths, task = item
                # handle task 函数是阻塞的, 这意味着:
                # 1. 它会阻塞后续拿到新的任务.
                # 2. 如果它执行了子任务, 其实不会阻塞.
                # 3. 如果它执行了 none-blocking 的任务, 也不会阻塞.
                # 4. 只有它执行的目标任务是自己的任务, 才会阻塞. 而且要阻塞等待儿孙们都执行完了, 才轮到自己执行.

                await self._consume_task(paths, task)
        except asyncio.CancelledError as e:
            # 允许被 cancel.
            self.logger.info("%s Cancel consuming pending task loop: %r", self.log_prefix, e)
        finally:
            self._closing_event.set()
            self.logger.info("%s Finished executing loop", self.log_prefix)

    async def _dispatch_children_task(self, paths: ChannelPaths, task: CommandTask) -> None:
        self._executing_command_task = task
        child_name = paths[0]
        # 子节点在路径上不存在.
        child = self.imported().get(child_name)
        if child is None:
            task.fail(CommandErrorCode.NOT_FOUND.error(f"channel `{task.chan}` not found"))
            return

        broker = await self.importlib.get_or_create_channel_broker(child)
        if broker is None:
            task.fail(CommandErrorCode.NOT_FOUND.error(f"channel `{task.chan}` not found"))
            return
        task.send_through.append(child_name)
        # 直接发送给子树.
        further_paths = paths[1:]
        await broker.push_task_with_paths(further_paths, task)

    async def _consume_task(self, paths: ChannelPaths, task: CommandTask) -> None:
        """
        尝试运行一个 task. 这个运行周期是全局唯一, 阻塞的.
        """
        self._consuming_command_task = task
        try:
            # 确保这个任务也可以被 clear 掉.
            await self._clear_lifecycle_task()
            await asyncio.sleep(0)
            # 检查是不是子节点的任务.
            if len(paths) > 0:
                await self._dispatch_children_task(paths, task)
                return

            # 执行任务, 并且解决回调的问题.
            await asyncio.sleep(0)
            # 执行任务.
            await self._execute_self_task(task)

        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.info("%s handle pending task exception: %r", self.log_prefix, e)
            # 所有在执行 handle pending task 阶段抛出的异常, 都不向上中断.
        finally:
            self._consuming_command_task = None

    async def _get_task_result(self, task: CommandTask) -> Any:
        # 准备执行.
        await asyncio.sleep(0)
        self.logger.info("%s start task %s", self.log_prefix, task.cid)
        # 初始化函数运行上下文.
        # 使用 dry run 来管理生命周期.
        async with ChannelCtx(self, task).in_ctx():
            return await task.dry_run()

    async def _execute_self_task(self, task: CommandTask, depth: int = 0) -> None:
        task.set_state(CommandTaskState.executing)
        task.exec_chan = self._name
        # 非阻塞函数不能返回 stack
        if depth > 10:
            task.fail(CommandErrorCode.INVALID_USAGE.error("stackoverflow"))
            return
        self._executing_cmd_tasks.add(task)
        # 确保 task 被执行了.
        asyncio_task = asyncio.create_task(self._ensure_task_executed(task, depth))
        if task.meta.interruptable:
            # 对于可被中断的任务, 它应该被放到 lifecycle task 里, 有新任务进来就会中断它.
            self._lifecycle_task = asyncio_task
        elif task.meta.blocking:
            # 阻塞等待 blocking 任务执行完毕.
            await asyncio_task

    async def _ensure_task_executed(self, task: CommandTask, depth: int) -> None:
        """
        运行属于自己这个 channel 的 task, 让它进入到 executing group 中.
        """
        try:
            task = self._parse_task(task)
            if task is None:
                return

            get_result_from_task = self._loop.create_task(self._get_task_result(task))
            origin_task_done = asyncio.create_task(task.wait(throw=False))
            wait_broker_close = asyncio.create_task(self._closing_event.wait())
            done, pending = await asyncio.wait(
                [origin_task_done, get_result_from_task, wait_broker_close],
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            if origin_task_done in done:
                # origin task 已经运行结束.
                return
            elif wait_broker_close in done:
                task.fail(CommandErrorCode.NOT_RUNNING.error("broker closed"))
                return
            result = await get_result_from_task
            # 如果返回值是 stack, 则意味着要循环堆栈.
            if isinstance(result, CommandResultStack):
                # 执行完所有的堆栈. 同时设置真实被执行的任务.
                await self._fulfill_task_with_its_result_stack(task, result, depth=depth)
            else:
                # 赋值给原来的 task.
                task.resolve(result)
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
            raise
        except Exception as e:
            if not task.done():
                task.fail(e)
            self.logger.error("%s task %s failed: %s", self.log_prefix, task.cid, e)
            raise
        finally:
            if not task.done():
                self.logger.info("%s failed to ensure task done: %s", self.log_prefix, task)
                task.fail(CommandErrorCode.UNKNOWN_ERROR.error(f"execution failed"))
            if task in self._executing_cmd_tasks:
                self._executing_cmd_tasks.remove(task)

    async def _fulfill_task_with_its_result_stack(
            self,
            owner: CommandTask,
            stack: CommandResultStack,
            depth: int = 0,
    ) -> None:
        try:
            if not owner.meta.blocking:
                owner.fail(CommandErrorCode.INVALID_USAGE.error(f"invalid command: none blocking task return stack"))
                return
            if depth > 10:
                owner.fail(CommandErrorCode.INVALID_USAGE.error("stackoverflow"))
                return

            self.logger.info(
                "%s Fulfilling task with stack, depth=%s task=%s",
                self.log_prefix, depth, owner,
            )
            # 遍历生成的新栈.
            async for sub_task in stack:
                await asyncio.sleep(0)
                if owner.done():
                    # 不要继续执行了.
                    break
                paths = Channel.split_channel_path_to_names(sub_task.chan)
                if len(paths) > 0:
                    # 发送给子孙了.
                    await self._dispatch_children_task(paths, sub_task)
                    continue

                # 递归阻塞等待任务被执行.
                await self._execute_self_task(sub_task, depth + 1)
                if sub_task.meta.blocking:
                    result = await sub_task
                    if isinstance(result, CommandResultStack):
                        # 递归执行
                        await self._fulfill_task_with_its_result_stack(sub_task, result, depth + 1)

            # 完成了所有子节点的调度后, 通知回调函数.
            # !!! 注意: 在这个递归逻辑中, owner 自行决定是否要等待所有的 child task 完成,
            #          如果有异常又是否要取消所有的 child task.
            await stack.callback(owner)
            return
        except Exception as e:
            # 不要留尾巴?
            # 有异常时, 同时取消所有动态生成的 task 对象. 包括发送出去的. 这样就不会有阻塞了.
            if not owner.done():
                self.logger.exception(
                    "%s Fulfill task stack failed, task=%s, exception=%s",
                    self.log_prefix, owner, e,
                )
                for child in stack.generated():
                    if not child.done():
                        child.fail(e)
                owner.fail(e)
            raise e

    async def _push_task_with_paths(self, paths: ChannelPaths, task: CommandTask) -> None:
        """
        基于路径将任务入栈.
        """
        try:
            # 是自己的, 而且是要立刻执行的任务.
            # call soon 这类任务
            await self._clear_lifecycle_task()
            if len(paths) == 0 and task.meta.call_soon:
                if task.meta.blocking:
                    # 需要立刻执行, 而且是一个阻塞类的任务, 则会清空所有运行中的任务. 这其中也递归地包含子节点的任务.
                    await self.clear()
                # 立刻将它放入 broker 的执行队列. 它会被尽快执行.
                await self._consume_task(paths, task)
                # 并不阻塞等待结果, 而是立刻返回.
                return

            # 普通的任务, 则会被丢入阻塞队列中排队执行.
            _queue = self._pending_task_queue
            # 入栈.
            _queue.put_nowait((paths, task))
            # set pending
            task.set_state(CommandTaskState.pending.value)
            self._has_task_queued.set()
        except asyncio.QueueFull:
            task.fail(CommandErrorCode.FAILED.error(f"channel queue is full, clear first"))

    async def _clear(self):
        await self._clear_pending_and_executing()

        async def clear_child(_child: Channel):
            child_broker = await self._importlib.get_or_create_channel_broker(_child)
            if child_broker and child_broker.is_running():
                await child_broker.clear()

        clear_tasks = []
        children = self.imported()
        for child in children.values():
            clear_tasks.append(clear_child(child))
        if len(clear_tasks) > 0:
            done = await asyncio.gather(*clear_tasks)
            for r in done:
                if isinstance(r, Exception):
                    self._logger.exception("%s clear child failed: %s", self.log_prefix, r)

    async def _clear_pending_and_executing(self) -> None:
        """
        当轨道命令被触发清空时候执行.
        """
        if not self._started.is_set() or self._closed_event.is_set():
            return
        await self._blocking_action_lock.acquire()
        try:
            await asyncio.sleep(0.0)
            _pending_task_queue = self._pending_task_queue
            self._pending_task_queue = asyncio.Queue()
            while not _pending_task_queue.empty():
                item = await _pending_task_queue.get()
                if item is not None:
                    paths, task = item
                    if not task.done():
                        task.fail(CommandErrorCode.CLEARED.error("cleared by broker"))
            _pending_task_queue.put_nowait(None)

            # 设置 task 为 fail 即可. 主循环永远会清除它.
            consuming_command_task = self._consuming_command_task
            if consuming_command_task is not None:
                if not consuming_command_task.done():
                    consuming_command_task.fail(CommandErrorCode.CLEARED.error(f"cleared by broker"))
            # 并行执行的 task 也需要被清除.
            if len(self._executing_cmd_tasks) > 0:
                for t in self._executing_cmd_tasks:
                    if not t.done():
                        t.fail(CommandErrorCode.CLEARED.error(f"cleared by broker"))
            self._executing_cmd_tasks.clear()
        except Exception as e:
            self.logger.exception("%s clear self failed: %s", self.log_prefix, e)
            raise
        finally:
            self._blocking_action_lock.release()
            self.logger.info("%s cleared", self.log_prefix)
