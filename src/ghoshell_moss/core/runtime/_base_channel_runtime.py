import contextlib

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Iterable, TypeVar, Generic, Callable, Coroutine
from typing_extensions import Self

from ghoshell_container import IoCContainer, Container

from ghoshell_moss.core.concepts.command import (
    CommandTask,
)
from ghoshell_moss.core.concepts.channel import (
    ChannelCtx,
    Channel,
    ChannelMeta,
    TaskDoneCallback,
    ChannelRuntime,
    ChannelFullPath,
    ChannelPaths,
)
from ghoshell_moss.core.concepts.errors import CommandErrorCode
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_common.contracts import LoggerItf
from ._import_lib import BaseChannelTree
import logging

__all__ = ["AbsChannelRuntime"]

_ChannelId = str
CHANNEL = TypeVar("CHANNEL", bound=Channel)


class AbsChannelRuntime(Generic[CHANNEL], ChannelRuntime, ABC):
    """
    实现基础的 Channel Runtime, 用来给所有的 Runtime 提供基准的生命周期.
    """

    def __init__(
            self,
            *,
            channel: CHANNEL,
            container: IoCContainer | None = None,
            logger: LoggerItf | None = None,
    ):
        self._channel: CHANNEL = channel
        self._name = channel.name()
        self._uid = channel.id()
        container: IoCContainer = container or Container(name="Channel/%s/%s" % (self._name, self._uid))
        self._container = self.prepare_container(container)
        self._logger: LoggerItf | None = logger
        # import lib 是最重要的.
        self._importlib: BaseChannelTree | None = None

        self._logger: LoggerItf | None = logger

        self._starting = False
        self._started = asyncio.Event()
        self._channel_running_lifecycle_task: Optional[asyncio.Task] = None
        # 用线程安全的事件. 考虑到 runtime 未来可能会跨线程被使用.
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()

        self._own_metas_cache: dict[ChannelFullPath, ChannelMeta] = {}
        # 可以注册监听, 监听 refresh meta 动作.
        self._refresh_meta_lock = asyncio.Lock()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._main_loop_task: Optional[asyncio.Task] = None
        # maintain a task group for cancel them during runtime.
        self._runtime_asyncio_task_group: set[asyncio.Task] = set()
        # register task done callback
        self._task_done_callbacks: list[TaskDoneCallback] = []
        self._exit_stack = contextlib.AsyncExitStack()
        # log_prefix
        self.log_prefix = "[Channel `%s`][%s][%s] " % (self._name, self.__class__.__name__, self._uid)

    @property
    def channel(self) -> CHANNEL:
        return self._channel

    @property
    def logger(self) -> LoggerItf:
        if self._logger is None:
            # 日志总要有吧.
            self._logger = self._container.get(LoggerItf) or logging.getLogger("moss")
        return self._logger

    @property
    def tree(self) -> BaseChannelTree:
        if not self._importlib:
            raise RuntimeError(f"channel is not running")
        return self._importlib

    @property
    def container(self) -> IoCContainer:
        """
        runtime 所持有的 ioc 容器.
        """
        return self._container

    def prepare_container(self, container: IoCContainer) -> IoCContainer:
        # 重写这个函数完成自定义.
        return container

    async def fetch_sub_runtime(self, path: ChannelFullPath) -> ChannelRuntime | None:
        paths = Channel.split_channel_path_to_names(path)
        return await self.tree.recursively_fetch_runtime(self, paths)

    @property
    def id(self) -> str:
        """
        runtime 的唯一 id.
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

    def own_metas(self) -> dict[ChannelFullPath, ChannelMeta]:
        return self._own_metas_cache

    def metas(self) -> dict[ChannelFullPath, ChannelMeta]:
        """
        返回 Channel 自身的 Meta.
        """
        if not self.is_running() or not self.is_connected():
            return {"": ChannelMeta.new_empty(self._uid, self.channel)}
        own_metas = self.own_metas()
        # 还是复制一份.
        if "" not in own_metas:
            return {"": ChannelMeta.new_empty(self._uid, self.channel)}
        metas = own_metas.copy()
        self_meta = metas[""]

        # 递归获取.
        children_names = self_meta.children
        children = self.sub_channels()
        if len(children) == 0:
            return metas
        for child_name, child in children.items():
            child_runtime = self._importlib.get_channel_runtime(child)
            if not child_runtime or not child_runtime.is_running():
                continue
            if child_name not in children_names:
                children_names.append(child_name)
            descendant_metas = child_runtime.metas()
            for full_path, meta in descendant_metas.items():
                new_full_path = Channel.join_channel_path(child_name, full_path)
                if new_full_path in metas:
                    continue
                metas[new_full_path] = meta

        self_meta.children = children_names
        return metas

    async def refresh_own_metas(self, force: bool = False) -> None:
        ctx = ChannelCtx(self)
        self._own_metas_cache = await ctx.run(self._generate_own_metas, force)

    @abstractmethod
    async def _generate_own_metas(self, force: bool) -> dict[ChannelFullPath, ChannelMeta]:
        """
        重新生成 meta 数据对象.
        """
        pass

    def refresh_metas(
            self,
    ) -> asyncio.Future[None]:
        """
        更新当前的 Channel Meta 信息. 递归创建所有子节点的 metas.
        """
        if not self._starting or self._closing_event.is_set():
            f = asyncio.Future()
            f.set_result(None)
            return f
        if not self.is_connected():
            f = asyncio.Future()
            f.set_result(None)
            return f
        return self.tree.refresh(self.channel.id(), wait=True)
        # await self._refresh_meta_lock.acquire()
        # try:
        #     if not self._starting or self._closing_event.is_set():
        #         return
        #     if not self.is_connected():
        #         return
        #
        #     # 生成时添加 ctx.
        #     await self.refresh_own_metas(force=True)
        #     # 创建异步的回调.
        #     await self._refresh_children_metas()
        # except asyncio.CancelledError:
        #     return
        # except Exception as exc:
        #     self.logger.exception("%s refresh self meta failed %s", self.log_prefix, exc)
        #     # 出现异常后, 刷新一个异常的 meta.
        # finally:
        #     self._refresh_meta_lock.release()
        #     self.logger.info(
        #         "%s refreshed meta",
        #         self.log_prefix,
        #     )

    async def _refresh_children_metas(self) -> None:
        children = self.sub_channels()
        if len(children) == 0:
            return
        refreshing = []
        for child in children.values():
            runtime = self._importlib.get_channel_runtime(child)
            if not runtime or not runtime.is_running():
                continue
            refreshing.append(runtime.refresh_metas())
        if len(refreshing) > 0:
            done = await asyncio.gather(*refreshing, return_exceptions=True)
            for t in done:
                if isinstance(t, Exception):
                    self.logger.error(f"%s refresh children meta failed %s", self.log_prefix, t)

    # --- status --- #

    def is_running(self) -> bool:
        """
        是否已经启动了. 如果 Runtime 被 close, is_running 为 false.
        """
        return self._started.is_set() and not self._closing_event.is_set()

    def is_available(self) -> bool:
        """
        当前 Channel 对于使用者而言, 是否可用.
        当一个 Runtime 是 running & connected 状态下, 仍然可能会因为种种原因临时被禁用.
        """
        return self.is_running() and self.is_connected() and self._is_available()

    @abstractmethod
    def _is_available(self) -> bool:
        pass

    # --- on task done --- #

    def _parse_task(self, task: CommandTask) -> CommandTask | None:
        if task is None:
            return None
        if task.done():
            return None
        elif not self.is_running():
            self.logger.error(
                "%s failed task %s: not running",
                self.log_prefix,
                task.cid,
            )
            task.fail(CommandErrorCode.NOT_RUNNING.error(f"channel {self.name} not running"))
            return None
        elif not self.is_connected():
            self.logger.info(
                "%s failed task %s: not connected",
                self.log_prefix,
                task.cid,
            )
            task.fail(CommandErrorCode.NOT_CONNECTED.error(f"channel {self.name} not connected"))
            return None
        elif not self.is_available():
            self.logger.info(
                "%s failed task %s: not available",
                self.log_prefix,
                task.cid,
            )
            task.fail(CommandErrorCode.NOT_AVAILABLE.error(f"channel {self.name} not available"))
            return None
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
            # 准备入参.
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
                self.create_asyncio_task(callback(task))
            else:
                # 同步运行.
                self._loop.run_in_executor(None, callback, task)

    @abstractmethod
    async def clear_own(self) -> None:
        pass

    # --- 开始与结束 --- #

    @contextlib.asynccontextmanager
    async def _importlib_ctx(self):
        try:
            if self._importlib is None:
                _importlib = self._container.get(BaseChannelTree)
                if _importlib is None:
                    _importlib = BaseChannelTree(self, self._container)
                    self.container.set(BaseChannelTree, _importlib)
                self._importlib = _importlib
            yield
        finally:
            if self._importlib.main is self:
                await self._importlib.close()

    @contextlib.asynccontextmanager
    async def _start_and_close_ctx(self):
        ctx = ChannelCtx(self)
        cor = ctx.run(self.on_start_up)
        self.logger.info(
            "%s started",
            self.log_prefix,
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
        self._channel_running_lifecycle_task = asyncio.create_task(ctx.run(self._execute_running_task))
        yield
        if self._channel_running_lifecycle_task and not self._channel_running_lifecycle_task.done():
            self._channel_running_lifecycle_task.cancel()
            try:
                await self._channel_running_lifecycle_task
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
            self.logger.debug("%s keep_running_task finished", self.log_prefix)

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

    @contextlib.asynccontextmanager
    async def _clear_runtime_asyncio_tasks(self):
        yield
        tasks = self._runtime_asyncio_task_group.copy()
        self._runtime_asyncio_task_group.clear()
        await_tasks = []
        for t in tasks:
            if t.done():
                continue
            t.cancel()
            await_tasks.append(t)
        for t in await_tasks:
            try:
                await t
            except asyncio.CancelledError:
                pass

    @abstractmethod
    async def _main_loop(self) -> None:
        pass

    def create_asyncio_task(self, cor: Coroutine) -> asyncio.Task:
        """
        create asyncio task during runtime
        """
        if self._loop is None:
            raise RuntimeError('channel not running')
        task = self._loop.create_task(cor)
        self._runtime_asyncio_task_group.add(task)
        task.add_done_callback(self._remove_done_asyncio_task)
        return task

    def _remove_done_asyncio_task(self, task: asyncio.Task) -> None:
        if task in self._runtime_asyncio_task_group:
            self._runtime_asyncio_task_group.remove(task)

    def _async_exit_ctx_funcs(self) -> Iterable[Callable]:
        yield self._importlib_ctx
        yield self._start_and_close_ctx
        yield self._running_task_ctx
        yield self._main_loop_ctx

    async def _main_runtime_loop(self) -> None:
        async with contextlib.AsyncExitStack() as ctx:
            for ctx_func in self._async_exit_ctx_funcs():
                await self._exit_stack.enter_async_context(ctx_func())
                self.logger.debug("%s context stack %s entered", self.log_prefix, ctx_func)
            if self.is_connected():
                pass
            self._started.set()
            self.logger.info("%s started", self.log_prefix)
            await self._closing_event.wait()

    async def start(self) -> Self:
        """
        启动 Channel Runtime.
        通常用 with statement 或 async exit stack 去启动.
        只会启动当前 channel 自身.
        """
        if self._starting:
            return self
        self._starting = True
        self._loop = asyncio.get_running_loop()
        await self._exit_stack.__aenter__()
        for ctx_func in self._async_exit_ctx_funcs():
            await self._exit_stack.enter_async_context(ctx_func())
            self.logger.debug("%s start stack %s entered", self.log_prefix, ctx_func)
        # 递归启动子节点.
        self._started.set()
        # 拥有 importlib 的根节点的话, 需要启动.
        if self._importlib.main is self:
            await self._importlib.start()
        self.logger.info("%s started", self.log_prefix)
        return self

    async def _start_sub_channels(self) -> None:
        children = self.sub_channels()
        if len(children) == 0:
            return

        async def _start_child(_channel: Channel):
            runtime = await self._importlib.compile_channel(_channel)
            if runtime is not None:
                await runtime.wait_started()

        start_all = []
        for child in children.values():
            start_all.append(_start_child(child))
        # 递归启动.
        done = await asyncio.gather(*start_all, return_exceptions=True)
        for t in done:
            if isinstance(t, Exception):
                self.logger.exception("%s failed to start sub channel %s", self.log_prefix, t)

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
        关闭当前 runtime. 同时阻塞销毁资源直到结束.
        只会关闭当前 channel 的 runtime.
        """
        if self._closing_event.is_set():
            return
        self._closing_event.set()
        try:
            self.logger.info(
                "%s begin to close",
                self.log_prefix,
            )
            # 停止所有行为.
            await self._exit_stack.aclose()
        finally:
            self._closed_event.set()
            if self._logger:
                self._logger.info(
                    "%s closed",
                    self.log_prefix,
                )
            # 做必要的清空.
            self.destroy()

    def destroy(self) -> None:
        # 防止互相持有.
        self._task_done_callbacks.clear()
        del self._channel
        del self._importlib
