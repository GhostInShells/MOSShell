import contextlib

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Iterable, TypeVar, Generic, Callable, Coroutine

import janus
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
from .tree import BaseChannelTree
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
        self._refreshing_task: Optional[asyncio.Task] = None

        self._own_metas_cache: dict[ChannelFullPath, ChannelMeta] = {}
        # 可以注册监听, 监听 refresh meta 动作.
        self._refresh_meta_lock = asyncio.Lock()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._main_loop_task: Optional[asyncio.Task] = None
        # maintain a task group for cancel them during runtime.
        self._runtime_asyncio_task_group: set[asyncio.Task] = set()
        # register task done callback
        self._task_done_callbacks: list[TaskDoneCallback] = []

        # compiling loop
        self._compiling_loop_task: asyncio.Task | None = None
        self._on_compile_task_queue: janus.Queue[tuple[ChannelPaths, CommandTask]] = janus.Queue()
        self._compiling_task: CommandTask | None = None

        self._exit_stack = contextlib.AsyncExitStack()
        # log_prefix
        self.log_prefix = "<Channel `%s` cls=%s id=%s name=%s>" % (
            self._name, self.__class__.__name__, self._uid, self.name
        )

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
    async def on_startup(self) -> None:
        """
        启动时函数.
        """
        pass

    # --- interface --- #

    def own_metas(self) -> dict[ChannelFullPath, ChannelMeta]:
        return self._own_metas_cache

    def refresh_own_metas(self) -> asyncio.Future[None]:
        """
        make sure refresh run once at a time
        """
        if self._refreshing_task is not None and not self._refreshing_task.done():
            return self._refreshing_task
        self._refreshing_task = self._loop.create_task(self._refresh_own_metas())
        return self._refreshing_task

    async def _refresh_own_metas(self) -> None:
        ctx = ChannelCtx(self)
        self._own_metas_cache = await ctx.run(self._generate_own_metas)

    @abstractmethod
    async def _generate_own_metas(self) -> dict[ChannelFullPath, ChannelMeta]:
        """
        重新生成 meta 数据对象.
        """
        pass

    def push_task(self, *tasks: CommandTask) -> None:
        for task in tasks:
            paths = Channel.split_channel_path_to_names(task.chan)
            self.push_task_with_paths(paths, task)

    def push_task_with_paths(self, paths: ChannelPaths, task: CommandTask) -> None:
        if not self.is_running():
            return None
        self._on_compile_task_queue.sync_q.put_nowait((paths, task))
        return None

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

    async def _on_task_compile_loop(self) -> None:
        while not self._closing_event.is_set():
            try:
                queue = self._on_compile_task_queue.async_q
                paths, task = await queue.get()
                task = self._parse_task(task)
                if task is None or task.done():
                    continue
                self._compiling_task = task
                self._add_task_done_callback(task)
                task.on_compiled()
                # prepare to send
                await self._consume_task_with_paths(paths, task)
                await asyncio.sleep(0.0)

            except janus.AsyncQueueShutDown:
                # shutdown the old queue.
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.logger.exception("%s prepare to compile task failed: %s", self.log_prefix, exc)
            finally:
                self._compiling_task = None

        self.logger.info("%s compile task finished", self.log_prefix)

    @abstractmethod
    async def _consume_task_with_paths(self, paths: ChannelPaths, task: CommandTask) -> None:
        """
        push the task to the real handling loop with paths
        """
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

    async def clear_own(self) -> None:
        # shutdown the compiling loop.
        old_queue = self._on_compile_task_queue
        self._on_compile_task_queue = janus.Queue()
        cleared_err = CommandErrorCode.CLEARED.error("cleared")
        while not old_queue.sync_q.empty():
            paths, item = old_queue.sync_q.get_nowait()
            if item and not item.done():
                item.fail(cleared_err)
        while not old_queue.async_q.empty():
            paths, item = old_queue.async_q.get_nowait()
            if item and not item.done():
                item.fail(cleared_err)
        old_queue.shutdown()
        if self._compiling_task is not None:
            if not self._compiling_task.done():
                self._compiling_task.fail(cleared_err)
        await self._clear_own()

    @abstractmethod
    async def _clear_own(self) -> None:
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
        try:
            ctx = ChannelCtx(self)
            cor = ctx.run(self.on_startup)
            self.logger.info(
                "%s started",
                self.log_prefix,
            )
            await cor
            yield
        finally:
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
        try:
            ctx = ChannelCtx(self)
            self._channel_running_lifecycle_task = asyncio.create_task(ctx.run(self._execute_running_task))
            yield
        finally:
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
        try:
            self._compiling_loop_task = self._loop.create_task(self._on_task_compile_loop())
            self._main_loop_task = self._loop.create_task(self._main_loop())
            yield
        finally:
            try:
                await self.clear()
                if self._main_loop_task and not self._main_loop_task.done():
                    self._main_loop_task.cancel()
                    try:
                        await self._main_loop_task
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        self.logger.exception("%s cancel main_loop_task failed: %s", self.log_prefix, e)
                self._main_loop_task = None
                self._on_compile_task_queue.shutdown(immediate=True)
                if self._compiling_loop_task and not self._compiling_loop_task.done():
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._compiling_loop_task
            except Exception as e:
                self.logger.exception(e)
                raise

    @contextlib.asynccontextmanager
    async def _clear_runtime_asyncio_tasks(self):
        try:
            yield
        finally:
            tasks = self._runtime_asyncio_task_group.copy()
            self._runtime_asyncio_task_group.clear()
            await_tasks = []
            for t in tasks:
                if t.done():
                    continue
                t.cancel()
                await_tasks.append(t)
            if len(await_tasks) > 0:
                await asyncio.gather(*await_tasks, return_exceptions=True)

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
        self._runtime_asyncio_task_group.discard(task)

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
