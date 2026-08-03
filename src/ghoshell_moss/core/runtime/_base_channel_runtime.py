import contextlib

import asyncio
from abc import ABC, abstractmethod
from typing import Optional, Iterable, TypeVar, Generic, Callable, Coroutine

import janus
from typing_extensions import Self

from ghoshell_container import IoCContainer, Container

from ghoshell_moss.core.concepts.command import (
    CommandTask,
    BaseCommandTask,
    CommandMeta,
)
from ghoshell_moss.core.concepts.channel import (
    ChannelCtx,
    Channel,
    ChannelMeta,
    TaskDoneCallback,
    ChannelRuntime,
    ChannelFullPath,
    ChannelPaths,
    ChannelScope,
    ChannelScopeType,
    ChannelScopeDefaultType,
    ChannelTree,
)
from ghoshell_moss.core.concepts.errors import CommandErrorCode
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_common.contracts import LoggerItf
from .tree import BaseChannelTree
import logging
import threading

__all__ = [
    "AbsChannelRuntime",
]

_ChannelId = str
CHANNEL = TypeVar("CHANNEL", bound=Channel)


class ChannelScopeImpl(ChannelScope):

    def __init__(
            self,
            task: CommandTask,
            scope_id: str,
            loop: asyncio.AbstractEventLoop,
            timeout: float | None,
            until: str,
    ):
        self._scope_id: str = scope_id
        self._channel: str = task.chan
        self._loop = loop
        self._committed: bool = False
        self._future = BaseCommandTask(
            chan=task.chan,
            meta=CommandMeta(name="__scope__"),
            func=None,
            tokens='',
            args=[],
            kwargs={},
        )
        # 预计 future 结束时, 清空所有的子任务.
        self._future.add_done_callback(self._clear_all_sub_tasks)
        self._sub_task_group: set[CommandTask] = set()
        self._timeout: float | None = timeout
        self._until: str = until
        self._tick_until_done_task: asyncio.Task | None = None
        self._stop_reason: str | None = None

        # 保证 task 和 scope 的生命周期同步.
        self._bind_task_to_scope(task)
        # 保证 scope 能在 task 被取消时感知到.
        task.add_done_callback(self._on_enter_and_exit_task_done_check)
        task.scope_id = self._scope_id

    @staticmethod
    def parse_kwargs(kwargs: dict) -> tuple[str, float | None]:
        until = kwargs.get("until", ChannelScopeDefaultType)
        if until not in ['flow', 'all', 'any']:
            raise ValueError(f"invalid until: {until}")
        timeout = kwargs.get("timeout", None)
        if timeout is not None:
            timeout = float(timeout)
        return until, timeout

    def add_sub_scope(self, scope: 'ChannelScopeImpl') -> None:
        self.add_task(scope._future)

    # bind enter task
    def _on_enter_and_exit_task_done_check(self, _task: CommandTask) -> None:
        if self.is_closed():
            return
        if _task.cancelled():
            self.close(stop_reason="Scope Cancelled")
        elif _task.is_critical_failed():
            self.close("Scope failed to start")

    @property
    def scope_id(self) -> str:
        return self._scope_id

    def add_task(self, task: CommandTask) -> CommandTask:
        bound = self._bind_task_to_scope(task)
        if not bound:
            return task
        self._sub_task_group.add(task)
        if self._until == 'any':
            self._ensure_any_task_done_then_cancel_the_scope(task)
        return task

    def _bind_task_to_scope(self, scope_task: CommandTask) -> bool:
        if scope_task.done():
            return False
        if self._future.done() or self._committed:
            if not scope_task.done():
                scope_task.cancel("Scope Closed")
            return False

        def _on_scope_close_clear_sub_task(_ft: CommandTask):
            nonlocal scope_task
            if not scope_task.done():
                scope_task.cancel("Scope Closed")

        self._future.add_done_callback(_on_scope_close_clear_sub_task)
        # 反向绑定, 如果发生致命异常, 就清空  scope
        scope_task.add_done_callback(self._on_sub_future_critical_error)
        scope_task.scope_id = self._scope_id
        return True

    def _on_sub_future_critical_error(self, _ft: CommandTask) -> None:
        if _ft.is_critical_failed() and not self._future.done():
            self._future.fail(CommandErrorCode.CRITICAL.error("Sub Scope critical failed"))

    def _ensure_any_task_done_then_cancel_the_scope(self, task: CommandTask) -> None:
        if task.done() or self.is_closed():
            return

        def _on_any_sub_task_done(_task: CommandTask) -> None:
            # make sure future is done
            self.close(stop_reason="Scope closed on task done")

        task.add_done_callback(_on_any_sub_task_done)

    def commit(self, task: CommandTask) -> CommandTask:
        if not self._committed:
            task.add_done_callback(self._on_enter_and_exit_task_done_check)
            task.scope_id = self._scope_id
            self._committed = True
        else:
            # 理论上永不发生, 做容错.
            task.cancel("Scope already committed")
        task.scope_id = self._scope_id
        return task

    def is_commited(self) -> bool:
        return self._committed

    async def tick(self, *, until: ChannelScopeType, timeout: float | None = None) -> None:
        """启动 scope 的生命周期检查. """
        if not until in ['flow', 'all', 'any']:
            raise ValueError(f"invalid argument until=`{until}`")
        elif timeout is not None:
            # raise type error or value error?
            timeout = float(timeout)
        # 完成赋值.
        self._timeout = timeout
        self._until = until
        if self._until == 'any':
            # 检查一下是否有已经完成的 task, 如果是直接 close.
            for task in self._sub_task_group:
                if task.done():
                    self.close()
                    return

        if self._timeout is not None and self._timeout > 0:
            async def _tick_until_done() -> None:
                await asyncio.sleep(timeout)
                if not self.is_closed():
                    self.close("Scope Timeout")

            # 正式开始计时.
            asyncio.create_task(_tick_until_done())

    async def wait_close(self) -> str | None:
        """等待 scope 运行结束"""
        if self.is_closed():
            return self._stop_reason
        try:
            wait_group = []
            if self._until == 'any':
                for task in self._sub_task_group:
                    if task.done():
                        return None
                    wait_group.append(task)
            elif self._until == 'flow':
                for task in self._sub_task_group:
                    if task.chan == self._channel:
                        wait_group.append(task)
                if len(wait_group) == 0:
                    # 容错逻辑.
                    wait_group = self._sub_task_group.copy()
            else:
                wait_group = self._sub_task_group.copy()

            if len(wait_group) > 0:
                if self._until == 'any':
                    await asyncio.wait(wait_group, return_when=asyncio.FIRST_COMPLETED)
                    return self._stop_reason
                else:
                    _ = await asyncio.gather(*wait_group, return_exceptions=True)
                    return self._stop_reason
            return None
        finally:
            self.close()

    def is_closed(self) -> bool:
        return self._future.done()

    def close(self, stop_reason: str = '') -> None:
        if not self._future.done():
            self._stop_reason = stop_reason or None
            if stop_reason:
                self._future.cancel(self._stop_reason or '')
            else:
                self._future.resolve(None)

    def _clear_all_sub_tasks(self, _ft) -> None:
        """自身 scope 结束时, 清空所有的状态."""
        if self._tick_until_done_task is not None:
            self._tick_until_done_task.cancel()
        sub_task_group = self._sub_task_group.copy()
        self._sub_task_group.clear()
        for task in sub_task_group:
            if not task.done():
                task.cancel()


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
        container: IoCContainer = container or self.create_raw_container(self._name, self._uid)
        self._container = self.prepare_container(container)
        self._logger: LoggerItf | None = logger
        # import lib 是最重要的.
        self._tree: BaseChannelTree | None = None

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
        self._on_task_done_callbacks: list[TaskDoneCallback] = []

        # compiling loop
        self._compiling_loop_task: asyncio.Task | None = None
        self._on_compile_task_queue: janus.Queue[tuple[ChannelPaths, CommandTask]] = janus.Queue()
        self._compiling_task: CommandTask | None = None

        self._exit_stack = contextlib.AsyncExitStack()
        # log_prefix
        self.log_prefix = "<Channel `%s` cls=%s id=%s name=%s>" % (
            self._name, self.__class__.__name__, self._uid, self.name
        )
        self._channel_scopes: dict[str, ChannelScopeImpl] = {}
        self._channel_scope_change_lock = threading.Lock()
        self._uncommitted_scopes: list[str] = []

    def __repr__(self):
        return self.log_prefix

    @classmethod
    def create_raw_container(cls, name: str, uid: str = '') -> Container:
        return Container(name="Channel/%s/%s" % (name, uid or 'uid-missed'))

    def open_scope(self, task: CommandTask) -> None:
        if not self.is_running():
            task.fail(CommandErrorCode.NOT_RUNNING.error("channel is not running"))
            return
        try:
            until, timeout = ChannelScopeImpl.parse_kwargs(task.kwargs)
            task.kwargs["until"] = until
            task.kwargs["timeout"] = timeout
            new_scope = ChannelScopeImpl(task, scope_id=task.cid, loop=self._loop, until=until, timeout=timeout)
        except Exception as e:
            task.fail(e)
            self._logger.exception(
                "%s failed to open scope for task %s, %s, error: %e",
                self.log_prefix, task.caller_name(), task.cid, e
            )
            return
        last = self.get_active_scope(None, pop=False)
        if last is not None:
            last.add_sub_scope(new_scope)
        with self._channel_scope_change_lock:
            self._channel_scopes[task.cid] = new_scope
            self._uncommitted_scopes.append(new_scope.scope_id)
        self._logger.info(
            "%s open scope for task %s, scope_id=%s",
            self.log_prefix, task.caller_name(), new_scope.scope_id,
        )

    def get_active_scope(self, scope_id: str | None, pop: bool) -> ChannelScopeImpl | None:
        if scope_id is None:
            with self._channel_scope_change_lock:
                while len(self._uncommitted_scopes) > 0:
                    scope_id = self._uncommitted_scopes[-1]
                    scope = self._channel_scopes.get(scope_id)
                    if scope is not None and scope.is_closed():
                        self._uncommitted_scopes.pop()
                        self._channel_scopes.pop(scope_id, None)
                        scope_id = None
                    else:
                        break
                if len(self._uncommitted_scopes) > 0:
                    scope_id = self._uncommitted_scopes[-1]
                else:
                    scope_id = None
        if scope_id is None:
            return None
        if pop:
            # 同步轻逻辑.
            with self._channel_scope_change_lock:
                if scope_id in self._channel_scopes:
                    scope = self._channel_scopes.pop(scope_id)
                    return scope
            return None
        else:
            return self._channel_scopes.get(scope_id, None)

    def commit_scope(self, task: CommandTask) -> None:
        with self._channel_scope_change_lock:
            while len(self._uncommitted_scopes) > 0:
                scope_id = self._uncommitted_scopes.pop()
                scope = self._channel_scopes.get(scope_id)
                if scope is not None and scope.is_closed():
                    self._channel_scopes.pop(scope_id, None)
                    self._logger.info(
                        "%s skip closed scope %s during commit",
                        self.log_prefix, scope_id,
                    )
                    continue
                break
            else:
                scope_id = None
                scope = None
        if scope_id is None or scope is None:
            task.cancel("Scope Closed")
            self._logger.info(
                "%s failed to commit scope %s with task %s, cid=%s",
                self.log_prefix, scope_id, task.caller_name(), task.cid,
            )
            return
        scope.commit(task)
        task.scope_id = scope.scope_id
        self._logger.info(
            "%s commit scope %s with task %s, cid=%s",
            self.log_prefix, scope_id, task.caller_name(), task.cid,
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
        if not self._tree:
            raise RuntimeError(f"channel is not running")
        return self._tree

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

    def _enqueue_task_with_paths(self, paths: ChannelPaths, task: CommandTask) -> None:
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

    def _parse_runtime_status_for_task(self, task: CommandTask) -> CommandTask:
        if task is None:
            raise RuntimeError(f"received task is None")
        if task.done():
            return task
        elif not self.is_running():
            self.logger.error(
                "%s failed task %s: not running",
                self.log_prefix,
                task.cid,
            )
            task.fail(CommandErrorCode.NOT_RUNNING.error(f"channel `{self.name}` is not running"))
            return task
        elif not self.is_connected():
            self.logger.info(
                "%s failed task %s: not connected",
                self.log_prefix,
                task.cid,
            )
            task.fail(CommandErrorCode.NOT_CONNECTED.error(f"channel {self.name} not connected"))
            return task
        elif not self.is_available():
            self.logger.info(
                "%s failed task %s: not available",
                self.log_prefix,
                task.cid,
            )
            task.fail(CommandErrorCode.NOT_AVAILABLE.error(f"channel {self.name} not available"))
            return task
        return task

    async def _compiled_task_loop(self) -> None:
        while not self._closing_event.is_set():
            try:
                queue = self._on_compile_task_queue.async_q
                paths, task = await queue.get()
                if task is None:
                    continue
                task = self._parse_runtime_status_for_task(task)
                if task is None or task.done():
                    continue
                self._compiling_task = task
                self._add_task_done_callback(task)
                if task is None:
                    # the task is not registered, shall raise invalid error to it.
                    continue
                if not task.is_bare_task():
                    # 只有非 bare 才执行 on compiled.
                    task.on_compiled()
                # prepare to send
                await self._consume_compiled_task_with_paths(paths, task)

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
    async def _consume_compiled_task_with_paths(self, paths: ChannelPaths, task: CommandTask) -> None:
        """
        push the task to the real handling loop with paths
        """
        pass

    def on_task_done(self, callback: TaskDoneCallback) -> None:
        # 注册 task 回调.
        self._on_task_done_callbacks.append(callback)

    def _add_task_done_callback(self, task: CommandTask) -> None:
        if len(self._on_task_done_callbacks) > 0:
            task.add_done_callback(self._task_done_callback)

    def _task_done_callback(self, task: CommandTask) -> None:
        if not self.is_running():
            return
        if len(self._on_task_done_callbacks) == 0:
            return
        for callback in self._on_task_done_callbacks:
            try:
                callback(task)
            except Exception as exc:
                self.logger.exception(
                    "%s on task done callback %s failed: %s",
                    self.log_prefix, callback, exc,
                )

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
        if len(self._channel_scopes) > 0:
            with self._channel_scope_change_lock:
                self._uncommitted_scopes.clear()
                scopes = self._channel_scopes.copy()
                self._channel_scopes.clear()
            if len(scopes) > 0:
                for scope in scopes.values():
                    scope.close('Scope Cleared')
        await self._clear_own()

    @abstractmethod
    async def _clear_own(self) -> None:
        pass

    # --- 开始与结束 --- #

    @contextlib.asynccontextmanager
    async def _ensure_channel_tree_binding_ctx(self):
        try:
            if self._tree is None:
                _tree = self._container.get(ChannelTree)
                if _tree is None:
                    _tree = BaseChannelTree(self, self._container)
                    self.container.set(BaseChannelTree, _tree)
                    self.container.set(ChannelTree, _tree)
                self._tree = _tree
            yield
        finally:
            if self._tree.main is self:
                await self._tree.close()

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
            self._compiling_loop_task = self._loop.create_task(self._compiled_task_loop())
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
        yield self._ensure_channel_tree_binding_ctx
        yield self._start_and_close_ctx
        yield self._running_task_ctx
        yield self._main_loop_ctx
        yield self._clear_runtime_asyncio_tasks

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
        self._started.set()
        # 拥有 tree 的根节点的话, 需要启动.
        if self._tree.main is self:
            # 启动 channel tree.  只有当自己是根节点的时候才有资格.
            await self._tree.start()
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
        self._on_task_done_callbacks.clear()
        self._channel = None
        self._tree = None
