from .channel import ChannelBroker
import asyncio
import contextvars
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable, Coroutine
from typing import (
    Optional, Iterable,
)

from ghoshell_container import IoCContainer, Container

from ghoshell_moss.core.concepts.command import CommandTask, CommandTaskStateType
from ghoshell_moss.core.concepts.states import StateStore, BaseStateStore
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_common.helpers import uuid
from ghoshell_common.contracts import LoggerItf
from .errors import CommandErrorCode
from .channel import ChannelCtx, Channel, ChannelMeta, TaskDoneCallback, RefreshMetaCallback, ChannelFullPath
import logging

__all__ = ['AbsChannelBroker']


class AbsChannelBroker(ChannelBroker, ABC):
    """
    实现基础的 Channel Broker, 用来给所有的 Broker 提供基准的生命周期.
    """

    def __init__(
            self,
            *,
            channel: "Channel",
            container: IoCContainer | None = None,
            logger: LoggerItf | None = None
    ):
        self._channel = channel
        self._name = channel.name()
        self._uid = channel.id()
        # 用不同的容器隔离依赖. 经过 prepare container 才进行封装.
        self._container: IoCContainer = Container(
            name=f'MossChannelBroker/{self._name}/{self._uid}',
            parent=container,
        )

        self._starting = False
        self._started = False
        self._running_task: Optional[asyncio.Task] = None
        # 用线程安全的事件. 考虑到 broker 未来可能会跨线程被使用.
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()
        self._children_brokers: dict[str, ChannelBroker] = {}

        self._loop: asyncio.AbstractEventLoop | None = None
        self._state_store: StateStore | None = None
        self._logger: LoggerItf | None = logger

        self._cached_meta: ChannelMeta = ChannelMeta.new_empty(self._uid, self.channel)
        # blocking lifecycle task 用来保证无论哪一层, 都不能有同时两个以上的生命周期任务在执行.
        self._lifecycle_task: asyncio.Task | None = None
        # 生命周期函数需要加锁.
        self._blocking_action_lock = asyncio.Lock()
        # 运行执行的并行任务.
        self._none_blocking_cmd_tasks: set[CommandTask] = set()
        self._executing_block_cm_task: CommandTask | None = None
        # 可以注册监听, 监听 refresh meta 动作.
        self._on_refresh_meta_callbacks: list[Callable[[ChannelMeta], Coroutine[None, None, None]]] = []

        self._task_done_callbacks: list[TaskDoneCallback] = []

        # log_prefix
        self.log_prefix = "[Channel %s %s][%s]" % (self._name, self._uid, self.__class__.__name__)

    @property
    def channel(self) -> "Channel":
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
            self._logger = self.container.force_fetch(LoggerItf)
        return self._logger

    @property
    def container(self) -> IoCContainer:
        """
        broker 所持有的 ioc 容器.
        """
        return self._container

    def prepare_container(self, container: IoCContainer) -> IoCContainer:
        # 重写这个函数完成自定义.
        if not container.bound(LoggerItf):
            container.set(LoggerItf, logging.getLogger("moss"))
        if not container.bound(StateStore):
            container.set(StateStore, BaseStateStore(owner=self._uid))
        return container

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

    def meta(self) -> ChannelMeta:
        """
        返回 Channel 自身的 Meta.
        """
        if not self.is_connected():
            return ChannelMeta.new_empty(self._uid, self.channel)
        return self._cached_meta

    def on_refresh_meta(self, callback: RefreshMetaCallback) -> None:
        self._on_refresh_meta_callbacks.append(callback)

    async def refresh_meta(
            self,
            callback: bool = True,
    ) -> None:
        """
        更新当前的 Channel Meta 信息. 用于支持被动拉取. 不会主动推送更新.
        """
        ctx = contextvars.copy_context()
        # 生成时添加 ctx.
        ChannelCtx.init(self)
        try:
            if not self._starting or self._closing_event.is_set():
                meta = ChannelMeta.new_empty(channel=self.channel, id=self._uid)
            else:
                meta = await ctx.run(self.generate_meta)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            self.logger.exception("%s refresh self meta failed %s", self.log_prefix, exc)
            # 出现异常后, 刷新一个异常的 meta.
            meta = ChannelMeta.new_empty(channel=self.channel, id=self._uid)

        self._cached_meta = meta
        self.logger.info(
            "%s refreshed meta", self.log_prefix,
        )
        # 创建异步的回调.
        if callback and self._on_refresh_meta_callbacks:
            for callback in self._on_refresh_meta_callbacks:
                if inspect.iscoroutinefunction(callback):
                    _ = asyncio.create_task(callback(meta))
                else:
                    _ = asyncio.create_task(asyncio.to_thread(callback, meta))

    @abstractmethod
    async def generate_meta(self) -> ChannelMeta:
        """
        重新生成 meta 数据对象.
        """
        pass

    def is_running(self) -> bool:
        """
        是否已经启动了. 如果 Broker 被 close, is_running 为 false.
        """
        return self._started and not self._closing_event.is_set()

    def is_available(self) -> bool:
        """
        当前 Channel 对于使用者而言, 是否可用.
        当一个 Broker 是 running & connected 状态下, 仍然可能会因为种种原因临时被禁用.
        """
        return self.is_running() and self.is_connected() and self.meta().available

    def on_task_done(self, callback: TaskDoneCallback) -> None:
        # 注册 task 回调.
        self._task_done_callbacks.append(callback)

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

    async def idle(self) -> None:
        """
        进入闲时状态.
        闲时状态指当前 Broker 及其 子 Channel 都没有 CommandTask 在运行的时候.
        """
        if not self.is_running():
            return
        try:
            await asyncio.sleep(0.0)
            await self._blocking_action_lock.acquire()
            await self._clear_lifecycle_task()
            ctx = contextvars.copy_context()
            ChannelCtx.init(self)
            on_idle_cor = ctx.run(self.on_idle)
            # idle 是一个在生命周期中单独执行的函数.
            task = asyncio.create_task(on_idle_cor)
            self._lifecycle_task = task
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
        # 先将 task 关闭掉.
        if self._executing_block_cm_task is not None and not self._executing_block_cm_task.done():
            self._executing_block_cm_task.fail(CommandErrorCode.CLEARED.error(f"cleared by broker"))
        self._executing_block_cm_task = None
        # 终止阻塞中的任务.
        if self._lifecycle_task and not self._lifecycle_task.done():
            self._lifecycle_task.cancel()
            try:
                await self._lifecycle_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.exception("%s clear lifecycle task failed: %s", self.log_prefix, e)
        self._lifecycle_task = None

    async def clear_all(self) -> None:
        if not self.is_running():
            return
        clear_tasks = [self._loop.create_task(self.clear())]
        for broker in self._children_brokers.values():
            if broker.is_running():
                clear_tasks.append(broker.clear_all())
        await asyncio.gather(*clear_tasks)

    async def clear(self) -> None:
        """
        当轨道命令被触发清空时候执行.
        """
        if not self._started or self._closed_event.is_set():
            return
        try:
            await asyncio.sleep(0.0)
            await self._blocking_action_lock.acquire()
            await self._clear_lifecycle_task()
            if len(self._none_blocking_cmd_tasks) > 0:
                for t in self._none_blocking_cmd_tasks:
                    if not t.done():
                        t.fail(CommandErrorCode.CLEARED.error(f"cleared by broker"))
            self._none_blocking_cmd_tasks.clear()
            # 阻塞等待到清空结束.
            # 同步阻塞等待 clear 执行完毕.
            ctx = contextvars.copy_context()
            ChannelCtx.init(self)
            cor = ctx.run(self.on_clear)
            await cor
        finally:
            self._blocking_action_lock.release()
            self.logger.info("%s cleared", self.log_prefix)

    async def pause(self) -> None:
        """
        设置当前 Broker 为 pause 状态.
        pause 状态下 Channel Broker 应该要进入某种安全姿态.
        """
        if not self._started or self._closed_event.is_set():
            return
        # 先清空所有的运动.
        await self.clear_all()
        try:
            await asyncio.sleep(0.0)
            await self._blocking_action_lock.acquire()
            await self._clear_lifecycle_task()
            ctx = contextvars.copy_context()
            ChannelCtx.init(self)
            pause_cor = ctx.run(self.on_pause)
            self._lifecycle_task = asyncio.create_task(pause_cor)
        finally:
            self.logger.info("%s is pausing", self.log_prefix)
            self._blocking_action_lock.release()

    @abstractmethod
    async def on_pause(self) -> None:
        pass

    @abstractmethod
    async def on_clear(self) -> None:
        """
        当轨道命令被触发清空时候执行.
        """
        pass

    async def start(self) -> None:
        """
        启动 Channel Broker.
        通常用 with statement 或 async exit stack 去启动.
        只会启动当前 channel 自身.
        """
        if self._starting:
            return
        self._starting = True
        self._loop = asyncio.get_running_loop()
        container = self.container
        self.prepare_container(container)
        # bootstrap container
        await asyncio.to_thread(container.bootstrap)
        # 启动 states 和 topics 模块.
        await self.states.start()
        ctx = contextvars.copy_context()
        ChannelCtx.init(self)
        cor = ctx.run(self.on_start_up)
        self.logger.info(
            "%s started", self.log_prefix,
        )
        await cor
        self._running_task = asyncio.create_task(ctx.run(self._keep_running_task))
        self._started = True
        # 刷新 meta.
        await self.refresh_meta()

    async def _keep_running_task(self) -> None:
        try:
            await self.on_running()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception("%s keep_running_task failed: %s", self.log_prefix, e)
        finally:
            self.logger.info("%s keep_running_task finished", self.log_prefix)

    @abstractmethod
    async def on_start_up(self) -> None:
        pass

    async def wait_closing(self) -> None:
        await self._closing_event.wait()

    async def wait_closed(self) -> None:
        await self._closed_event.wait()

    def close_sync(self) -> None:
        if not self.is_running():
            return
        # 运行关闭逻辑.
        self._loop.create_task(self.close())

    async def close(self) -> None:
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
            await self.clear_all()
            if self._running_task and not self._running_task.done():
                self._running_task.cancel()
                try:
                    await self._running_task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    self.logger.exception("%s close running task failed %s", self.log_prefix, e)

            self._running_task = None
            ctx = contextvars.copy_context()
            ChannelCtx.init(self)
            on_close_cor = ctx.run(self.on_close)
            try:
                # 等待运行全部结束.
                await on_close_cor
            except Exception as e:
                self.logger.exception("%s close self failed: %s", self.log_prefix, e)

            # 关闭 state store. 每个 Broker 都得有自己的 state store.
            if self._state_store:
                await self._state_store.close()
            self._state_store = None

            # 关闭容器运行.
            self.logger.info(
                "%s prepare to shutdown", self.log_prefix,
            )
            await asyncio.to_thread(self.container.shutdown)
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
        self._lifecycle_task = None
        self._none_blocking_cmd_tasks.clear()
        self._on_refresh_meta_callbacks.clear()

    @abstractmethod
    async def on_close(self) -> None:
        pass

    @abstractmethod
    async def on_running(self) -> None:
        pass

    async def execute_task_soon(self, task: CommandTask) -> None:
        """
        在 Broker 中执行一个 command task. 会尽快返回, 由 Task 自身完成阻塞.
        """
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

        try:
            await asyncio.sleep(0)
            await self._blocking_action_lock.acquire()
            # 如果是阻塞类型的任务, 必须清空主要执行中的任务.
            task.set_state(CommandTaskStateType.executing)
            task.add_done_callback(self._task_done_callback)
            if task.meta.blocking:
                # 清除其它 lifecycle 任务.
                await self._clear_lifecycle_task()
                # 通过一个 task 确保 task 一定会被执行完.
                cor = self._ensure_task_done(task)
                ensure_task_done = asyncio.create_task(cor)
                self._lifecycle_task = ensure_task_done
                self._executing_block_cm_task = task
            else:
                cor = self._ensure_task_done(task)
                _ = asyncio.create_task(cor)
                self._none_blocking_cmd_tasks.add(task)
        finally:
            self._blocking_action_lock.release()
            self.logger.info("%s executing task %s", self.log_prefix, task.cid)

    async def _ensure_task_done(self, task: CommandTask) -> None:
        if task.done():
            return

        # 准备执行.
        task.exec_chan = self.name
        try:
            await asyncio.sleep(0)
            # 在这里让出控制权, 保证 finally 一定被执行.
            self.logger.info("%s start task %s", self.log_prefix, task.cid)
            # 初始化函数运行上下文.
            ctx = contextvars.copy_context()
            ChannelCtx.init(self, task)
            # 使用 dry run 来管理生命周期.
            run_cor = ctx.run(task.dry_run)
            execution_task = asyncio.create_task(run_cor)
            task_done_outside = asyncio.create_task(task.wait(throw=False))
            done, pending = await asyncio.wait([execution_task, task_done_outside], return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            # 为结果赋值.
            if not task.done():
                result = await execution_task
                task.resolve(result)
            self.logger.info("%s resolved task %s", self.log_prefix, task.cid)

        except Exception as e:
            self.logger.error("%s task %s failed: %s", self.log_prefix, task.cid, e)
            if not task.done():
                task.fail(e)
            raise
        finally:
            if not task.done():
                task.fail(CommandErrorCode.UNKNOWN_ERROR.error(f"task not done after execution"))
            self.logger.info(
                "%s done task %s at state", self.log_prefix, task.cid, task.state,
            )

