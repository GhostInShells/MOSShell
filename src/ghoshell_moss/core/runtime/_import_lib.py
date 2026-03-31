from abc import abstractmethod
from typing import Optional, Callable, Protocol
from ghoshell_container import IoCContainer, Container

from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.concepts.channel import (
    Channel,
    ChannelRuntime,
    ChannelTree,
    ChannelFullPath,
    ChannelMeta,
)
from ghoshell_common.contracts import LoggerItf
import logging
import time
import contextlib
import asyncio

__all__ = ["BaseChannelTree"]

_ChannelId = str

_AddRuntime = Callable[[ChannelRuntime], asyncio.Task]
_RemoveRuntime = Callable[[ChannelRuntime], asyncio.Task]


async def _noop():
    pass


class ChannelTreeContext(Protocol):

    @abstractmethod
    def exists(self, id: _ChannelId) -> bool:
        pass

    @abstractmethod
    def add(self, path: ChannelFullPath, channel: Channel) -> asyncio.Future | None:
        pass

    @abstractmethod
    def remove(self, id: _ChannelId) -> asyncio.Future | None:
        pass

    @abstractmethod
    def refresh(self, id: _ChannelId, wait: bool = False) -> asyncio.Future:
        pass

    @abstractmethod
    def get(self, id: _ChannelId) -> ChannelRuntime | None:
        pass


class ChannelRuntimeNode:

    def __init__(
            self,
            id: _ChannelId,
            path: str,
            loop: asyncio.AbstractEventLoop,
            logger: LoggerItf,
            refresh_interval: float = 0.0,
    ):
        self.id = id
        self.path = path
        self.logger = logger
        self.refreshed_at: float = 0.0
        self.refreshing_lock = asyncio.Lock()
        self.loop = loop
        self.refreshing_task: Optional[asyncio.Task] = None
        self.refresh_interval: float = refresh_interval
        self.failure: str = ''

        self.children: set[_ChannelId] = set()
        self.virtual_children: set[_ChannelId] = set()
        self.refresh_time: int = 0
        self.children_names: set[str] = set()
        self.logger_prefix = "<ChannelRuntimeNode path=%s id=%s>" % (path, id)

    def __repr__(self):
        return self.logger_prefix

    def is_refreshing(self) -> bool:
        return self.refreshing_task is not None and not self.refreshing_task.done()

    def refresh(
            self,
            runtime: ChannelRuntime,
            ctx: ChannelTreeContext,
            wait: bool,
    ) -> asyncio.Future:
        """
        更新一个节点, 但一个时间点只会更新一次.
        通过 asyncio task 返回最近的一轮更新状态.
        如果一直更新不成功, 可以废弃节点运行状态.
        """
        if not runtime.is_running():
            # 容错. 应该不会被调用到.
            self.logger.error("%r refresh after running done", self)
            return asyncio.create_task(_noop())
        if self.refreshing_task is not None and not self.refreshing_task.done():
            # 返回未完成的 task.
            return self.refreshing_task
        # 创建新的 task.
        self.refreshing_task = asyncio.create_task(self._refresh(runtime, ctx, wait))
        return asyncio.shield(self.refreshing_task)

    def get_own_metas(self, runtime: ChannelRuntime) -> tuple[dict[ChannelFullPath, ChannelMeta], bool]:
        """
        获取一个节点的
        """
        if not runtime.is_running():
            metas = {'': ChannelMeta.new_empty(self.id, runtime.channel, "not running")}
            return metas, False
        if not runtime.is_connected():
            metas = {'': ChannelMeta.new_empty(self.id, runtime.channel, "not connected")}
            return metas, False
        if not runtime.is_available():
            metas = {'': ChannelMeta.new_empty(self.id, runtime.channel, "not available")}
            return metas, False
        if self.failure:
            metas = {'': ChannelMeta.new_empty(self.id, runtime.channel, self.failure)}
            return metas, False
        return runtime.own_metas().copy(), True

    async def _refresh(
            self,
            runtime: ChannelRuntime,
            # 用 ctx 解决互相持有的递归困境.
            ctx: ChannelTreeContext,
            recursive_wait: bool,
    ) -> None:
        now = time.time()
        async with self.refreshing_lock:
            # 检查不合法.
            if now < self.refreshed_at + self.refresh_interval:
                return
            if not runtime.is_running() or not runtime.is_connected() or not runtime.is_available():
                return
            try:
                self.refresh_time += 1
                # 先更新结构.
                existing_sub_channels = await self._refresh_structure(runtime, ctx, recursive_wait)
                self.logger.info("%r refreshed structure", self)
                await asyncio.sleep(0.0)
                # 再更新 meta.
                waiting_tasks = []
                for channel_id in existing_sub_channels:
                    task = ctx.refresh(channel_id, wait=recursive_wait)
                    if task and recursive_wait:
                        waiting_tasks.append(task)
                wait_self = asyncio.create_task(runtime.refresh_own_metas(force=True))
                # 先阻塞等待自己.

                await wait_self
                if recursive_wait and len(waiting_tasks) > 0:
                    # 然后等待子孙.
                    _ = await asyncio.gather(*waiting_tasks, return_exceptions=True)
                self.logger.info("%r refreshed self and sub channels", self)
                # 更新最后刷新时间.
                self.failure = ''
            except asyncio.CancelledError:
                self.logger.info("%r refreshed cancelled", self)
                raise
            except Exception as e:
                self.logger.error("%r refreshed exception: %s", self, e)

                # 更新失败, 不允许使用.
                self.failure = "refresh failed: %s" % e
            finally:
                self.refreshed_at = time.time()
                self.logger.info("%r refreshed done", self)

    async def _refresh_structure(
            self,
            runtime: ChannelRuntime,
            ctx: ChannelTreeContext,
            recursive_wait: bool,
    ) -> set[_ChannelId]:
        """
        更新 channel 的树形结构, 同时返回需要被刷新的 channel id.
        需要新建的 channel, 本身在新建完后就会执行刷新.
        """
        # 准备创建的节点.
        creating_children_channels: dict[ChannelFullPath, Channel] = {}
        sub_channels = runtime.sub_channels()
        existing_sub_channels: set[_ChannelId] = set()
        # 首先刷新树形结构. 发现失联节点删除, 发现新节点添加.
        for name, child in sub_channels.items():
            _channel_id = child.id()
            if self.refresh_time == 1 or _channel_id in self.children:
                existing_sub_channels.add(_channel_id)
            # 已经完成过初始化.
            if self.refresh_time == 1:
                # 没有第一次创建过. 才允许创建父节点.
                if ctx.exists(_channel_id):
                    # 被别人先抢为儿子孙子了.
                    continue
                # 添加到自己的孩子中.
                self.children.add(_channel_id)
                # 添加新节点. 不过应该只会在第一次运行.
                fullpath = Channel.join_channel_path(self.path, name)
                # 先注册要创建的节点.
                creating_children_channels[fullpath] = child

        # 开始准备动态节点.
        new_virtual_children = set()
        for name, child in runtime.virtual_sub_channels().items():
            _channel_id = child.id()
            if _channel_id in self.virtual_children:
                # 是已经注册过的.
                new_virtual_children.add(_channel_id)
                existing_sub_channels.add(_channel_id)
                continue
            # 尝试创建这个节点.
            if ctx.exists(_channel_id):
                # 已经被别人占了. 这一轮没有机会创建.
                continue
            new_virtual_children.add(_channel_id)
            fullpath = Channel.join_channel_path(self.path, name)
            creating_children_channels[fullpath] = child

        removing_children: list[_ChannelId] = []
        for _channel_id in self.virtual_children:
            # 不在新的 virtual children 列表里, 则意味着要移除.
            if _channel_id not in new_virtual_children:
                removing_children.append(_channel_id)

        # 先移除, 然后再创建.
        if len(removing_children) > 0:
            self.logger.info("%r removing unlink channel: %d", self, len(removing_children))
            removing_tasks = []
            for _channel_id in removing_children:
                task = ctx.remove(_channel_id)
                if task:
                    removing_tasks.append(task)
            if len(removing_tasks) > 0:
                # 阻塞等待该移除的节点正确移除. 否则不能启动新的节点.
                _ = await asyncio.gather(*removing_tasks, return_exceptions=True)

        # 开始创建所有的新节点.
        if len(creating_children_channels) > 0:
            self.logger.info("%r create new children channel: %d", self, len(creating_children_channels))
            creating_tasks = []
            for path, child in creating_children_channels.items():
                task = ctx.add(path, child)
                if task:
                    creating_tasks.append(task)

            if recursive_wait:
                # 如果必须要等待, 则等待所有的节点正确创建.
                _ = await asyncio.gather(*creating_tasks, return_exceptions=True)

        # 赋值, 更新新的动态节点.
        self.virtual_children = new_virtual_children
        return existing_sub_channels

    async def clear(self):
        if self.is_refreshing():
            self.refreshing_task.cancel()
            try:
                await self.refreshing_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                self.logger.exception("%r clear exception: %s", self, e)
        del self.loop
        del self.logger


class BaseChannelTree(ChannelTree, ChannelTreeContext):
    """
    唯一的 lib 用来管理所有可以被 import 的 channel runtime
    """

    def __init__(self, main: ChannelRuntime, container: IoCContainer | None = None):
        self._main = main
        self._name = "MossChannelImportLib/{}/{}".format(main.name, main.id)
        self._id = main.channel.id()
        self._container = container or Container(name=self._name)
        # 绑定自身到容器中. 凡是用这个容器启动的 runtime, 都可以拿到 ChannelImportLib 并获取子 channel runtime.
        self._logger: Optional[LoggerItf] = None
        # 所有的 runtime.
        self._runtimes: dict[_ChannelId, ChannelRuntime] = {}
        # runtime 的刷新状态.
        self._runtime_status_nodes: dict[ChannelFullPath, ChannelRuntimeNode] = {}
        self._runtime_id_to_paths: dict[_ChannelId, ChannelFullPath] = {}

        self._runtimes_lock: asyncio.Lock = asyncio.Lock()

        self._topics: TopicService | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._main_loop_task: asyncio.Task | None = None
        self._start: bool = False
        self._started: asyncio.Event = asyncio.Event()
        self._closed: bool = False
        self._closing_event: asyncio.Event = asyncio.Event()
        self._task_group: set[asyncio.Task] = set()
        self._ctx_stack = contextlib.AsyncExitStack()
        self._error: Exception | None = None
        self.log_prefix = "<ChannelImportlib root=%s id=%s>" % (main.name, main.id)

    def __repr__(self):
        return self.log_prefix

    def exists(self, id: _ChannelId) -> bool:
        if not self.is_running():
            return False
        return id in self._runtimes

    def add(self, path: ChannelFullPath, channel: Channel) -> asyncio.Future | None:
        """
        添加一个新的节点到运行时.
        """
        if not self.is_running():
            return None
        channel_id = channel.id()
        if channel_id in self._runtimes:
            return None
        # 创建新的 runtime.
        runtime = channel.bootstrap(self._container)
        self._runtimes[channel_id] = runtime
        node = ChannelRuntimeNode(channel_id, path, self._loop, logger=self._logger)
        # 注册 node 节点.
        self._runtime_status_nodes[path] = node
        # 建立查找关系.
        self._runtime_id_to_paths[channel_id] = path

        async def _start_runtime():
            nonlocal node, runtime, channel_id
            try:
                # 启动节点.
                if not runtime.is_running():
                    await runtime.start()
            except Exception as e:
                # 启动失败会删除节点.
                self.logger.exception("%r start %s channel exception: %s", self, path, e)
                _task = self.remove(channel_id)
                if _task:
                    await _task
            #  首次启动时, 强制递归刷新.
            await self.refresh(channel_id, wait=True)

        # 创建异步任务.
        task = asyncio.create_task(_start_runtime())
        # 添加到任务池.
        self._add_task(task)
        return asyncio.shield(task)

    def remove(self, id: _ChannelId) -> asyncio.Future | None:
        """
        从运行时里删除一个 runtime id.
        """
        if not self.is_running():
            return None
        if id not in self._runtimes:
            # 没注册过, 就返回.
            return None
        runtime = self._runtimes.pop(id)
        node = None
        if id in self._runtime_id_to_paths:
            path = self._runtime_id_to_paths.pop(id)
            if path in self._runtime_status_nodes:
                node = self._runtime_status_nodes.pop(path)

        async def _stop_runtime():
            nonlocal node, runtime
            removing_chain = []
            if node:
                # 解除关联.
                await node.clear()
                # 确保子孙节点被递归清楚了.
                for _id in node.virtual_children:
                    sub_task = self.remove(_id)
                    if sub_task:
                        removing_chain.append(sub_task)
                for _id in node.children:
                    sub_task = self.remove(_id)
                    if sub_task:
                        removing_chain.append(sub_task)
            # 等待自身 runtime 运行完毕.
            await runtime.clear()

        task = asyncio.create_task(_stop_runtime())
        self._add_task(task)
        return asyncio.shield(task)

    def refresh(self, id: _ChannelId, wait: bool = False) -> asyncio.Future:
        if not self.is_running():
            return asyncio.create_task(_noop())
        path = self._runtime_id_to_paths.get(id, None)
        node = self._runtime_status_nodes.get(path, None)
        runtime = self._runtimes.get(id, None)
        if node is None or runtime is None:
            return asyncio.create_task(_noop())

        # 通过 Node 运行一个刷新任务.
        return node.refresh(runtime, self, wait=wait)

    def get(self, id: _ChannelId) -> ChannelRuntime | None:
        if not self.is_running():
            return None
        return self._runtimes.get(id, None)

    def _add_task(self, task: asyncio.Task) -> None:
        if not self.is_running() or task.done():
            return None
        task.add_done_callback(self._remove_task)
        self._task_group.add(task)
        return None

    def _remove_task(self, task: asyncio.Task) -> None:
        if task in self._task_group:
            self._task_group.remove(task)

    def get_channel_runtime(self, channel: Channel) -> ChannelRuntime | None:
        if self._closed:
            return None
        if channel is self._main.channel:
            # 根节点不启动.
            return self._main

        if not self.is_running():
            return None

        channel_id = channel.id()
        return self._runtimes.get(channel_id)

    async def compile_channel(self, channel: Channel) -> ChannelRuntime | None:
        # 只有创建这一段需要上锁.
        if not self.is_running():
            return None
        channel_id = channel.id()
        runtime = self._runtimes.get(channel_id)
        # 只要 runtime 存在就立刻返回.
        if runtime is not None:
            return runtime
        await self._runtimes_lock.acquire()
        try:
            # 用自身的容器启动 ChannelImportLib.
            runtime = channel.bootstrap(self._container)
            # 避免抢锁嵌套成环.
            self._runtimes[channel_id] = runtime
            _ = asyncio.create_task(runtime.start())
            return runtime
        except Exception as e:
            self.logger.exception(
                "%s failed to build channel %s, id=%s: %s", self._name, channel.name(), channel.id(), e
            )
            return None
        finally:
            self._runtimes_lock.release()

    @property
    def main(self) -> ChannelRuntime:
        return self._main

    @property
    def topics(self) -> TopicService:
        if not self.is_running():
            raise RuntimeError("Not running")
        return self._topics

    @property
    def logger(self):
        if self._logger is None:
            logger = logging.getLogger("moss")
            self._logger = logger
        return self._logger

    def is_running(self) -> bool:
        return self._start and not self._closed

    @contextlib.asynccontextmanager
    async def _container_ctx_manager(self):
        try:
            self._container.set(BaseChannelTree, self)
            self._container.set(ChannelTree, self)
            self._logger = self._container.get(LoggerItf)
            if self._logger is None:
                self._logger = logging.getLogger("moss")
                self._container.set(LoggerItf, self._logger)
            yield
        finally:
            self._container.unbound(BaseChannelTree)
            self._container.unbound(ChannelTree)
            self._container = None

    @contextlib.asynccontextmanager
    async def _topics_ctx_manager(self):
        topic_started = False
        try:
            self._topics = self._container.get(TopicService)
            if not self._topics:
                self._topics = self._create_default_topics()
                self._container.set(TopicService, self._topics)
            if not self._topics.is_running():
                await self._topics.start()
                topic_started = True
            yield
        finally:
            if topic_started:
                await self._topics.close()

    async def _main_loop(self):
        try:
            async with contextlib.AsyncExitStack() as stack:
                await stack.enter_async_context(self._container_ctx_manager())
                await stack.enter_async_context(self._topics_ctx_manager())
                # 阻塞刷新等待根节点递归启动.
                node = ChannelRuntimeNode(
                    id=self._id,
                    path='',
                    loop=self._loop,
                    logger=self.logger,
                )
                # 添加爱根节点.
                self._runtimes[node.id] = self._main
                self._runtime_status_nodes[node.path] = node
                self._runtime_id_to_paths[node.id] = node.path

                await self.refresh(self._main.channel.id(), wait=True)
                self._started.set()
                # 等待到关闭发生.
                await self._closing_event.wait()
                self._closed = True
                await self._clear_all_runtimes()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.exception("%r main loop exception: %s", self, e)
            self._error = e
        finally:
            self._closed = True
            self.logger.info("%r main loop stopped", self)

    async def _clear_all_runtimes(self) -> None:
        runtimes = self._runtimes.copy()
        self._runtimes.clear()
        nodes = self._runtime_status_nodes.copy()
        self._runtime_status_nodes.clear()
        stop_any_refreshing = []
        for node in nodes.values():
            stop_any_refreshing.append(node.clear())
        done = await asyncio.gather(*stop_any_refreshing, return_exceptions=True)
        for r in done:
            if isinstance(r, Exception):
                self.logger.error("%s stop all the runtime node error: %s", self.log_prefix, r)
        stop_the_world = []
        for runtime in runtimes.values():
            stop_the_world.append(runtime.close())
        done = await asyncio.gather(*stop_the_world, return_exceptions=True)
        for r in done:
            if isinstance(r, Exception):
                self.logger.error("%s clear all runtimes error: %s", self.log_prefix, r)
        self._main = None

    async def start(self) -> None:
        if self._start:
            await self._started.wait()
            return
        self._start = True
        self._loop = asyncio.get_event_loop()
        self._main_loop_task = self._loop.create_task(self._main_loop())
        await self._started.wait()
        if self._error:
            raise self._error

    async def close(self) -> None:
        if self._closed or self._closing_event.is_set():
            return
        self._closing_event.set()
        if self._main_loop_task is not None:
            await self._main_loop_task
            self._main_loop_task = None
        if self._error:
            raise self._error

    def _create_default_topics(self) -> TopicService:
        from ghoshell_moss.core.topic import QueueBasedTopicService
        return QueueBasedTopicService(sender=self.main.id)
