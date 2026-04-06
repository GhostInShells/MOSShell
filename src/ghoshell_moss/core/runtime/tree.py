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
from ghoshell_moss.core.concepts.command import Command, CommandUniqueName
from ghoshell_common.contracts import LoggerItf
import logging
import time
import contextlib
import asyncio

__all__ = ["BaseChannelTree"]

_ChannelId = str
_ChannelName = str

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

        self.sustain_children: set[_ChannelId] = set()
        self.virtual_children: set[_ChannelId] = set()
        self.children_names: dict[_ChannelId, _ChannelName] = dict()
        self.refresh_time: int = 0
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
                wait_self = runtime.refresh_own_metas()
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
        new_children_names: dict[_ChannelId, _ChannelName] = dict()
        # 首先刷新树形结构. 发现失联节点删除, 发现新节点添加.
        for name, child in sub_channels.items():
            _channel_id = child.id()
            if self.refresh_time == 1 or _channel_id in self.sustain_children:
                existing_sub_channels.add(_channel_id)
                # 管理 names.
                new_children_names[_channel_id] = name
            # 已经完成过初始化.
            if self.refresh_time == 1:
                # 没有第一次创建过. 才允许创建父节点.
                if ctx.exists(_channel_id):
                    # 被别人先抢为儿子孙子了.
                    continue
                # 添加到自己的孩子中.
                self.sustain_children.add(_channel_id)
                # 添加新节点. 不过应该只会在第一次运行.
                fullpath = Channel.join_channel_path(self.path, name)
                # 先注册要创建的节点.
                creating_children_channels[fullpath] = child

        # 开始准备动态节点.
        new_virtual_children = set()
        for name, child in runtime.virtual_sub_channels().items():
            # 不允许同名子节点.
            if name in new_children_names:
                continue
            _channel_id = child.id()
            if _channel_id in self.virtual_children:
                # 是已经注册过的.
                new_virtual_children.add(_channel_id)
                existing_sub_channels.add(_channel_id)
                new_children_names[_channel_id] = name
                continue
            # 尝试创建这个节点.
            if ctx.exists(_channel_id):
                # 已经被别人占了. 这一轮没有机会创建.
                continue
            new_virtual_children.add(_channel_id)
            fullpath = Channel.join_channel_path(self.path, name)
            creating_children_channels[fullpath] = child
            new_children_names[_channel_id] = name

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
        self.children_names = new_children_names
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
        self._channel_id_to_paths: dict[_ChannelId, ChannelFullPath] = {}

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
        self._channel_id_to_paths[channel_id] = path

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
        if id in self._channel_id_to_paths:
            path = self._channel_id_to_paths.pop(id)
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
                for _id in node.sustain_children:
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
        path = self._channel_id_to_paths.get(id, None)
        node = self._runtime_status_nodes.get(path, None)
        runtime = self._runtimes.get(id, None)
        if node is None or runtime is None:
            return asyncio.create_task(_noop())
        if not runtime.is_connected():
            # 只有连接后才会刷新.
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

    def get_channel_runtime(self, channel: Channel, running: bool = False) -> ChannelRuntime | None:
        if self._closed:
            return None
        if not self.is_running():
            return None
        if channel is self._main.channel:
            return self._main
        channel_id = channel.id()
        result = self._runtimes.get(channel_id)
        if result is None:
            return None
        if running and not result.is_running():
            return None
        return result

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
                self._channel_id_to_paths[node.id] = node.path

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

    def get_running_runtime(self, channel_id: str) -> ChannelRuntime | None:
        if channel_id not in self._runtimes:
            return None
        runtime = self._runtimes[channel_id]
        if not runtime.is_running():
            return None
        return runtime

    def all(self, root: ChannelFullPath = "") -> dict[ChannelFullPath, ChannelRuntime]:
        root_node = self._runtime_status_nodes.get(root)
        if root_node is None:
            return {}

        def _recursive_find_runtime(
                _result: dict[ChannelFullPath, ChannelRuntime],
                _node: ChannelRuntimeNode,
                _relative_path: str,
        ):
            _runtime = self._runtimes.get(_node.id)
            if _runtime is None:
                return
            _result[_relative_path] = _runtime
            for _child_id, _child_name in _node.children_names.items():
                runtime = self.get_running_runtime(_child_id)
                child_relative_path = Channel.join_channel_path(_relative_path, _child_name)
                if runtime is None:
                    continue
                _child_full_path = self._channel_id_to_paths.get(_child_id)
                if _child_full_path:
                    _child_node = self._runtime_status_nodes.get(_child_full_path)
                    if _child_node is None:
                        continue
                    # 深度优先递归.
                    _recursive_find_runtime(_result, _child_node, child_relative_path)

        result = {}
        _recursive_find_runtime(_result=result, _node=root_node, _relative_path='')
        return result

    def metas(self, channel: Channel | None = None) -> dict[ChannelFullPath, ChannelMeta]:
        channel = channel or self._main.channel
        channel_id = channel.id()
        root_path = self._channel_id_to_paths.get(channel_id, None)
        if root_path is None:
            return {}
        return self._metas(root_path)

    def _metas(self, path: ChannelFullPath = '') -> dict[ChannelFullPath, ChannelMeta]:
        node = self._runtime_status_nodes.get(path)
        if node is None:
            return {}
        runtime = self._runtimes.get(node.id)
        if runtime is None:
            return {}
        if not runtime.is_running():
            return {}
        if not runtime.is_connected():
            return {'': ChannelMeta.new_empty(runtime.channel.id(), runtime.channel, "not connected")}
        if not runtime.is_available():
            return {'': ChannelMeta.new_empty(runtime.channel.id(), runtime.channel, "not available")}
        metas = runtime.own_metas().copy()
        if '' not in metas:
            return {}
        # 递归获取子节点所有的 meta.
        self_meta = metas['']
        # 赋值子节点名字. 这个参数是实质动态创建的.
        child_names = list(node.children_names.values())
        self_meta.children = child_names
        for child_id, child_name in node.children_names.items():
            virtual = child_id in node.virtual_children
            sub_full_path = Channel.join_channel_path(path, child_name)
            # 递归获取 metas.
            sub_metas = self._metas(sub_full_path)
            for sub_relative_path, meta in sub_metas.items():
                relative_sub_path = Channel.join_channel_path(child_name, sub_relative_path)
                if virtual:
                    meta = meta.model_copy(update={'virtual': True})
                metas[relative_sub_path] = meta
        return metas

    def get_channel_path(self, channel_id: str) -> ChannelFullPath | None:
        if channel_id in self._channel_id_to_paths:
            return self._channel_id_to_paths[channel_id]
        return None

    def get_runtime_by_path(self, path: ChannelFullPath | str, root: Channel | None = None) -> ChannelRuntime | None:
        root_path = ''
        if root is not None:
            root_id = root.id()
            root_path = self._channel_id_to_paths.get(root_id)
            if root_path is None:
                return None
        search_path = Channel.join_channel_path(root_path, path)
        if search_path not in self._runtime_status_nodes:
            return None
        node = self._runtime_status_nodes[search_path]
        return self.get_running_runtime(node.id)

    def get_channel_node(self, channel: Channel) -> ChannelRuntimeNode | None:
        channel_id = channel.id()
        if channel_id not in self._runtimes:
            return None
        runtime = self._runtimes[channel_id]
        if not runtime.is_running():
            return None
        path = self._channel_id_to_paths.get(channel_id)
        if not path:
            self.logger.error("%s get runtime path by %s error: not found", self.log_prefix, channel_id)
        node = self._runtime_status_nodes.get(path)
        return node

    def get_children_runtimes(self, channel: Channel) -> dict[str, "ChannelRuntime"]:
        channel_id = channel.id()
        if channel_id not in self._runtimes:
            return {}
        runtime = self._runtimes[channel_id]
        if not runtime.is_running():
            return {}
        if not runtime.is_available():
            return {}
        path = self._channel_id_to_paths.get(channel_id, None)
        if path is None:
            self.logger.error("%s get runtime path by %s error: not found", self.log_prefix, channel_id)
            return {}
        node = self._runtime_status_nodes.get(path)
        if not node:
            self.logger.error(
                "%s get runtime node by path=%s, id=%s error: not found",
                self.log_prefix, path, channel_id,
            )
            return {}
        children = {}
        for _channel_id, name in node.children_names.items():
            runtime = self.get_running_runtime(_channel_id)
            if runtime:
                children[name] = runtime
        return children

    def get_child_runtime(self, channel: Channel, child_name: str) -> ChannelRuntime | None:
        node = self.get_channel_node(channel)
        if node is None:
            return None
        full_path = Channel.join_channel_path(node.path, child_name)
        child_node = self._runtime_status_nodes.get(full_path)
        if not child_node:
            return None
        return self._runtimes.get(child_node.id)

    def get_command(self, channel: Channel, name: CommandUniqueName) -> Command | None:
        """
        递归查找一个 command 是否存在.
        """
        runtime = self.get_channel_runtime(channel, running=True)
        if runtime is None:
            return None
        return self._get_command(runtime, name)

    def _get_command(self, runtime: ChannelRuntime, unique_name: CommandUniqueName) -> Command | None:
        if runtime is None or not runtime.is_running() or not runtime.is_available():
            # 不用调用了, 直接判断.
            return None
        # 判断是不是被当前 runtime 所 own 的.
        if runtime.has_own_command(unique_name):
            # 直接返回 runtime 所持有的.
            return runtime.get_own_command(unique_name)
        relative_path, name = Command.split_unique_name(unique_name)
        if not relative_path:
            # 如果没有 relative path, 则不用继续找下去了.
            return None
        # has relative path.
        paths = Channel.split_channel_path_to_names(relative_path, 1)
        child_name = paths[0]
        # 先找到当前的节点路径.
        current_node = self.get_channel_node(runtime.channel)
        if current_node is None:
            return None
        # 找到预期中小孩的路径.
        child_path = Channel.join_channel_path(current_node.path, child_name)
        # 小孩必须存在, 可能并没有资格挂载.
        child_node = self._runtime_status_nodes.get(child_path)
        if not child_node:
            return None
        # 验证小孩的 runtime 存在.
        child_runtime = self._runtimes.get(child_node.id)
        if not child_runtime:
            return None
        further_path = "".join(paths[1:])
        return self._get_command(child_runtime, Command.make_unique_name(further_path, name))

    def commands(self, channel: Channel, available_only: bool = True) -> dict[ChannelFullPath, dict[str, Command]]:
        """
        递归获取一个 channel 所有的子命令, 按路径完成分组.
        """
        runtime = self.get_channel_runtime(channel, running=True)
        if runtime is None:
            return {}
        result = {}
        commands = runtime.own_commands(available_only)
        for unique_name, command in commands.items():
            path, name = Command.split_unique_name(unique_name)
            if path not in result:
                result[path] = {}
            if name not in result[path]:
                result[path][name] = command

        children = self.get_children_runtimes(channel)
        if len(children) > 0:
            for child_name, runtime in children.items():
                sub_commands = runtime.commands(available_only=True)
                for sub_path, command_group in sub_commands.items():
                    full_path = Channel.join_channel_path(child_name, sub_path)
                    if full_path not in result:
                        result[full_path] = {}
                    for command_name, command in command_group.items():
                        if command_name not in result[full_path]:
                            result[full_path][command_name] = command
        return result

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
        from ghoshell_moss.topic import QueueBasedTopicService
        return QueueBasedTopicService(sender=self.main.id)
