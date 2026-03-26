import contextlib

import asyncio
from typing import Optional
from ghoshell_container import IoCContainer, Container

from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.concepts.channel import (
    Channel,
    ChannelRuntime,
    ChannelImportLib,
)
from ghoshell_common.contracts import LoggerItf
import logging

__all__ = ["BaseImportLib"]

_ChannelId = str


class BaseImportLib(ChannelImportLib):
    """
    唯一的 lib 用来管理所有可以被 import 的 channel runtime
    """

    def __init__(self, main: ChannelRuntime, container: IoCContainer | None = None):
        self._main = main
        self._name = "MossChannelImportLib/{}/{}".format(main.name, main.id)
        self._container = Container(
            name=self._name,
            parent=container,
        )
        # 绑定自身到容器中. 凡是用这个容器启动的 runtime, 都可以拿到 ChannelImportLib 并获取子 channel runtime.
        self._container.set(BaseImportLib, self)
        self._logger: Optional[LoggerItf] = None
        self._runtimes: dict[_ChannelId, ChannelRuntime] = {}
        self._runtimes_lock: asyncio.Lock = asyncio.Lock()
        self._topics: TopicService | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_exit_stack = contextlib.AsyncExitStack()
        self._start: bool = False
        self._close: bool = False

    def get_channel_runtime(self, channel: Channel) -> ChannelRuntime | None:
        if channel is self._main.channel:
            # 根节点不启动.
            return self._main

        if not self.is_running():
            return None

        channel_id = channel.id()
        return self._runtimes.get(channel_id)

    async def get_or_create_channel_runtime(self, channel: Channel) -> ChannelRuntime | None:
        if runtime := self.get_channel_runtime(channel):
            await runtime.wait_started()
            if runtime.is_running():
                return runtime
            else:
                return None
        # 第一次创建.
        runtime = await self.compile_channel(channel)
        if runtime is None:
            return None
        await runtime.wait_started()
        return runtime

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
            self._logger = self._container.get(LoggerItf)
            if self._logger is None:
                logger = logging.getLogger("moss")
                self._logger = logger
                self._container.set(LoggerItf, logger)
        return self._logger

    def is_running(self) -> bool:
        return self._start and not self._close

    @contextlib.asynccontextmanager
    async def _container_ctx_manager(self):
        await asyncio.to_thread(self._container.bootstrap)
        yield
        await asyncio.to_thread(self._container.shutdown)

    @contextlib.asynccontextmanager
    async def _topics_ctx_manager(self):
        self._topics = self._container.get(TopicService)
        if not self._topics:
            self._topics = self._create_default_topics()
            self._container.set(TopicService, self._topics)
        topic_started = False
        if not self._topics.is_running():
            await self._topics.start()
            topic_started = True
        yield
        if topic_started:
            await self._topics.close()

    async def start(self) -> None:
        if self._start:
            return
        self._start = True
        self._loop = asyncio.get_event_loop()
        await self._async_exit_stack.__aenter__()
        await self._async_exit_stack.enter_async_context(self._container_ctx_manager())
        await self._async_exit_stack.enter_async_context(self._topics_ctx_manager())

    def _create_default_topics(self) -> TopicService:
        from ghoshell_moss.core.topic import QueueBasedTopicService

        return QueueBasedTopicService(sender=self.main.id)

    async def close(self) -> None:
        if self._close:
            return
        self._close = True
        # todo: 实现更可靠的生命周期.
        await self._runtimes_lock.acquire()
        try:
            clear_runtimes = []
            clear_runtime_tasks = []
            closing_runtime_ids = set()
            for runtime in self._runtimes.values():
                if runtime.is_running():
                    if runtime.id in closing_runtime_ids:
                        continue
                    closing_runtime_ids.add(runtime.id)
                    clear_task = self._loop.create_task(runtime.close())
                    clear_runtimes.append(runtime)
                    clear_runtime_tasks.append(clear_task)
            done = await asyncio.gather(*clear_runtime_tasks, return_exceptions=True)
            idx = 0
            self._runtimes.clear()
            for t in done:
                if isinstance(t, Exception):
                    runtime = clear_runtimes[idx]
                    self.logger.exception(
                        "%s close runtime %s, id=%s failed: %s", self._name, runtime.name, runtime.id, t
                    )
                idx += 1
        finally:
            self._runtimes_lock.release()
            await self._async_exit_stack.__aexit__(None, None, None)
            if self._loop:
                self._loop.run_in_executor(None, self._container.shutdown)
