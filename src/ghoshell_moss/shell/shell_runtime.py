from typing import Dict, Optional
from ghoshell_moss.concepts.command import CommandTask, Command, CommandWrapper
from ghoshell_moss.concepts.channel import ChannelMeta, Channel
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent
from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid
from ghoshell_container import IoCContainer
from ghoshell_moss.shell.channel_runtime import ChannelRuntime, ChannelPath
import logging
import asyncio

_ChannelId = str
_ChannelFullPath = str


class ShellRuntime:

    def __init__(
            self,
            container: IoCContainer,
            main_channel: Channel,
    ):
        self.id = uuid()
        self.container: IoCContainer = container
        self.main_channel: Channel = main_channel

        # --- runtime --- #
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._channel_id_to_runtime_map: Dict[_ChannelId, ChannelRuntime] = {}
        """使用 channel id 指向所有的 channel runtime 实例. """
        self._channel_path_to_channel_map: Dict[_ChannelId, Channel] = {}
        """channel path 所指向的 channel id"""

        # --- lifecycle --- #

        self._starting = False
        self._started = False
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()
        # --- cache --- #
        self._logger = None

    @property
    def logger(self) -> LoggerItf:
        if self._logger is None:
            self._logger = self.container.get(LoggerItf) or logging.getLogger('moss')
        return self._logger

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f"ShellRuntime is not running")

    async def get_or_create_runtime(
            self,
            channel_path: str,
            /,
            channel: Optional[Channel] = None,
    ) -> Optional[ChannelRuntime]:
        """获取一个已经初始化的 channel runtime, 基于 a.b.c 这样的 path. """
        self._check_running()

        # prepare channel
        if channel is not None:
            pass
        else:
            # 永远动态构建.
            channel = self.main_channel.get_channel(channel_path)

        if channel is None:
            return None

        if not channel.is_running():
            # 动态启动 channel.
            channel_client = channel.bootstrap(self.container)
            await channel_client.start()
            # 重新注册映射关系.
        return await self.get_or_create_runtime_by_channel(channel)

    async def get_or_create_runtime_by_channel(self, channel: Channel) -> ChannelRuntime:
        """尝试获取或创建一个 channel runtime, 并且关闭掉 client id 中已经存在的 channel runtime"""
        self._check_running()
        if not channel.is_running():
            # 运行时启动 channel.
            channel_client = channel.bootstrap(self.container)
            await channel_client.start()
        channel_id = channel.client.id
        if channel_id in self._channel_id_to_runtime_map:
            # 先看是否已经存在.
            channel_runtime = self._channel_id_to_runtime_map[channel_id]
            # 存在的话, 仍然检查一下 client 实例是否一致.
            if channel_runtime.channel is channel:
                # 一致直接返回.
                return channel_runtime
            else:
                # 不一致的话关闭掉.
                await channel_runtime.close()

        # 创建新的 runtime, 记录到 channel runtime map 里.
        channel_runtime = await self.create_channel_runtime(channel)
        await channel_runtime.start()
        self._channel_id_to_runtime_map[channel_id] = channel_runtime
        return channel_runtime

    async def create_channel_runtime(self, channel: Channel) -> ChannelRuntime:
        """创建 channel runtime 实例. 不会去启动他们. """
        return ChannelRuntime(
            self.container,
            channel,
            self.dispatch_task_to_channel,
            stop_event=self._closing_event,
        )

    def _get_main_channel_runtime(self) -> ChannelRuntime:
        main_channel_id = self.main_channel.client.id
        return self._channel_id_to_runtime_map[main_channel_id]

    def add_task(self, *tasks: CommandTask) -> None:
        """
        添加 task 到运行时. 这些 task 会阻塞在 Channel Runtime 队列中直到获取执行机会.
        """
        if not self.is_running():
            # todo: log
            return
        main_runtime = self._get_main_channel_runtime()
        for task in tasks:
            if task.done():
                # 不处理.
                continue
            channel_paths = Channel.split_channel_path_to_names(task.meta.chan)
            main_runtime.add_task_with_paths(channel_paths, task)

    async def dispatch_task_to_channel(self, channel: Channel, paths: ChannelPath, task: CommandTask) -> None:
        self.logger.info("dispatching task %s to channel %s with paths", task, channel, paths)
        runtime = await self.get_or_create_runtime_by_channel(channel)
        runtime.add_task_with_paths(paths, task)

    async def channel_metas(
            self,
            available_only: bool = True,
            config: Dict[_ChannelFullPath, ChannelMeta] | None = None
    ) -> Dict[_ChannelFullPath, ChannelMeta]:
        channels = self.main_channel.all_channels()
        result = {}
        for channel_path, channel in channels.items():
            runtime = await self.get_or_create_runtime(channel_path, channel=channel)
            if runtime is None:
                continue
            if available_only and not runtime.is_available():
                continue
            meta = runtime.channel_meta()
            meta = meta.model_copy()
            # 替换 channel 的名称.
            meta.name = channel_path
            result[channel_path] = meta
        if config:
            result = self._update_chan_metas_with_config(result, config)
        return result

    async def commands(
            self,
            available_only: bool = True,
            config: Optional[Dict[_ChannelFullPath, ChannelMeta]] = None,
    ) -> Dict[_ChannelFullPath, Dict[str, Command]]:
        self._check_running()
        if not config:
            # 不从 meta, 而是从 runtime 里直接获取 commands.
            result = {}
            for channel_path, channel in self.main_channel.all_channels().items():
                runtime = await self.get_or_create_runtime(channel_path, channel)
                if available_only and not runtime.is_available():
                    continue
                real_commands = runtime.commands(available_only=available_only)
                wrapped_commands = {}
                for name, real_command in real_commands.items():
                    wrapped_command_mta = real_command.meta().model_copy()
                    # 替换所有的 command 的 channel 名称.
                    wrapped_command_mta.chan = channel_path
                    wrapped_commands[name] = CommandWrapper(wrapped_command_mta, real_command.__call__)
                result[channel_path] = wrapped_commands
            return result

        channel_metas = config
        result = {}
        for channel_path, meta in channel_metas.items():
            if available_only and not meta.available:
                continue
            runtime = await self.get_or_create_runtime(channel_path)
            if runtime is None:
                continue
            commands = runtime.commands(available_only=available_only)
            output_commands = {}
            for command_meta in meta.commands:
                if command_meta.name not in commands:
                    # 定义的命令并不存在.
                    continue
                real_command = commands[command_meta.name]
                wrapped_command_meta = real_command.meta().model_copy()
                # 修改了 channel path
                wrapped_command_meta.chan = channel_path
                wrapped_command = CommandWrapper(wrapped_command_meta, real_command.__call__)
                output_commands[command_meta.name] = wrapped_command
            result[channel_path] = output_commands
        return result

    @staticmethod
    def _update_chan_metas_with_config(
            metas: Dict[_ChannelFullPath, ChannelMeta],
            config: Dict[_ChannelFullPath, ChannelMeta],
    ) -> Dict[_ChannelFullPath, ChannelMeta]:
        result = {}
        for channel_path, meta in config.items():
            if channel_path not in metas:
                # 真实的 channel 不存在.
                continue
            origin_meta = metas[channel_path]
            configured_meta = meta.model_copy()
            configured_meta.available = meta.available and origin_meta.available
            result[channel_path] = configured_meta
        return result

    async def clear(self, *chans: str, recursively: bool = True) -> None:
        """
        清空指定的 channel. 如果 chans 为空, 则清空所有的 channel.
        """
        if len(chans) == 0:
            chans = self.main_channel.all_channels().keys()
            await self._clear(*chans)
            return

        elif recursively:
            paths = set()
            for chan in chans:
                self._recursive_get_runtime_channel_names(chan, paths)
            await self._clear(*paths)
        else:
            await self._clear(*chans)

    async def defer_clear(self, *chans: str, recursively: bool = True) -> None:
        """
        标记 channel 在得到新命令的时候, 先清空.
        """
        if len(chans) == 0:
            chans = self._channel_path_to_channel_map.keys()
            await self._defer_clear(*chans)
            return

        elif recursively:
            paths = set()
            for chan in chans:
                self._recursive_get_runtime_channel_names(chan, paths)
            await self._defer_clear(*paths)
        else:
            await self._defer_clear(*chans)

    def _recursive_get_runtime_channel_names(self, channel_path: str, channel_name_set: set[str]) -> None:
        if channel_path not in self._channel_path_to_channel_map:
            return
        channel_name_set.add(channel_path)
        channel = self._channel_path_to_channel_map[channel_path]
        # 递归寻找所有子节点.
        for child in channel.children().values():
            sub_path = Channel.join_channel_path(channel_path, child.name())
            self._recursive_get_runtime_channel_names(sub_path, channel_name_set)

    async def _clear(self, *chans: str) -> None:
        for chan in chans:
            runtime = await self.get_or_create_runtime(chan)
            if runtime is not None:
                await runtime.clear()

    async def _defer_clear(self, *chans: str) -> None:
        for chan in chans:
            runtime = await self.get_or_create_runtime(chan)
            if runtime is not None:
                await runtime.clear()

    def is_busy(self) -> bool:
        self._check_running()
        for runtime in self._channel_id_to_runtime_map.values():
            if not runtime.is_busy():
                return False
        return True

    def is_running(self) -> bool:
        return self._started and not self._closing_event.is_set() and self._event_loop is not None

    def is_idle(self) -> bool:
        self._check_running()
        for runtime in self._channel_id_to_runtime_map.values():
            if runtime.is_busy():
                return False
        return True

    async def wait_idle(self, timeout: float | None = None) -> None:
        if not self.is_running():
            return
        runtime_wait_idle = []
        for runtime in self._channel_id_to_runtime_map.values():
            runtime_wait_idle.append(runtime.wait_until_idle(timeout))
        # 等待所有的 idle.
        await asyncio.gather(*runtime_wait_idle)

    async def wait_closed(self, timeout: float | None = None) -> None:
        if not self.is_running():
            return
        await asyncio.wait_for(self._closed_event.wait(), timeout)

    # --- lifecycle --- #

    async def start(self) -> None:
        """
        启动 Shell 的 runtime.
        """
        if self._starting:
            return
        self._starting = True
        # 获取 loop 实例.
        self._event_loop = asyncio.get_running_loop()
        # 确保容器启动.
        await asyncio.to_thread(self.container.bootstrap)
        # 启动所有的 client.
        await self._recursive_bootstrap_channel(self.main_channel)
        # 启动所有的 runtime.
        await self._bootstrap_all_channel_runtimes()
        # 完成 channel runtime 的创建.
        self._started = True

    async def _bootstrap_all_channel_runtimes(self) -> None:
        # 所有的子孙 channel, 包含 main channel.
        all_channels = self.main_channel.all_channels()
        # 构建原始的 map.
        self._channel_path_to_channel_map = all_channels
        # 还有自身.
        self._channel_path_to_channel_map[''] = self.main_channel

        # 并行初始化所有的 runtime.
        bootstrap_runtimes = []
        for channel_path, channel in all_channels.items():
            channel_runtime = await self.create_channel_runtime(channel)
            if channel_runtime is None:
                self.logger.error(f"Channel {channel_path} can't create runtime")
                continue
            bootstrap_runtimes.append(channel_runtime.start())
            # 注册 path 和 id 之间的关系.
            client_id = channel.client.id
            self._channel_id_to_runtime_map[client_id] = channel_runtime
        # 启动所有的 runtime.
        await asyncio.gather(*bootstrap_runtimes)

    async def _recursive_bootstrap_channel(self, channel: Channel) -> None:
        """递归地启动这些  channel. """
        if not channel.is_running():
            # 有些 channel 可能在图里已经启动过了. channel 反正不允许成环.
            client = channel.bootstrap(self.container)
            await client.start()

        children = channel.children()
        gathering_tasks = []
        for child in children.values():
            gathering_tasks.append(self._recursive_bootstrap_channel(child))
        # 并发启动所有的 client.
        done = await asyncio.gather(*gathering_tasks)
        for t in done:
            if isinstance(t, Exception):
                # 并不中断启动.
                self.logger.exception(t)

    async def close(self) -> None:
        """
        shell 停止运行.
        """
        if self._closing_event.is_set():
            return
        self._closing_event.set()
        try:

            stop_runtimes = []
            for runtime in self._channel_id_to_runtime_map.values():
                stop_runtimes.append(runtime.close())
            # 关闭所有的 runtime. 关闭 runtime 就会关闭 client.
            done = await asyncio.gather(*stop_runtimes, return_exceptions=False)
            for t in done:
                if isinstance(t, Exception):
                    self.logger.exception(t)
                    # 结束时可以抛出异常.
                    raise t

            # 关闭 ioc 容器.
            self.container.shutdown()
        finally:
            self._closed_event.set()
            # 清空核心状态.
            self._channel_id_to_runtime_map.clear()
            self._channel_path_to_channel_map.clear()
            self._event_loop = None
            self._started = False
            self._starting = False
