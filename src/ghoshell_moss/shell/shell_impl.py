from typing import Dict, Optional, List, Iterable
from typing_extensions import Literal, Self
from ghoshell_moss.concepts.shell import MOSSShell, Output, NewInterpreterKind
from ghoshell_moss.concepts.command import Command, CommandTask, CommandWrapper
from ghoshell_moss.concepts.channel import Channel, ChannelMeta
from ghoshell_moss.concepts.interpreter import Interpreter
from ghoshell_moss.ctml.interpreter import CTMLInterpreter
from ghoshell_moss.mocks.outputs import ArrOutput
from ghoshell_moss.shell.main_channel import ShellMainChannel
from ghoshell_moss.shell.runtime import ChannelRuntimeImpl
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent, TreeNotify
from ghoshell_common.helpers import uuid
from ghoshell_container import IoCContainer, Container
import logging
import asyncio


class ShellImpl(MOSSShell):

    def __init__(
            self,
            *,
            description: Optional[str] = None,
            container: IoCContainer | None = None,
            main_channel: Channel | None = None,
            output: Optional[Output] = None,
    ):
        self.container = Container(parent=container, name=f"MOSShell")
        self.container.set(MOSSShell, self)
        # output
        self._output: Output = output or self.container.get(Output) or ArrOutput()
        self.container.set(Output, self._output)
        # logger
        self.logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        self.container.set(logging.Logger, self.logger)

        # init main channel
        self._main_channel = main_channel or ShellMainChannel(
            name="",
            block=True,
            description=description or "",
        )

        # --- lifecycle --- #
        self._starting = False
        self._started = False
        self._closing = False

        self._stop_event = ThreadSafeEvent()
        self._stopped_event = ThreadSafeEvent()
        self._running_loop: Optional[asyncio.AbstractEventLoop] = None
        self._idle_notifier = TreeNotify(name="")

        # --- runtime --- #
        self._main_channel_runtime = ChannelRuntimeImpl(
            container=self.container,
            channel=self._main_channel,
            logger=self.logger,
            stop_event=self._stop_event,
            is_idle_notifier=self._idle_notifier,
        )
        # --- configuration --- #
        self._configured_channel_metas: Optional[Dict[str, ChannelMeta]] = None
        # --- interpreter --- #
        self._interpreter: Optional[Interpreter] = None

    def is_running(self) -> bool:
        return self._started and not self._stop_event.is_set() and self._main_channel_runtime.is_running()

    def is_idle(self) -> bool:
        return self.is_running() and not self._idle_notifier.is_set()

    def interpret(
            self,
            kind: NewInterpreterKind = "clear",
            *,
            stream_id: Optional[int] = None,
    ) -> Interpreter:
        self._check_running()
        if self._interpreter is not None:
            self._running_loop.call_soon_threadsafe(self._interpreter.stop)
            self._interpreter = None
            if kind == "defer_clear":
                self._running_loop.call_soon_threadsafe(self.defer_clear)
            elif kind == "clear":
                self._running_loop.call_soon_threadsafe(self.clear)
        callback = self._append_command_task if kind != "dry_run" else None
        interpreter = CTMLInterpreter(
            commands=self.commands().values(),
            output=self._output,
            stream_id=stream_id or uuid(),
            callback=callback,
            logger=self.logger,
        )
        if callback is not None:
            self._interpreter = interpreter
        return interpreter

    def _append_command_task(self, task: CommandTask | None) -> None:
        self._check_running()
        self._running_loop.call_soon_threadsafe(self._main_channel_runtime.append, task)

    def with_output(self, output: Output) -> None:
        self._output = output

    @property
    def main_channel(self) -> Channel:
        return self._main_channel

    def register(self, *channels: Channel, parent: str = "") -> None:
        if parent == "":
            self._main_channel.with_children(*channels)
        else:
            parent_channel = self._main_channel.descendants().get(parent, None)
            if parent_channel is None:
                raise KeyError(f"Channel {parent} not found")
            parent_channel.with_children(*channels)
        for channel in channels:
            self._running_loop.call_soon_threadsafe(
                self._main_channel_runtime.get_or_create_child_runtime,
                channel,
            )

    def configure(self, *metas: ChannelMeta) -> None:
        metas = {}
        for meta in metas:
            metas[meta.name] = meta
        if len(metas) > 0:
            self._configured_channel_metas = metas

    def channels(self) -> Dict[str, Channel]:
        channels = {"": self._main_channel}
        for name, channel in self._main_channel.descendants().items():
            channels[name] = channel
        return channels

    async def channel_metas(self) -> Dict[str, ChannelMeta]:
        self._check_running()
        channels = self.channels()
        if self._configured_channel_metas is not None:
            result = {}
            for meta in self._configured_channel_metas.values():
                meta = await self._update_channel_meta_in_runtime(meta, channels)
                result[meta.name] = meta
            return result
        else:
            return await self._get_all_channel_metas()

    async def _get_all_channel_metas(self) -> Dict[str, ChannelMeta]:
        result = {}
        channels = self.channels()
        for name, channel in channels.items():
            meta = channel.client.meta(no_cache=True)
            result[name] = meta
        return result

    async def _get_channel_metas_in_runtime(self, metas: Dict[str, ChannelMeta]) -> Dict[str, ChannelMeta]:
        result = {}
        # 根据已经配置的
        channels = self._main_channel.descendants()
        channels[""] = self._main_channel

        for name, meta in metas.items():
            result[name] = meta
        return result

    async def _update_channel_meta_in_runtime(self, meta: ChannelMeta, channels: Dict[str, Channel]) -> ChannelMeta:
        # 如果这个 meta 并没有实际的 channel 支持, 则将它设置为不可用.
        meta = meta.model_copy()
        name = meta.name
        if name not in channels:
            meta.available = False
            return meta

        runtime = await self._main_channel_runtime.get_chan_runtime(name)
        if runtime is None:
            meta.available = False
            return meta

        meta.available = runtime.is_available()
        if meta.available:
            # commands map
            commands = {c.name(): c for c in runtime.commands(recursive=False, available_only=False)}
            # change available.
            for command_meta in meta.commands:
                if command_meta.name not in commands:
                    command_meta.available = False
                else:
                    command_meta.available = commands[name].is_available()
        return meta

    def commands(self, available: bool = True) -> Dict[str, Command]:
        """
        动态获取 commands. 因为可能会有变动.
        """
        self._check_running()
        commands = {c.name(): c for c in self._main_channel_runtime.commands(recursive=True, available_only=available)}
        if self._configured_channel_metas is None:
            return commands
        else:
            result = {}
            for name, meta in self._configured_channel_metas.items():
                if not meta.available:
                    continue
                for command_meta in meta.commands:
                    if command_meta.name not in commands:
                        continue
                    real_command = commands[command_meta.name]
                    wrapped = CommandWrapper(real_command, meta=command_meta)
                    result[name] = wrapped
        return commands

    async def append(self, *tasks: CommandTask) -> None:
        self._check_running()
        await self._main_channel_runtime.append(*tasks)

    async def clear(self, *chans: str) -> None:
        self._check_running()
        names = list(chans)
        if len(names) == 0:
            names = [""]
        for name in names:
            channel_runtime = await self._main_channel_runtime.get_chan_runtime(name)
            if channel_runtime is not None:
                await channel_runtime.clear()

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        self._check_running()
        return self._idle_notifier.wait_sync(timeout)

    async def wait_until_idle(self, timeout: float | None = None) -> None:
        self._check_running()
        await asyncio.wait_for(self._idle_notifier.wait(), timeout=timeout)

    async def defer_clear(self, *chans: str) -> None:
        self._check_running()
        names = list(chans)
        if len(names) == 0:
            await self._main_channel_client.defer_clear()
            return
        # 可以并行执行.
        clearing = []
        for name in names:
            child = await self._main_channel_client.get_chan_runtime(name)
            clearing.append(child.defer_clear())
        await asyncio.gather(*clearing)

    async def system_prompt(self) -> str:
        raise NotImplementedError()

    def _check_running(self) -> None:
        if not self.is_running():
            raise RuntimeError(f"Shell not running")

    async def start(self) -> None:
        if self._closing:
            raise RuntimeError("shell runtime can not re-enter")
        if self._starting:
            return
        self._starting = True
        self._running_loop = asyncio.get_running_loop()
        # 启动容器. 通常已经启动了.
        await asyncio.to_thread(self.container.bootstrap)

        # 启动所有的 runtime.
        await self._main_channel_runtime.start()
        # 启动自己的 task
        self._started = True

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self._stop_event.set()
        # 先关闭所有的 runtime. 递归关闭.
        await self._main_channel_runtime.close()
