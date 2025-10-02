from typing import Dict, Optional
from ghoshell_moss.concepts.shell import MOSSShell, Output, InterpreterKind
from ghoshell_moss.concepts.command import Command, CommandTask, CommandWrapper
from ghoshell_moss.concepts.channel import Channel, ChannelMeta
from ghoshell_moss.concepts.interpreter import Interpreter
from ghoshell_moss.ctml.interpreter import CTMLInterpreter
from ghoshell_moss.mocks.outputs import ArrOutput
from ghoshell_moss.shell.main_channel import MainChannel
from ghoshell_moss.shell.runtime import ChannelRuntimeImpl
from ghoshell_moss.helpers.asyncio_utils import ThreadSafeEvent, TreeNotify
from ghoshell_common.helpers import uuid
from ghoshell_container import IoCContainer, Container
import logging
import asyncio

__all__ = ['DefaultShell', 'new_shell']


class DefaultShell(MOSSShell):

    def __init__(
            self,
            *,
            name: str = "shell",
            description: Optional[str] = None,
            container: IoCContainer | None = None,
            main_channel: Channel | None = None,
            output: Optional[Output] = None,
    ):
        self.name = name
        self.container = Container(parent=container, name=f"MOSShell")
        self.container.set(MOSSShell, self)
        # output
        self._output: Output = output or self.container.get(Output) or ArrOutput()
        self.container.set(Output, self._output)
        # logger
        self.logger = self.container.get(logging.Logger) or logging.getLogger("moss")
        self.container.set(logging.Logger, self.logger)

        # init main channel
        self._main_channel = main_channel or MainChannel(
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
        self._closed_event: asyncio.Event = asyncio.Event()

    def is_running(self) -> bool:
        return self._started and not self._stop_event.is_set() and self._main_channel_runtime.is_running()

    def is_idle(self) -> bool:
        return self.is_running() and not self._idle_notifier.is_set()

    def interpreter(
            self,
            kind: InterpreterKind = "clear",
            *,
            stream_id: Optional[int] = None,
    ) -> Interpreter:
        close_running_interpreter = None
        if self._interpreter is not None:
            if self._interpreter.is_running():
                close_running_interpreter = self._interpreter
                self._interpreter = None

        async def _on_start():
            self._check_running()

            if not self.is_idle():
                if kind == "defer_clear":
                    await self.defer_clear()
                elif kind == "clear":
                    await self.clear()

                if close_running_interpreter is not None:
                    await self._interpreter.stop()

        callback = self._append_command_task if kind != "dry_run" else None
        interpreter = CTMLInterpreter(
            commands=self.commands().values(),
            output=self._output,
            stream_id=stream_id or uuid(),
            callback=callback,
            logger=self.logger,
            on_startup=_on_start,
        )
        if callback is not None:
            self._interpreter = interpreter
        return interpreter

    def _append_command_task(self, task: CommandTask | None) -> None:
        self._check_running()
        try:
            if task is not None:
                self._main_channel_runtime.append(task)
        except Exception as exc:
            self.logger.exception(exc)
            self._stop_event.set()

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
        commands = {}
        for c in self._main_channel_runtime.commands(recursive=True, available_only=available):
            commands[c.name()] = c
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

    def append(self, *tasks: CommandTask) -> None:
        self._check_running()
        self._main_channel_runtime.append(*tasks)

    async def clear(self, *chans: str) -> None:
        self._check_running()
        names = list(chans)
        if len(names) == 0:
            names = [""]
        for name in names:
            channel_runtime = await self._main_channel_runtime.get_chan_runtime(name)
            if channel_runtime is not None:
                await channel_runtime.clear()

    async def wait_until_closed(self) -> None:
        if not self.is_running():
            return
        await self._closed_event.wait()

    async def wait_until_idle(self, timeout: float | None = None) -> None:
        if not self.is_running():
            return
        await asyncio.wait_for(self._idle_notifier.wait(), timeout=timeout)

    async def defer_clear(self, *chans: str) -> None:
        self._check_running()
        names = list(chans)
        if len(names) == 0:
            await self._main_channel_runtime.defer_clear()
            return
        # 可以并行执行.
        clearing = []
        for name in names:
            child = await self._main_channel_runtime.get_chan_runtime(name)
            clearing.append(child.defer_clear())
        await asyncio.gather(*clearing)

    async def system_prompt(self) -> str:
        # todo
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
        self.container.shutdown()
        self._closed_event.set()


def new_shell(
        name: str = "shell",
        description: Optional[str] = None,
        container: IoCContainer | None = None,
        main_channel: Channel | None = None,
        output: Optional[Output] = None,
) -> MOSSShell:
    """语法糖, 好像不甜"""
    return DefaultShell(
        name=name,
        description=description,
        container=container,
        main_channel=main_channel,
        output=output,
    )
