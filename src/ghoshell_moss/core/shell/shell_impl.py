import asyncio
import logging
from typing import Optional

from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid
from ghoshell_container import Container, IoCContainer

from ghoshell_moss.core.concepts.channel import Channel, ChannelFullPath, ChannelMeta
from ghoshell_moss.core.concepts.command import (
    RESULT,
    BaseCommandTask,
    Command,
    CommandMeta,
    CommandTask,
    CommandWrapper,
)
from ghoshell_moss.core.concepts.errors import CommandErrorCode
from ghoshell_moss.core.concepts.interpreter import Interpreter
from ghoshell_moss.core.concepts.shell import InterpreterKind, MOSSShell, Speech
from ghoshell_moss.core.concepts.states import MemoryStateStore, StateStore
from ghoshell_moss.core.ctml.interpreter import CTMLInterpreter
from ghoshell_moss.core.shell.main_channel import MainChannel
from ghoshell_moss.core.shell.shell_runtime import ShellRuntime
from ghoshell_moss.speech.mock import MockSpeech

__all__ = ["DefaultShell", "new_shell"]


class ExecuteInChannelRuntimeCommand(Command[RESULT]):
    """
    the command will execute in channel runtime

    一种特殊的 Command.
    它被当作函数使用的时候, 命令不会立刻执行, 而是发送到 ChannelRuntime 里等待执行.
    预计用来做纯代码编程时使用.
    """

    def __init__(self, shell: "DefaultShell", command: Command):
        self._shell = shell
        self._command = command

    def name(self) -> str:
        return self._command.name()

    def is_available(self) -> bool:
        return self._command.is_available()

    def meta(self) -> CommandMeta:
        return self._command.meta()

    async def refresh_meta(self) -> None:
        await self._command.refresh_meta()

    async def __call__(self, *args, **kwargs) -> RESULT:
        task = BaseCommandTask.from_command(self._command, *args, **kwargs)
        try:
            # push task into the shell
            runtime = await self._shell.runtime.get_or_create_runtime(task.meta.chan)
            if runtime is None:
                raise CommandErrorCode.NOT_AVAILABLE.error("Not available")

            runtime.add_task(task)
            await task.wait(throw=False)
            # 减少抛出异常的调用栈.
            if exp := task.exception():
                raise exp
            return task.result()
        finally:
            if not task.done():
                task.cancel()


class DefaultShell(MOSSShell):
    def __init__(
        self,
        *,
        name: str = "shell",
        description: Optional[str] = None,
        container: IoCContainer | None = None,
        main_channel: Channel | None = None,
        speech: Optional[Speech] = None,
        state_store: Optional[StateStore] = None,
    ):
        self.name = name
        self.container = Container(parent=container, name="MOSShell")
        self.container.set(MOSSShell, self)
        self._main_channel = main_channel or MainChannel(name="", description="")
        self._desc = description

        # output
        if not speech:
            speech = MockSpeech()
        self.speech: Speech = speech
        self.container.set(Speech, speech)
        # state
        if not state_store:
            state_store = MemoryStateStore(owner=self.name)
        self.state_store: StateStore = state_store
        self.container.set(StateStore, state_store)

        # --- lifecycle --- #
        self._starting = False
        self._started = False
        self._closing = False
        self._closed = False
        self._logger = None

        # --- interpreter --- #
        self._interpreter: Optional[Interpreter] = None

        # init main channel
        self._runtime: Optional[ShellRuntime] = None

    @property
    def runtime(self) -> ShellRuntime:
        self._check_running()
        return self._runtime

    @property
    def logger(self) -> LoggerItf:
        if self._logger is None:
            logger = self.container.get(LoggerItf)
            if logger is None:
                logger = logging.getLogger("moss")
                self.container.set(LoggerItf, logger)
            self._logger = logger
        return self._logger

    def is_running(self) -> bool:
        self_running = self._started and not self._closing
        return self_running and self._runtime and self._runtime.is_running()

    async def wait_connected(self, *channel_paths: str) -> None:
        paths = list(channel_paths)
        _all = self.main_channel.all_channels()
        if not paths:
            channels = _all
        else:
            channels = {}
            for path in paths:
                if chan := _all.get(path):
                    channels[path] = chan
        wait_tasks = []
        for chan in channels.values():
            if chan.is_running():
                wait_tasks.append(chan.broker.wait_connected())
        await asyncio.gather(*wait_tasks)

    def is_closed(self) -> bool:
        return self._closing

    def _check_running(self):
        if not self.is_running():
            raise RuntimeError(f"Shell {self.name} not running")

    def is_idle(self) -> bool:
        self._check_running()
        return self._runtime.is_idle()

    def _append_command_task(self, task: CommandTask | None) -> None:
        if task is not None:
            self._runtime.add_task(task)

    async def interpreter(
        self,
        kind: InterpreterKind = "clear",
        *,
        stream_id: Optional[int] = None,
        channel_metas: dict[ChannelFullPath, ChannelMeta] | None = None,
    ) -> Interpreter:
        close_running_interpreter = None
        if self._interpreter is not None:
            if self._interpreter.is_running():
                close_running_interpreter = self._interpreter
                self._interpreter = None

        async def _on_start():
            # clear only when interpreter start
            self._check_running()
            if not self.is_idle():
                if kind == "defer_clear":
                    await self.defer_clear()
                elif kind == "clear":
                    await self.clear()

                if close_running_interpreter is not None and not "dry_run":
                    await close_running_interpreter.stop()

        await self._runtime.refresh_metas()
        channel_metas = await self._runtime.channel_metas(available_only=True, config=channel_metas)
        commands = await self._runtime.commands(available_only=True, config=channel_metas)
        callback = self._append_command_task if kind != "dry_run" else None
        interpreter = CTMLInterpreter(
            commands=commands,
            speech=self.speech,
            stream_id=stream_id or uuid(),
            callback=callback,
            logger=self.logger,
            on_startup=_on_start,
            channel_metas=channel_metas,
        )
        if callback is not None:
            self._interpreter = interpreter
        return interpreter

    def with_speech(self, speech: Speech) -> None:
        if self.is_running():
            raise RuntimeError(f"Shell {self.name} already running")
        self.speech = speech

    @property
    def main_channel(self) -> Channel:
        return self._main_channel

    def channels(self) -> dict[str, Channel]:
        return self.main_channel.all_channels()

    async def channel_metas(
        self,
        available_only: bool = True,
        /,
        config: dict[ChannelFullPath, ChannelMeta] | None = None,
        refresh: bool = False,
    ) -> dict[str, ChannelMeta]:
        self._check_running()
        if refresh:
            await self._runtime.refresh_metas()
        return await self._runtime.channel_metas(available_only=available_only, config=config)

    def add_task(self, *tasks: CommandTask) -> None:
        self._check_running()
        self._runtime.add_task(*tasks)

    async def stop_interpretation(self) -> None:
        if self._interpreter is not None:
            await self._interpreter.stop()
            self._interpreter = None

    async def wait_until_closed(self) -> None:
        if not self.is_running():
            return
        await self._runtime.wait_closed()

    async def commands(
        self, available_only: bool = True, /, config: dict[ChannelFullPath, ChannelMeta] | None = None
    ) -> dict[ChannelFullPath, dict[str, Command]]:
        self._check_running()
        return await self._runtime.commands(available_only=True, config=config)

    async def get_command(self, chan: str, name: str, /, exec_in_chan: bool = False) -> Optional[Command]:
        self._check_running()
        runtime = await self._runtime.get_or_create_runtime(chan)
        if runtime is None:
            return None
        real_command = runtime.channel.broker.get_command(name)
        meta = real_command.meta().model_copy()
        meta.chan = chan
        command = CommandWrapper(meta, real_command.__call__)
        return ExecuteInChannelRuntimeCommand(self, command)

    async def wait_until_idle(self, timeout: float | None = None) -> None:
        if not self.is_running():
            return
        await self._runtime.wait_idle(timeout)

    async def clear(self, *chans: str) -> None:
        self._check_running()
        await self._runtime.clear(*chans, recursively=True)

    async def defer_clear(self, *chans: str) -> None:
        self._check_running()
        await self._runtime.defer_clear(*chans, recursively=True)

    async def system_prompt(self) -> str:
        # todo
        raise NotImplementedError()

    async def start(self) -> None:
        if self._closing:
            self.logger.warning("Shell already closing")
            raise RuntimeError("Shell runtime can not re-enter")
        if self._starting:
            self.logger.info("Shell already started")
            return
        self.logger.info("Shell starting")
        self._starting = True
        await self.speech.start()
        shell_runtime = ShellRuntime(
            Container(name="shell_runtime", parent=self.container),
            self.main_channel,
        )
        # 启动容器. 通常已经启动了.
        await shell_runtime.start()
        self._runtime = shell_runtime
        # 启动自己的 task
        self._started = True
        self.logger.info("Shell started")

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self._interpreter is not None:
            await self._interpreter.stop()
            self._interpreter = None
        await self._runtime.close()
        self._logger.info("Shell %s runtime closed", self.name)
        await self.speech.close()
        self._logger.info("Shell %s speech closed", self.name)
        self._runtime = None
        self._closed = True
        self._logger.info("Shell %s closed", self.name)


def new_shell(
    name: str = "shell",
    description: Optional[str] = None,
    container: IoCContainer | None = None,
    main_channel: Channel | None = None,
    speech: Optional[Speech] = None,
) -> MOSSShell:
    """语法糖, 好像不甜"""
    return DefaultShell(
        name=name,
        description=description,
        container=container,
        main_channel=main_channel,
        speech=speech,
    )
