import asyncio
import logging
from typing import Optional, Iterable, Callable, Any

from ghoshell_common.contracts import LoggerItf
from ghoshell_common.helpers import uuid
from ghoshell_container import Container, IoCContainer

from ghoshell_moss.core.concepts.channel import Channel, ChannelFullPath, ChannelMeta, ChannelBroker, ChannelCtx
from ghoshell_moss.core.concepts.command import (
    RESULT,
    BaseCommandTask,
    Command,
    CommandMeta,
    CommandTask,
    CommandWrapper,
    CommandUniqueName,
)
from ghoshell_moss.core.concepts.errors import CommandErrorCode, FatalError
from ghoshell_moss.core.concepts.interpreter import Interpreter
from ghoshell_moss.core.concepts.shell import InterpreterKind, MOSSShell
from ghoshell_moss.core.concepts.speech import Speech
from ghoshell_moss.core.concepts.states import BaseStateStore, StateStore
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.core.ctml.interpreter import CTMLInterpreter
from ghoshell_moss.core.shell.main_channel import MainChannel
from ghoshell_moss.speech.mock import MockSpeech
import contextlib

__all__ = ["DefaultShell", "new_shell"]


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
        self._name = name
        self._desc = description

        self._container = Container(parent=container, name="MOSShell")
        self._container.set(MOSSShell, self)
        self._main_channel = main_channel or MainChannel(name="", description="")

        self._speech: Speech | None = speech

        # state
        self._state_store: StateStore | None = None

        # logger
        self._logger = None

        # --- lifecycle --- #
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._exit_stack = contextlib.AsyncExitStack()

        self._main_loop_task: Optional[asyncio.Task] = None
        self._push_task_queue: asyncio.Queue[CommandTask | None] = asyncio.Queue()

        self._start: bool = False
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()

        # --- interpreter --- #
        self._interpreter: Optional[Interpreter] = None

        # --- runtime --- #
        self._main_broker: Optional[ChannelBroker] = None
        self._log_prefix = "[MOSSShell name=%s] " % self._name

    @property
    def container(self) -> IoCContainer:
        return self._container

    @property
    def states(self) -> StateStore:
        if self._state_store is None:
            raise RuntimeError("State store is not set")
        return self._state_store

    @property
    def speech(self) -> Speech:
        if self._speech is None:
            raise RuntimeError("Speech is not set")
        return self._speech

    async def __aenter__(self):
        if self._start:
            return
        self._start = True
        self._event_loop = asyncio.get_event_loop()
        # 进入开机过程.
        await self._exit_stack.__aenter__()
        for ctx_manager in self._bootstrap_stacks():
            # 进入每一个开启状态.
            await self._exit_stack.enter_async_context(ctx_manager())

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.logger.exception(exc_val)
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)

    def _bootstrap_stacks(self) -> Iterable[Callable]:
        yield self._ioc_context_manager
        yield self._state_store_context_manager
        yield self._speech_context_manager
        yield self._broker_context_manager
        yield self._main_loop_context_manager

    @contextlib.asynccontextmanager
    async def _ioc_context_manager(self):
        await asyncio.to_thread(self._container.bootstrap)

        # 日志准备.
        if self._logger is None:
            logger = self._container.get(LoggerItf)
            if logger is None:
                logger = logging.getLogger('moss')
                self._container.set(LoggerItf, self._logger)
            self._logger = logger

        yield
        await asyncio.to_thread(self._container.shutdown)

    @contextlib.asynccontextmanager
    async def _state_store_context_manager(self):
        if self._state_store is None:
            state_store = self._container.get(StateStore)
            if state_store is None:
                state_store = BaseStateStore(owner=f"shell/{self._name}")
                self._container.set(StateStore, self._state_store)
            self._state_store = state_store
        await self._state_store.start()
        yield
        await self._state_store.close()

    @contextlib.asynccontextmanager
    async def _speech_context_manager(self):
        """
        启动关闭音频模块.
        """
        if self._speech is None:
            speech = self._container.get(Speech)
            if speech is None:
                speech = MockSpeech()
                self._container.set(Speech, speech)
            self._speech = speech
        await self.speech.start()
        yield
        await self.speech.close()

    @contextlib.asynccontextmanager
    async def _broker_context_manager(self):
        """
        开启 channel broker.
        """
        self._main_broker = self._main_channel.bootstrap(self._container)
        # 开启 Broker
        await self._main_broker.start()
        yield
        # 关闭 Broker. k
        await self._main_broker.close()

    @contextlib.asynccontextmanager
    async def _main_loop_context_manager(self):
        self._main_loop_task = asyncio.create_task(self._push_task_loop())
        yield
        if not self._main_loop_task.done():
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass

    async def _push_task_loop(self):
        try:
            failed_count = 0
            while not self._closing_event.is_set():
                try:
                    _queue = self._push_task_queue
                    item = await asyncio.wait_for(_queue.get(), timeout=1)
                    if item is None:
                        # 接受毒丸防止死锁.
                        continue
                except asyncio.TimeoutError:
                    continue

                try:
                    if not self.is_running():
                        item.fail(CommandErrorCode.NOT_RUNNING.error("shell is not running"))
                        continue
                    await self._main_broker.push_task(item)
                    # 清零.
                    failed_count = 0
                except asyncio.CancelledError:
                    raise
                except FatalError as e:
                    self.logger.exception("%s fatal error: %s", self._log_prefix, e)
                    raise
                except Exception as e:
                    # 不处理特殊异常.
                    self.logger.exception("%s push task exception: %s", self._log_prefix, e)
                    failed_count += 1
                    # 连续 5 个特殊异常. 本来一个都应该没有
                    if failed_count > 5:
                        # 中断主循环.
                        raise
        finally:
            self.logger.info("%s push task loop done", self._log_prefix)

    # --- lifetime functions --- #

    @property
    def runtime(self) -> ChannelBroker:
        self._check_running()
        return self._main_broker

    @property
    def logger(self) -> LoggerItf:
        if self._logger is None:
            logger = self._container.get(LoggerItf) or logging.getLogger("moss")
            self._logger = logger
        return self._logger

    def is_running(self) -> bool:
        self_running = self._start and not self._closing_event.is_set()
        return self_running and self._main_broker and self._main_broker.is_running()

    async def wait_connected(self, *channel_paths: str) -> None:
        if not self.is_running():
            return
        paths = list(channel_paths)
        if len(paths) == 0:
            await self._main_broker.wait_connected()

        waiting = []
        for path in paths:
            broker = await self._main_broker.fetch_sub_broker(path)
            if broker is None or not broker.is_running():
                continue
            waiting.append(broker.wait_connected())
        if len(waiting) > 0:
            await asyncio.gather(*waiting)

    async def wait_until_idle(self, timeout: float | None = None) -> None:
        if not self.is_running():
            return
        if timeout is None:
            await self._main_broker.wait_idle()
        else:
            await asyncio.wait_for(self._main_broker.wait_idle(), timeout=timeout)

    def is_closed(self) -> bool:
        return self._closed_event.is_set()

    def _check_running(self):
        if not self.is_running():
            raise RuntimeError(f"Shell {self._name} not running")

    def is_idle(self) -> bool:
        return self.is_running() and self._main_broker.is_idle()

    def _interpreter_callback_task(self, task: CommandTask | None) -> None:
        if task is not None:
            self.push_task(task)

    async def interpreter(
            self,
            kind: InterpreterKind = "clear",
            *,
            stream_id: Optional[int] = None,
            config: dict[ChannelFullPath, ChannelMeta] | None = None,
            prepare_timeout: float = 2.0,
    ) -> Interpreter:
        self._check_running()

        # 方便理解不同类型的处理逻辑. 看待 interpreter 的副作用问题.
        callback = None
        if kind == "clear":
            # clear 会先清空.
            await self.clear()
            # 清除当前存在的 interpretation.
            await self.stop_interpretation()
            callback = self._interpreter_callback_task
        elif kind == "dry_run":
            # dry_run 不会对 shell 产生真实影响, 可以用来做纯解析.
            callback = None
        elif kind == "append":
            # append 会追加命令, 而不是清除.
            callback = self._interpreter_callback_task
            if self._interpreter and self._interpreter.is_running():
                # 停止旧的 interpreter 继续提交新的信息.
                self._interpreter.commit()
            self._interpreter = None

        # 阻塞等待刷新结果.
        await self.refresh_metas(timeout=prepare_timeout)
        config = self.channel_metas(available_only=True, config=config)
        commands = self.commands(available_only=True, config=config)
        interpreter = CTMLInterpreter(
            commands=commands,
            speech=self.speech,
            stream_id=stream_id or uuid(),
            callback=callback,
            logger=self.logger,
            channel_metas=config,
        )

        # 会接受回调的话, 更新最新的 interpreter.
        if callback is not None:
            self._interpreter = interpreter
        return interpreter

    def with_speech(self, speech: Speech) -> None:
        if self.is_running():
            raise RuntimeError(f"Shell {self._name} already running")
        self._speech = speech

    @property
    def main_channel(self) -> Channel:
        return self._main_channel

    async def refresh_metas(self, timeout: float | None = None) -> None:
        if not self.is_running():
            return
        # 保证这个任务最终被执行完毕吧.
        refresh_meta_task = self._event_loop.create_task(self._main_broker.refresh_metas(force=True))
        if timeout is not None:
            sleep_task = asyncio.create_task(asyncio.sleep(timeout))
            done, pending = await asyncio.wait([refresh_meta_task, sleep_task], return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            # 有任何一个结束了就退出.
        else:
            await refresh_meta_task

    def channel_metas(
            self,
            available_only: bool = True,
            config: Optional[dict[ChannelFullPath, ChannelMeta]] = None,
    ) -> dict[str, ChannelMeta]:
        if not self.is_running():
            return {}
        metas = self._main_broker.metas()
        result = {}

        if config is not None:
            # 对齐人工配置项.
            for channel_full_path, channel_meta in config.items():
                origin_channel_meta = metas.get(channel_full_path)
                if origin_channel_meta is None:
                    continue

                config_meta = channel_meta.model_copy()
                # 状态对齐.
                config_meta.available = config_meta.available and origin_channel_meta.available
                if available_only and not config_meta.available:
                    continue
                config_meta.channel_id = origin_channel_meta.channel_id
                config_meta.dynamic = True
                # instruction 用配置好的.
                config_meta.instructions = config_meta.instructions or origin_channel_meta.instructions
                # 这里用更新的.
                config_meta.context = origin_channel_meta.context
                commands = []
                exists = set(cmd.name for cmd in origin_channel_meta.commands)
                for cmd in config_meta.commands:
                    if cmd.name not in exists:
                        continue
                    commands.append(cmd)
                config_meta.commands = commands
                result[ChannelMeta.channel_full_path] = config_meta
            return result

        elif not available_only:
            # 直接返回.
            return metas
        # 检查 available only.
        for channel_path, channel_meta in metas.items():
            if channel_meta.available:
                result[channel_path] = channel_meta
        return result

    def push_task(self, *tasks: CommandTask) -> None:
        self._check_running()
        # 线程安全加入 tasks.
        self._event_loop.call_soon_threadsafe(self._push_task_queue.put_nowait, *tasks)

    async def stop_interpretation(self) -> None:
        self._check_running()
        if self._interpreter is not None and self._interpreter.is_running():
            # 考虑线程安全问题. 先简单做一层防御.
            stop_task = self._event_loop.create_task(self._interpreter.stop())
            self._interpreter = None
            await stop_task

    async def wait_until_closed(self) -> None:
        if not self.is_running():
            return
        await self._closed_event.wait()

    def commands(
            self,
            available_only: bool = True,
            *,
            config: dict[ChannelFullPath, ChannelMeta] | None = None,
            exec_in_chan: bool = False,
    ) -> dict[ChannelFullPath, dict[str, Command]]:
        self._check_running()

        commands = self._main_broker.commands(available_only=available_only)
        if config is None:
            return commands

        # --- config --- #

        # 不从 meta, 而是从 runtime 里直接获取 commands.
        result = {}
        for channel_path, configured_channel_meta in config.items():
            if channel_path not in commands:
                continue
            configured_commands = {}
            channel_commands = commands[channel_path]
            for configured_command_meta in configured_channel_meta.commands:
                if available_only and not configured_command_meta.available:
                    continue
                real_command = channel_commands.get(configured_command_meta.name)
                if real_command is None:
                    continue
                configured_command = CommandWrapper.wrap(real_command, meta=configured_command_meta)
                configured_commands[configured_command_meta.name] = configured_command
            result[channel_path] = configured_commands
        return commands

    async def get_command(self, chan: str, name: str, /, exec_in_chan: bool = False) -> Optional[Command]:
        self._check_running()
        broker = await self._main_broker.fetch_sub_broker(chan)
        if broker is None or not broker.is_available():
            return None

        real_command = broker.get_self_command(name)
        if not exec_in_chan:
            return real_command
        return self._wrap_real_command(chan, real_command, None)

    def _wrap_real_command(self, chan: str, command: Command, meta: CommandMeta | None) -> CommandWrapper:
        """
        确保 Shell 提供的 Command 一定会在 channel 里执行.
        """
        origin_func = command.__call__
        if isinstance(command, CommandWrapper):
            origin_func = command.func
        _broker = ChannelCtx.broker()
        _task = ChannelCtx.task()
        print("++++++++++++", _broker, _task)

        # 创建一个入栈函数.
        async def _exec_in_chan_func(*args, **kwargs) -> Any:
            # 检查是不是在 channel 里被运行的.
            _broker = ChannelCtx.broker()
            if _broker is not None:
                # 如果是在 channel 里运行的, 则直接调用其真函数运行结果即可.
                return await origin_func(*args, **kwargs)

            # 并不是在 broker 里运行的, 检查是否有 task 对象.
            task = ChannelCtx.task()
            if task is not None:
                # 如果上下文里已经有了 task, 则仍然执行结果.
                return await origin_func(*args, **kwargs)
            else:
                # 发送到 broker 里, 等待 Channel 运行它.
                task = BaseCommandTask.from_command(
                    command,
                    chan,
                    args=args,
                    kwargs=kwargs,
                )
                self.push_task(task)
                return await task

        command = CommandWrapper(meta or command.meta(), _exec_in_chan_func, available_fn=command.is_available)
        return command

    async def clear(self) -> None:
        if not self.is_running():
            return
        _queue = self._push_task_queue
        # 直接换新的 _queue.
        self._push_task_queue = asyncio.Queue()

        async def _clear_old_queue() -> None:
            """
            清空一个队列的安全做法.
            """
            _queue.put_nowait(None)
            while not _queue.empty():
                try:
                    # queue.get 如果不暂停它, 它会死锁住
                    item = await asyncio.wait_for(_queue.get(), timeout=1)
                    if item is None:
                        break
                    elif isinstance(item, CommandTask):
                        item.fail(CommandErrorCode.CLEARED.error("cleared by shell"))
                except asyncio.TimeoutError:
                    # 不非空, 但自己没拿到.
                    # 塞一个毒丸, 确认在 clear 之前一定要亲手拿到毒丸.
                    _queue.put_nowait(None)
                    continue
            _queue.put_nowait(None)
            _queue.task_done()

        clear_queue = self._event_loop.create_task(_clear_old_queue())
        await clear_queue
        _ = await asyncio.gather(self.speech.clear(), self._main_broker.clear())


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
