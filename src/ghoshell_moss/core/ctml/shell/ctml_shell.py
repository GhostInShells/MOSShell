import asyncio
import contextlib
import logging
from collections.abc import Callable, Iterable
from typing import Any, Optional, AsyncGenerator

from ghoshell_moss.message import unique_id
from ghoshell_container import Container, IoCContainer

from ghoshell_moss.contracts.logger import get_moss_logger, LoggerItf
from ghoshell_moss.message import Message
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.core.concepts.channel import (
    Channel,
    ChannelCtx,
    ChannelFullPath,
    ChannelMeta,
    ChannelRuntime,
)
from ghoshell_moss.core.blueprint import PrimeChannel
from ghoshell_moss.core.concepts.command import (
    BaseCommandTask,
    Command,
    CommandMeta,
    CommandTask,
    CommandWrapper,
)
from ghoshell_moss.core.concepts.interpreter import Interpreter, Interpretation
from ghoshell_moss.core.concepts.shell import InterpreterKind, MOSShell
from ghoshell_moss.core.concepts.topic import Topic, TopicModel
from ghoshell_moss.core.concepts.errors import PausedError
from ghoshell_moss.core.ctml.interpreter import CTMLInterpreter
from ghoshell_moss.core.ctml.versions import get_moss_ctml_meta_instruction, CTML_VERSION
from ghoshell_moss.core.ctml.v1_0.prompts import make_static_messages, make_dynamic_messages
from ghoshell_moss.core.ctml.shell.ctml_main import create_ctml_main_chan, default_primitive_map
from ghoshell_moss.core.helpers import ThreadSafeEvent, ThreadSafeFuture
from ghoshell_moss.core.speech.null import NullSpeech
from ghoshell_moss.core.speech.speech_module import build_content_command
from ghoshell_moss.contracts.speech import Speech
from collections import deque
import time

__all__ = ["CTMLShell", "new_ctml_shell"]


class CTMLShell(MOSShell[PrimeChannel]):
    def __init__(
            self,
            *,
            name: str = "MOSShell",
            description: Optional[str] = None,
            parent_container: IoCContainer | None = None,
            main_channel: PrimeChannel | None = None,
            speech: Optional[Speech] = None,
            logger: LoggerItf | None = None,
            experimental: bool = True,
            primitives: list[str | Command] | None = None,
            meta_instruction: str | None = None,
            refresh_moss_static: bool = True,
            capture_errors_on_exit: bool = False,
    ):
        self._name = name
        self._desc = description
        self._capture_errors_on_exit = capture_errors_on_exit

        self._container = Container(name=name, parent=parent_container)
        self._container.set(MOSShell, self)

        self._primitives: list[str | Command] | None = primitives
        # register primitives
        self._main_channel = main_channel or create_ctml_main_chan(
            experimental=experimental,
            with_default_primitives=primitives is None,
            description=description,
        )
        if primitives:
            for primitive_item in primitives:
                if isinstance(primitive_item, Command):
                    self._main_channel.build.add_command(primitive_item)
                elif isinstance(primitive_item, str):
                    primitive = default_primitive_map.get(primitive_item)
                    if primitive is None:
                        raise ValueError(f"Unknown primitive {primitive_item}")
                    self._main_channel.build.add_command(primitive)

        self._speech: Speech = speech
        self._ctml_meta_instruction = meta_instruction or get_moss_ctml_meta_instruction(CTML_VERSION)
        self._clearing_task: asyncio.Future[None] | None = None

        # cache
        self._do_refresh_moss_static = refresh_moss_static
        self._moss_static_cache: str | None = None
        self._moss_dynamic_cache: list[Message] | None = None
        self._moss_dynamic_refreshed_at: float = 0.0

        self._last_channel_metas: dict[ChannelFullPath, ChannelMeta] | None = None
        self._last_channel_metas_built_at: float = 0.0
        self._last_channel_metas_refreshed_at: float = 0.0
        self._refresh_meta_future: asyncio.Future | None = None

        # logger
        self._logger = logger

        # --- lifecycle --- #
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._exit_stack = contextlib.AsyncExitStack()
        self._paused = False

        self._start: bool = False
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()

        # --- interpreter --- #
        self._interpreter: Optional[Interpreter] = None

        # --- runtime --- #
        self._main_runtime: Optional[ChannelRuntime] = None
        self._log_prefix = "[MOSSShell name=%s] " % self._name

        # --- hook? --- #
        self._wait_any_task: deque[ThreadSafeFuture[CommandTask]] = deque()

    @property
    def container(self) -> IoCContainer:
        return self._container

    def meta_instruction(self) -> str:
        return self._ctml_meta_instruction

    def static_messages(self) -> str:
        if self._moss_static_cache is None:
            self._moss_static_cache = make_static_messages(self.channel_metas(available_only=False))
        return self._moss_static_cache

    # --- introspection (read-only, for debugging / test reflection) --- #

    @property
    def moss_dynamic_refreshed_at(self) -> float:
        """monotonic time of last ``dynamic_messages(available_only=True)`` cache build.

        0.0 if cache 从未建立. 调试 + 单测的反身性入口 — 与 ``stale_time`` 协议配合
        可观察 "本次调用是否复用缓存".
        """
        return self._moss_dynamic_refreshed_at

    @property
    def has_moss_dynamic_cache(self) -> bool:
        """``available_only=True`` 路径下的 dynamic messages 是否已缓存."""
        return self._moss_dynamic_cache is not None

    @property
    def has_moss_static_cache(self) -> bool:
        """``static_messages`` 是否已缓存. (``refresh_metas`` 时按需失效.)"""
        return self._moss_static_cache is not None

    @property
    def channel_metas_built_at(self) -> float:
        """monotonic time of last ``channel_metas`` 本地字典重建.

        与 ``channel_metas_refreshed_at`` 区分: built_at 是本地视图重建,
        refreshed_at 是下层 runtime 的远端刷新. 两层缓存的反身性时间戳.
        """
        return self._last_channel_metas_built_at

    @property
    def channel_metas_refreshed_at(self) -> float:
        """monotonic time of last ``refresh_metas`` 完成时间."""
        return self._last_channel_metas_refreshed_at

    def dynamic_messages(
            self,
            available_only: bool = True,
            *,
            stale_time: float = 0.0,
    ) -> list[Message]:
        if not available_only:
            return make_dynamic_messages(self.channel_metas(available_only=available_only))
        now = time.monotonic()
        if self._moss_dynamic_cache is None or (now - self._moss_dynamic_refreshed_at) > stale_time:
            self._moss_dynamic_cache = make_dynamic_messages(self.channel_metas(available_only=available_only))
            self._moss_dynamic_refreshed_at = now
        return self._moss_dynamic_cache

    def interpreting(self) -> Optional[Interpreter]:
        return self._interpreter

    @property
    def name(self) -> str:
        return self._name

    def topics(self) -> TopicService:
        self._check_running()
        return self._main_runtime.tree.topics

    async def __aenter__(self):
        if self._start:
            raise RuntimeError("Shell is already started")
        self._start = True
        self._event_loop = asyncio.get_running_loop()
        # 进入开机过程.
        await self._exit_stack.__aenter__()
        for ctx_manager in self._bootstrap_stacks():
            # 进入每一个开启状态.
            await self._exit_stack.enter_async_context(ctx_manager())
        self.logger.info("%s shell started", self._log_prefix)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            if isinstance(exc_val, asyncio.CancelledError):
                pass
            else:
                self.logger.exception(exc_val)
        self.logger.info("%s shell is exiting", self._log_prefix)
        await self._exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        self.logger.info("%s exited", self._log_prefix)
        return self._capture_errors_on_exit or None

    def _bootstrap_stacks(self) -> Iterable[Callable[[], contextlib.AbstractAsyncContextManager[None]]]:
        yield self._ioc_context_manager
        yield self._speech_context_manager
        yield self._runtime_context_manager

    @contextlib.asynccontextmanager
    async def _ioc_context_manager(self) -> AsyncGenerator[None, None]:
        await asyncio.to_thread(self._container.bootstrap)

        # 日志准备.
        if self._logger is None:
            logger = self._container.get(LoggerItf)
            if logger is None:
                logger = logging.getLogger("moss")
                self._container.set(LoggerItf, logger)
            self._logger = logger

        try:
            yield
        finally:
            await asyncio.to_thread(self._container.shutdown)

    @contextlib.asynccontextmanager
    async def _speech_context_manager(self):
        """
        启动关闭音频模块.
        """
        if self._speech:
            self._container.set(Speech, self._speech)
        else:
            speech = self._container.get(Speech)
            if speech is None:
                speech = NullSpeech()
                self._container.set(Speech, speech)
            self._speech = speech

        # 注册 __content__ 内核命令（shell 始终拥有说话能力）
        content_cmd = build_content_command(self._speech)
        self.main_channel.build.add_command(content_cmd, override=False)

        await self._speech.start()
        try:
            yield
        finally:
            await self._speech.close()

    @contextlib.asynccontextmanager
    async def _runtime_context_manager(self):
        """
        开启 channel runtime.
        """
        self._main_runtime = self._main_channel.bootstrap(self._container)
        # 开启 Runtime
        await self._main_runtime.start()
        try:
            yield
        finally:
            # 关闭 Runtime. k
            await self._main_runtime.close()

    # --- lifetime functions --- #

    @property
    def runtime(self) -> ChannelRuntime:
        self._check_running()
        return self._main_runtime

    def pause(self, toggle: bool = True, callback: Callable[[], None] | None = None) -> None:
        """急停 — 同步 fire-and-forget. callback 在清理完成时 fire.

        pause(True) 触发 clear(), 通过 ``add_done_callback`` 链式通知.
        pause(False) 恢复, callback 同步调用.
        """
        self._paused = toggle
        if self._paused:
            fut = self.clear()
            if callback:
                fut.add_done_callback(lambda _: callback())
        elif callback:
            callback()

    def is_paused(self) -> bool:
        return self._paused

    @property
    def logger(self) -> LoggerItf:
        if self._logger is None:
            logger = self._container.get(LoggerItf) or get_moss_logger()
            self._logger = logger
        return self._logger

    def is_running(self) -> bool:
        self_running = self._start and not self._closing_event.is_set()
        return self_running and self._main_runtime and self._main_runtime.is_running()

    async def wait_connected(self, *channel_paths: str) -> None:
        if not self.is_running():
            return
        paths = list(channel_paths)
        if len(paths) == 0:
            await self._main_runtime.wait_connected()

        waiting = []
        for path in paths:
            runtime = self._main_runtime.fetch_sub_runtime(path)
            if runtime is None or not runtime.is_running():
                continue
            waiting.append(runtime.wait_connected())
        if len(waiting) > 0:
            _ = await asyncio.gather(*waiting)

    async def wait_until_idle(self, timeout: float | None = None) -> None:
        if not self.is_running():
            return
        if timeout is None:
            await self._main_runtime.wait_idle()
        else:
            await asyncio.wait_for(self._main_runtime.wait_idle(), timeout=timeout)

    def is_closed(self) -> bool:
        return self._closed_event.is_set()

    def _check_running(self):
        if not self.is_running():
            raise RuntimeError(f"Shell `{self._name}` not running")

    def is_idle(self) -> bool:
        return self.is_running() and self._main_runtime.is_idle()

    def _interpreter_callback_task(self, task: CommandTask | None) -> None:
        if task is not None:
            self.push_task(task)

    def _check_paused(self) -> None:
        if self._paused:
            raise PausedError(f"Shell `{self._name}` is paused")

    async def interpreter(
            self,
            kind: InterpreterKind = "clear",
            *,
            refresh_metas: bool = True,
            meta_instruction: str | None = None,
            stream_id: Optional[int] = None,
            config: list[ChannelFullPath] | None = None,
            prepare_timeout: float = 2.0,
            ignore_wrong_command: bool = False,
            token_replacements: dict[str, str] | None = None,
            clear_after_exit: bool | None = None,
            task_context: dict[str, Any] | None = None,
    ) -> Interpreter:
        self._check_running()
        self._check_paused()

        # 方便理解不同类型的处理逻辑. 看待 interpreter 的副作用问题.
        callback = None
        interrupted_interpretation = None
        undone_tasks = None
        if kind == "clear":
            # clear 会先清空.
            await self.clear()
            # 清除当前存在的 interpretation.
            interrupted_interpretation = await self.stop_interpretation()
            callback = self._interpreter_callback_task
        elif kind == "dry_run":
            # dry_run 不会对 shell 产生真实影响, 可以用来做纯解析.
            callback = None
        elif kind == "append":
            # append 会追加命令, 而不是清除.
            callback = self._interpreter_callback_task
            if self._interpreter and self._interpreter.is_running():
                # 停止旧的 interpreter 继续提交新的信息.
                undone_tasks = self._interpreter.incomplete_tasks()
                interrupted_interpretation = await self._interpreter.close(cancel_executing=False)
            self._interpreter = None

        # 阻塞等待刷新结果.
        if refresh_metas:
            await self.refresh_metas(timeout=prepare_timeout)
        config = self.channel_metas(available_only=True, config=config)
        commands = self.commands(available_only=True, config=config)
        interpreter = CTMLInterpreter(
            kind=kind,
            moss_meta_instruction=meta_instruction or self.meta_instruction(),
            interrupted=interrupted_interpretation,
            undone_tasks=undone_tasks,
            commands=commands,
            stream_id=stream_id or unique_id(),
            callback=callback,
            logger=self.logger,
            channel_metas=config,
            ignore_wrong_command=ignore_wrong_command,
            tokens_replacement=token_replacements,
            clear_after_exit=clear_after_exit,
            moss_static=self._moss_static_cache,
            task_context=task_context,
        )

        # 会接受回调的话, 更新最新的 interpreter.
        if callback is not None:
            self._interpreter = interpreter
        return interpreter

    @property
    def main_channel(self) -> PrimeChannel:
        return self._main_channel

    def pub_topic(self, topic: Topic | TopicModel, *, name: str = "") -> None:
        if not self.is_running():
            raise RuntimeError(f"Shell {self._name} not running")
        if isinstance(topic, TopicModel):
            topic = topic.to_topic()
        if not isinstance(topic, Topic):
            raise ValueError(f"Topic {topic} is not Topic or TopicModel type")

        self._main_runtime.tree.topics.pub(topic=topic, name=name, creator=f"shell/{self._name}")

    async def refresh_metas(
            self,
            timeout: float | None = None,
            *,
            stale_time: float | None = None,
    ) -> bool:
        if not self.is_running():
            return False
        now = time.monotonic()
        stale_time = stale_time or 0.0
        # 最后刷新时间可靠, 则不刷新.
        if stale_time > 0.0 and now < (self._last_channel_metas_refreshed_at + stale_time):
            # 如果近期执行过刷新, 则在同样的时间内不反复触发.
            return True
        return await self._refresh_channel_metas(timeout=timeout)

    async def _refresh_channel_metas(self, timeout: float | None) -> bool:
        # 等待运行结束.
        self._last_channel_metas = None
        if self._do_refresh_moss_static:
            self._moss_static_cache = None

        refresh_meta_future = self.runtime.refresh_metas()
        # 等待到结束.
        try:
            done, pending = await asyncio.wait(
                [refresh_meta_future],
                timeout=timeout,
            )
            return refresh_meta_future in done
        finally:
            # 更新最后一次等待完刷新的时间.
            now = time.monotonic()
            if now > self._last_channel_metas_refreshed_at:
                self._last_channel_metas_refreshed_at = now

    def _update_channel_metas(self):
        self._last_channel_metas = self._main_runtime.metas()
        self._last_channel_metas_built_at = time.monotonic()

    def channel_metas(
            self,
            available_only: bool = False,
            config: Optional[list[ChannelFullPath]] = None,
            *,
            stale_time: float | None = None,
    ) -> dict[str, ChannelMeta]:
        if not self.is_running():
            return {}
        stale_time = stale_time or 0.0
        now = time.monotonic()
        if self._last_channel_metas is None or (now - self._last_channel_metas_built_at) > stale_time:
            self._update_channel_metas()

        metas = self._last_channel_metas
        if config:
            result = {}
            for path in config:
                if path in metas:
                    meta = metas[path]
                    if meta.available or not available_only:
                        result[path] = metas[path]
            return result

        if not available_only:
            return metas

        result = {}
        for channel_path, channel_meta in metas.items():
            if channel_meta.available or not available_only:
                result[channel_path] = channel_meta
        return result

    def push_task(self, *tasks: CommandTask) -> None:
        self._check_running()
        self._check_paused()
        for task in tasks:
            # wait any task
            while len(self._wait_any_task) > 0:
                ft = self._wait_any_task.popleft()
                if not ft.done():
                    ft.set_result(task)

        self._main_runtime.push_task(*tasks)

    async def stop_interpretation(self) -> Optional[Interpretation]:
        self._check_running()
        if self._interpreter is not None and self._interpreter.is_running():
            # 考虑线程安全问题. 先简单做一层防御.
            old = self._interpreter
            self._interpreter = None
            stop_task = self._event_loop.create_task(old.close(cancel_executing=True))
            # 防止交叉感染.
            future = asyncio.get_running_loop().create_future()

            def done_callback(_t: asyncio.Task) -> None:
                if not future.done():
                    future.set_result(None)

            stop_task.add_done_callback(done_callback)
            return await future
        return None

    async def wait_until_closed(self) -> None:
        if not self.is_running():
            return
        await self._closed_event.wait()

    async def wait_any_task(self) -> CommandTask:
        ft = ThreadSafeFuture[CommandTask]()
        try:
            self._wait_any_task.append(ft)
            return await ft
        finally:
            if not ft.done():
                ft.cancel()

    def commands(
            self,
            available_only: bool = True,
            *,
            config: dict[ChannelFullPath, ChannelMeta] | None = None,
            exec_in_chan: bool = False,
    ) -> dict[ChannelFullPath, dict[str, Command]]:
        self._check_running()

        commands = self._main_runtime.commands(available_only=available_only)
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
        command_unique_name = Command.make_unique_name(chan, name)
        real_command = self.runtime.get_command(command_unique_name)
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

        # 创建一个入栈函数.
        async def _exec_in_chan_func(*args, **kwargs) -> Any:
            # 检查是不是在 channel 里被运行的.
            _runtime = ChannelCtx.runtime()
            if _runtime is not None:
                # 如果是在 channel 里运行的, 则直接调用其真函数运行结果即可.
                return await origin_func(*args, **kwargs)

            # 并不是在 runtime 里运行的, 检查是否有 task 对象.
            task = ChannelCtx.task()
            if task is not None:
                # 如果上下文里已经有了 task, 则仍然执行结果.
                return await origin_func(*args, **kwargs)
            else:
                # 发送到 runtime 里, 等待 Channel 运行它.
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

    async def _noop(self) -> None:
        return

    def clear(self) -> asyncio.Future[None]:
        if not self.is_running():
            return asyncio.create_task(self._noop())
        if self._clearing_task is not None and not self._clearing_task.done():
            return self._clearing_task
        self._clearing_task = self._event_loop.create_task(self._clear())
        return self._clearing_task

    async def _clear(self):
        done = await asyncio.gather(
            self._speech.clear(),
            self._main_runtime.tree.clear(self._main_runtime),
            self.stop_interpretation(),
            return_exceptions=True,
        )
        for t in done:
            if isinstance(t, Exception):
                self._logger.error("%s clear shell failed: %s", self._log_prefix, str(t))


def new_ctml_shell(
        name: str = "shell",
        description: Optional[str] = None,
        parent_container: IoCContainer | None = None,
        main_channel: Channel | None = None,
        speech: Optional[Speech] = None,
        logger: Optional[LoggerItf] = None,
        experimental: bool = True,
        meta_instruction: str | None = None,
        primitives: list[str | Command] | None = None,
        capture_errors_on_exit: bool = False,
) -> CTMLShell:
    """系统默认提供的 shell"""
    return CTMLShell(
        name=name,
        description=description,
        parent_container=parent_container,
        main_channel=main_channel,
        speech=speech,
        logger=logger,
        experimental=experimental,
        primitives=primitives,
        meta_instruction=meta_instruction,
        capture_errors_on_exit=capture_errors_on_exit,
    )


async def ctml_shell_test(
        *channels: Channel,
        ctml: str,
        builder: Callable[[CTMLShell], None] | None = None,
        main: PrimeChannel | None = None,
        logger: LoggerItf | None = None,
        timeout: float | None = None,
) -> list[CommandTask]:
    """
    simple method to test ctml
    """
    shell = new_ctml_shell(main_channel=main)
    if logger is not None:
        shell.container.set(LoggerItf, logger)
    for channel in channels:
        shell.main_channel.import_channels(channel)
    if builder is not None:
        builder(shell)
    async with shell:
        interpreter = await shell.interpreter(clear_after_exit=True)
        async with interpreter:
            interpreter.feed(ctml)
            interpreter.commit()
            if timeout is not None:
                tasks = await asyncio.wait_for(interpreter.wait_tasks(throw=True), timeout=timeout)
            else:
                tasks = await interpreter.wait_tasks(throw=True)
            return list(tasks.values())
