"""
A Shell built on top of the streaming interpreter, and the encapsulation of the body.
It understands the body, interprets model output, and schedules commands across
parallel Channels (经络, "meridian").
"""

import asyncio
from typing import Callable
import contextlib
from abc import ABC, abstractmethod
from typing import Literal, Optional, AsyncIterable, Generic, TypeVar, Any, Protocol
from ghoshell_container import IoCContainer
from ghoshell_moss.core.concepts.channel import Channel, ChannelFullPath, ChannelMeta, ChannelRuntime
from ghoshell_moss.core.concepts.command import Command, CommandTask, CommandToken
from ghoshell_moss.core.concepts.interpreter import Interpreter, Interpretation
from ghoshell_moss.core.concepts.topic import TopicService
from ghoshell_moss.message import Message

__all__ = [
    "InterpreterKind",
    "MOSShell",
    "Tracer",
]

InterpreterKind = Literal["clear", "append", "dry_run"]

MAIN_CHANNEL = TypeVar("MAIN_CHANNEL", bound=Channel)


class MOSShell(Generic[MAIN_CHANNEL], ABC):
    """
    Model-Operated Operating System Shell.
    A Shell exposed to the model, letting an AI operate the system it inhabits.

    The core technical goal is a duplex runtime that gives a persistent agent
    realtime perception, interaction and control, plus near-unlimited reflexivity.

    Minimal form of the Shell's full-duplex interaction:

    Create a Shell instance.
    >>> def create_shell(...) -> MOSShell:
    >>>     ...

    Give the Shell various Channels; some Channels support an
    install / uninstall / open / close paradigm.

    >>> def build_shell(shell: MOSShell, channels: list[Channel]) -> MOSShell:
    >>>     shell.main_channel.import_channels(*channels)
    >>>     return shell

    This Channels system should contain a complete AIOS paradigm:

    + Instructions: modification of the AI's own instructions module.
    + Memories: the AI's memory system.
    + Mind: thought management and control.
        - Skills: attention mechanisms the AI manages through Skills, to focus on different tasks.
        - TasksManager: the AI's multi-task management; supports tree nesting, switching among
          multiple Tasks, and maintaining an independent context per task.
    + Tools: the various available tools.
        + Desktops: desktop software the AI owns, to operate its host OS.
            - Apps: local applications the AI can manage; each app has its own Runtime.
        - Terminal: command lines the AI can directly operate and modify.
        + Assets: various local resources the AI can manage.
        - Modules: all invocable python modules the AI can manage within its own Runtime.
    + LAN: the various tools available on the local network.
        + HomeAssistant: smart home.
        + AI Assistants: the various AIs it can converse with.
    + Sencors: all invocable perception modules.
    + UserInterfaces: the various interfaces for interacting with humans.
    + Bodies: the various physical bodies it can control.
    """
    # 设计意图（原文）:
    # 这个技术实现的核心目标, 是通过一个双工运行的 Runtime, 为一个持久化智能体提供 Realtime 感知,
    # 交互和控制能力. 以及提供几乎无限的反身性.
    #
    # 然后 Shell 运行可以通过 Topic 来进行通讯, 用 CSP 范式来创建持久运行 Agent 逻辑:
    # 在 Shell 能够持续, 稳定运行的情况下, AI (Ghost) 运行在 Shell 中, 持续地与现实世界交互.

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        pass

    @abstractmethod
    def topics(self) -> TopicService:
        pass

    # --- channels --- #

    @property
    @abstractmethod
    def main_channel(self) -> MAIN_CHANNEL:
        """
        The Shell's own main channel (主轨). The main channel also registers all sub-channels.
        Its name must be the empty string.
        Positioned like python's __main__ module.
        """
        pass

    @abstractmethod
    async def refresh_metas(
            self,
            timeout: float | None = None,
            *,
            stale_time: float | None = None,
    ) -> bool:
        """
        Refresh meta for all channels.
        :param timeout: maximum wait time; returns once exceeded, possibly with some channels unrefreshed.
        :param stale_time: expiry of the most recent refresh. If that refresh is younger than this,
                    returns immediately without refreshing. None disables the comparison.
        """
        pass

    @property
    @abstractmethod
    def runtime(self) -> ChannelRuntime:
        pass

    # --- runtime methods --- #

    @abstractmethod
    def pause(self, toggle: bool = True, callback: Callable[[], None] | None = None) -> None:
        """
        Emergency stop, effective immediately. Forbids new command input until pause is cancelled.

        callback fires when clear completes (done semantics). The caller must ensure thread safety.
        """
        pass

    @abstractmethod
    def is_paused(self) -> bool:
        """
        Whether the shell is paused.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        Whether the shell is running.
        """
        pass

    @abstractmethod
    async def wait_connected(self, *channel_paths: str) -> None:
        """
        Force-wait for the given channels, or all channels, to finish connecting.
        Usually unnecessary; mainly for tests.
        """
        pass

    @abstractmethod
    async def wait_any_task(self) -> CommandTask:
        """wait any task is pushed into the shell and notify with it"""
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """
        Whether the shell has been closed.
        """
        pass

    @abstractmethod
    def is_idle(self) -> bool:
        """
        Whether the shell is idle. Idle means no command is running.
        """
        pass

    @abstractmethod
    async def wait_until_idle(self, timeout: float | None = None) -> None:
        """
        Wait until all commands in the shell have finished.
        """
        # todo: 应该可以指定某个具体的 channel.
        pass

    @abstractmethod
    async def wait_until_closed(self) -> None:
        """
        Block until the Shell is closed.
        """
        pass

    @abstractmethod
    def commands(
            self, available_only: bool = True, *, config: dict[ChannelFullPath, ChannelMeta] | None = None
    ) -> dict[ChannelFullPath, dict[str, Command]]:
        """
        All available commands of the current runtime.
        Note the key is a channel path. E.g. `foo.bar:baz` means the command comes from
        channel `foo.bar` and is named 'baz'.
        """
        pass

    @abstractmethod
    def channel_metas(
            self,
            available_only: bool = False,
            selection: Optional[list[ChannelFullPath]] = None,
            *,
            stale_time: float | None = None,
    ) -> dict[ChannelFullPath, ChannelMeta]:
        """
        Channel meta of the current running state.
        :param available_only: only show runnable channels.
        :param selection: the selected channel full paths.
        :param stale_time: the method reads from cache; caches older than stale time are dropped
                    and rebuilt from all channels.
        """
        pass

    @abstractmethod
    def on_channel_metas_generation(
            self,
            callback: Callable[[dict[ChannelFullPath, ChannelMeta]], None],
    ) -> Callable[[], None]:
        """Register a callback invoked when channel metas finish rebuilding. Returns an unregister handle."""
        ...

    @abstractmethod
    def meta_instruction(self) -> str:
        """
        meta instruction of the MOSS
        """
        pass

    @abstractmethod
    def static_messages(self) -> str:
        """
        instructions of all channels
        """
        pass

    @abstractmethod
    def dynamic_messages(self, available_only: bool = True, *, stale_time: float = 0.0) -> list[Message]:
        """
        context messages of all the channels.
        """
        pass

    @abstractmethod
    async def get_command(self, chan: str, name: str, /, exec_in_chan: bool = False) -> Optional[Command]:
        """
        Get a runnable channel command.
        This syntax reads like `from channel_path import command_name`.

        :param chan: the channel path, e.g. foo.bar
        :param name: command name
        :param exec_in_chan: when True, calling this command like a function still sends a
                    command task into the channel.
        :return: None means the command does not exist.
        """
        pass

    # --- interpret --- #

    @abstractmethod
    def interpreting(self) -> Optional[Interpreter]:
        """The Interpreter currently running."""
        pass

    @contextlib.asynccontextmanager
    async def interpreter_in_ctx(
            self,
            kind: InterpreterKind = "clear",
            *,
            meta_instruction: str | None = None,
            stream_id: Optional[str] = None,
            config: Optional[list[ChannelFullPath]] = None,
            ignore_wrong_command: bool = False,
            clear_after_exit: bool | None = None,
            task_context: dict[str, Any] | None = None,
    ):
        """
        A small bit of syntactic sugar.
        """
        interpreter = await self.interpreter(
            kind=kind,
            meta_instruction=meta_instruction,
            stream_id=stream_id,
            config=config,
            ignore_wrong_command=ignore_wrong_command,
            clear_after_exit=clear_after_exit,
            task_context=task_context,
        )
        async with interpreter:
            yield interpreter

    # token_replacements 费用推导（原文）:
    #             假设用 n 个代理 token, 平均每个代理 token 消耗是 m, 代理掉 v 个token, 在 t 次多轮对话中平均使用了 k 个代理 token.
    #             t 轮 instruction 多消耗的 token: n * m * t
    #             t 轮输出实际减少的 tokens:  (v - m) * k * t
    #             所以 (v - m) * k * 3 > n * m    就有正收益.
    #             假设 m = 1, v = 10, k=3, n=20,  每轮多消耗 20 个点,  每轮减少 80 个点开销. 大意如此.
    @abstractmethod
    async def interpreter(
            self,
            kind: InterpreterKind = "clear",
            *,
            refresh_metas: bool = True,
            stream_id: Optional[str] = None,
            config: Optional[list[ChannelFullPath]] = None,
            prepare_timeout: float = 2.0,
            ignore_wrong_command: bool = False,
            token_replacements: dict[str, str] | None = None,
            meta_instruction: str | None = None,
            clear_after_exit: bool | None = None,
            task_context: dict[str, Any] | None = None,
    ) -> Interpreter:
        """
        Create an interpreter for interpretation.
        :param kind: preamble behavior when creating the Interpreter:
                    clear   - clear all running commands first.
                    append  - append commands instead of clearing. Stops the previous interpreter
                              from submitting new input, while its already-executing tasks keep running.
                    dry_run - the interpreter still executes normally, but does NOT push the
                              generated command tasks to the shell (pure parsing).

        :param stream_id: set an explicit stream id. Every command token generated during the
                    interpreter's whole lifecycle is tagged with it.

        :param config: when dynamic channel metas are passed, the commands available at runtime
                    become the intersection of the real commands and the channel metas passed here.
                    This is a way to dynamically change runtime capabilities.

        :param prepare_timeout: time allowed for the preparation phase. If it is not done in time,
                    the interpreter is returned directly.

        :param ignore_wrong_command: do not raise a parse error on hallucinated commands.
        :param refresh_metas: refresh the shell before running.

        :param token_replacements: replace part of the tokens obtained from the interpreter feed by
                    key, substituting the value. This swaps tokens in the instruction for tokens in
                    the output; response speed and cost can be tuned.

        :param clear_after_exit: clear undone tasks after exit.
        :param meta_instruction: can replace the system default moss syntax prompt. Usually only
                    modified when debugging.
        :param task_context: copy the task context into every command task.
        """
        pass

    async def parse_text_to_command_tokens(
            self,
            text: str | AsyncIterable[str],
    ) -> AsyncIterable[CommandToken]:
        """
        Syntactic sugar demonstrating how to turn text into command tokens.
        """
        interpreter = await self.interpreter("dry_run")
        if isinstance(text, str):

            async def generate():
                yield text

            text_stream = generate()
        else:
            text_stream = text
        async for token in interpreter.aparse_text_to_command_tokens(text_stream):
            if token is None:
                break
            yield token

    async def parse_tokens_to_command_tasks(
            self,
            tokens: AsyncIterable[CommandToken],
            *,
            ignore_wrong_command: bool = False,
    ) -> AsyncIterable[CommandTask]:
        """
        Syntactic sugar demonstrating how to turn command tokens into command tasks.
        """
        _token_queue = asyncio.Queue[CommandToken | None]()
        _task_queue = asyncio.Queue[CommandTask | None | Exception]()
        interpreter = await self.interpreter("dry_run", ignore_wrong_command=ignore_wrong_command)

        async def sender():
            try:
                async for token in tokens:
                    await _token_queue.put(token)
                    await asyncio.sleep(0.0)
            except Exception as e:
                raise e
            finally:
                _token_queue.put_nowait(None)

        sender_task = asyncio.create_task(sender())
        consumer_task = asyncio.create_task(
            interpreter.parse_tokens_to_command_tasks(_token_queue, _task_queue.put_nowait),
        )
        try:
            while True:
                item = await _task_queue.get()
                if item is None:
                    break
                yield item
                await asyncio.sleep(0.0)
            await consumer_task
        finally:
            if not sender_task.done():
                sender_task.cancel()
                try:
                    await sender_task
                except asyncio.CancelledError:
                    pass
            if not consumer_task.done():
                consumer_task.cancel()
                try:
                    await consumer_task
                except asyncio.CancelledError:
                    pass

    async def parse_text_to_tasks(
            self,
            text: str | AsyncIterable[str] | list[str],
            *,
            ignore_wrong_command: bool = False,
    ) -> AsyncIterable[CommandTask]:
        """
        Syntactic sugar demonstrating how to turn text directly into command tasks.
        """

        async def generate_text():
            if isinstance(text, str):
                yield text
                return
            elif isinstance(text, list):
                for content in text:
                    yield content
                return
            else:
                async for content in text:
                    yield content

        tokens = self.parse_text_to_command_tokens(generate_text())
        async for task in self.parse_tokens_to_command_tasks(tokens, ignore_wrong_command=ignore_wrong_command):
            yield task

    # --- runtime methods --- #

    @abstractmethod
    def push_task(self, *tasks: CommandTask) -> None:
        """
        Add tasks to the runtime. These tasks block in the Channel Runtime queue until they get
        a chance to execute.
        """
        pass

    @abstractmethod
    async def stop_interpretation(self) -> Optional[Interpretation]:
        """
        Interrupt the running interpretation.
        """
        # 临时实现的中断方法. 原理设计有问题.
        # todo: 重新设计 shell 的中断逻辑.
        pass

    @abstractmethod
    def clear(self) -> asyncio.Future[None]:
        """
        Clear all commands.
        Note clear broadcasts in a tree: clearing a parent channel also clears all its sub-channels.
        """
        pass

    async def start(self) -> None:
        """
        Start the Shell's runtime.
        """
        await self.__aenter__()

    async def close(self) -> None:
        """
        Stop the shell.
        """
        await self.__aexit__(None, None, None)

    @abstractmethod
    async def __aenter__(self):
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def add_tracer(self, tracer: 'Tracer') -> None:
        """Register a tracer to observe the shell lifecycle.

        Fire-and-forget semantics: on every event the shell iterates tracers and checks
        is_closed / is_running to decide whether to fire. There is no active unsubscribe API;
        the shell skips any tracer that reports is_closed() == True.
        """
        ...


class Tracer(Protocol):
    """An observer of the shell runtime. The shell calls it back at key lifecycle points.

    Fire-and-forget: while iterating tracers the shell skips any with is_closed() or
    not is_running(); exceptions are caught by the shell and logged, never affecting
    the main flow.

    Implementation notes:
    - All on_xxx methods must be thread-safe (they may be called from the shell thread or a channel thread).
    - Keep method bodies lightweight; do not block the shell's main flow.
    - is_closed() == True is terminal: the tracer is finished and the shell stops firing at it.
    """

    def is_running(self) -> bool:
        """Whether the tracer is actively receiving. When False the shell skips this fire (useful for pausing)."""
        ...

    def is_closed(self) -> bool:
        """Whether the tracer is closed. When True the shell skips it forever; it may be GC'd later."""
        ...

    def on_task_pushed(self, task: CommandTask) -> None:
        """Called back when a command task is pushed into the shell."""
        ...

    def on_task_done(self, task: CommandTask) -> None:
        """Called back when a command task finishes (success / failure / cancellation all count as done)."""
        ...

    def on_interpreter_stopped(self, interpreter: Interpreter) -> None:
        """Called back when an interpreter has finished closing.

        The compile-time exception (INTERPRET_ERROR) is available via ``interpreter.exception()``,
        and the final Interpretation snapshot via ``interpreter.interpretation()``.
        """
        ...
#
