import asyncio
import contextlib
from abc import ABC, abstractmethod
from collections.abc import AsyncIterable
from typing import Literal, Optional

from ghoshell_container import IoCContainer

from ghoshell_moss.core.concepts.channel import Channel, ChannelFullPath, ChannelMeta, MutableChannel, ChannelRuntime
from ghoshell_moss.core.concepts.command import Command, CommandTask, CommandToken
from ghoshell_moss.core.concepts.states import StateStore
from ghoshell_moss.core.concepts.interpreter import Interpreter
from ghoshell_moss.core.concepts.speech import Speech

__all__ = [
    "InterpreterKind",
    "MOSSShell",
]

InterpreterKind = Literal["clear", "append", "dry_run"]


class MOSSShell(ABC):
    """
    Model-Operated System Shell
    面向模型提供的 Shell, 让 AI 可以操作自身所处的系统.

    Shell 自身也可以作为 Channel 向上提供, 而自己维护一个完整的运行时. 这需要上一层下发的实际上是 command tokens.
    这样才能实现本地 shell 的流式处理.
    """

    _container: IoCContainer

    # todo: 干掉 speech 抽象, 或者用更好的方式解决它.
    speech: Speech

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        pass

    @property
    @abstractmethod
    def states(self) -> StateStore:
        pass

    @abstractmethod
    def with_speech(self, speech: Speech) -> None:
        """
        注册 Speech 对象.
        todo: 准备彻底重构这个实现.
        """
        pass

    # --- channels --- #

    @property
    @abstractmethod
    def main_channel(self) -> MutableChannel:
        """
        Shell 自身的主轨. 主轨同时可以用来注册所有的子轨.
        主轨的名称必须是空字符串.
        定位类似于 python 的 __main__ 模块.
        """
        pass

    @property
    @abstractmethod
    def runtime(self) -> ChannelRuntime:
        pass

    # --- runtime methods --- #

    @abstractmethod
    def is_running(self) -> bool:
        """
        shell 是否在运行中.
        """
        pass

    @abstractmethod
    async def wait_connected(self, *channel_paths: str) -> None:
        """
        强行等待指定的轨道, 或者所有的轨道完成连接.
        通常并不是必要的. 只是为了测试.
        """
        pass

    @abstractmethod
    def is_closed(self) -> bool:
        """
        是否已经关闭运行.
        """
        pass

    @abstractmethod
    def is_idle(self) -> bool:
        """
        是否在闲置状态. 闲置状态指的是没有任何 command 在运行.
        """
        pass

    @abstractmethod
    async def wait_until_idle(self, timeout: float | None = None) -> None:
        """
        等待到 shell 所有的 command 运行结束.
        todo: 应该可以指定某个具体的 channel.
        """
        pass

    @abstractmethod
    async def wait_until_closed(self) -> None:
        """
        阻塞等到 Shell 被关闭.
        """
        pass

    @abstractmethod
    def commands(
        self, available_only: bool = True, *, config: dict[ChannelFullPath, ChannelMeta] | None = None
    ) -> dict[ChannelFullPath, dict[str, Command]]:
        """
        当前运行时所有的可用的命令.
        注意, key 是 channel path. 例如 foo.bar:baz 表示 command 来自 channel `foo.bar`, 名称是 'baz'
        """
        pass

    @abstractmethod
    def channel_metas(
        self,
        available: bool = True,
    ) -> dict[ChannelFullPath, ChannelMeta]:
        """
        当前运行状态中的 Channel meta 信息.
        key 是 channel path, 例如 foo.bar
        如果为 '', 表示为主 channel.
        """
        pass

    @abstractmethod
    async def get_command(self, chan: str, name: str, /, exec_in_chan: bool = False) -> Optional[Command]:
        """
        获取一个可以运行的 channel command.
        这个语法可以理解为 from channel_path import command_name

        :param chan: channel 的 path, 例如 foo.bar
        :param name: command name
        :param exec_in_chan: 表示这个 command 在像函数一样调用时, 仍然会发送 command task 到 channel 中.
        :return: None 表示命令不存在.
        """
        pass

    # --- interpret --- #

    @contextlib.asynccontextmanager
    async def interpreter_in_ctx(
        self,
        kind: InterpreterKind = "clear",
        *,
        stream_id: Optional[str] = None,
        config: Optional[dict[ChannelFullPath, ChannelMeta]] = None,
    ) -> Interpreter:
        interpreter = await self.interpreter(kind=kind, stream_id=stream_id, config=config)
        async with interpreter:
            yield interpreter

    @abstractmethod
    async def interpreter(
        self,
        kind: InterpreterKind = "clear",
        *,
        stream_id: Optional[str] = None,
        config: Optional[dict[ChannelFullPath, ChannelMeta]] = None,
        prepare_timeout: float = 2.0,
    ) -> Interpreter:
        """
        实例化一个 interpreter 用来做解释.
        :param kind: 实例化 Interpreter 时的前置行为:
                     clear 表示清空所有运行中命令.
                     defer_clear 表示延迟清空, 但一旦有新命令, 就会被清空.
                     run 表示正常运行.
                     dry_run 表示 interpreter 虽然会正常执行, 但不会把生成的 command task 推送给 shell.
        :param stream_id: 设置一个指定的 stream id,
                          interpreter 整个运行周期生成的 command token 都会用它做标记.
        :param config: 如果传入了动态的 channel metas,
                              则运行时可用的命令由真实命令和这里传入的 channel metas 取交集.
                              是一种动态修改运行时能力的办法.
        :param prepare_timeout: 准备过度阶段允许的时间.
        """
        pass

    async def parse_text_to_command_tokens(
        self,
        text: str | AsyncIterable[str],
        kind: InterpreterKind = "dry_run",
    ) -> AsyncIterable[CommandToken]:
        """
        语法糖, 用来展示如何把文本生成 command tokens.
        """
        from ghoshell_moss.core.helpers.stream import create_thread_safe_stream

        sender, receiver = create_thread_safe_stream()

        async def _parse_token():
            with sender:
                async with self.interpreter_in_ctx(kind) as interpreter:
                    interpreter.parser().with_callback(sender.append)
                    if isinstance(text, str):
                        interpreter.feed(text)
                    else:
                        async for delta in text:
                            interpreter.feed(delta)
                    await interpreter.wait_parse_done()

        t = asyncio.create_task(_parse_token())
        async for token in receiver:
            if token is None:
                break
            yield token
        await t

    async def parse_tokens_to_command_tasks(
        self,
        tokens: AsyncIterable[CommandToken],
        kind: InterpreterKind = "dry_run",
    ) -> AsyncIterable[CommandTask]:
        """
        语法糖, 用来展示如何将 command tokens 生成 command tasks.
        """
        from ghoshell_moss.core.helpers.stream import create_thread_safe_stream

        sender, receiver = create_thread_safe_stream()

        async def _parse_task():
            with sender:
                async with self.interpreter_in_ctx(kind) as interpreter:
                    interpreter.with_callback(sender.append)
                    async for token in tokens:
                        interpreter.root_task_element().on_token(token)
                    await interpreter.wait_parse_done()

        t = asyncio.create_task(_parse_task())
        async for task in receiver:
            if task is None:
                break
            yield task
        await t

    async def parse_text_to_tasks(
        self,
        text: str | AsyncIterable[str],
        kind: InterpreterKind = "dry_run",
    ) -> AsyncIterable[CommandTask]:
        """
        语法糖, 用来展示如何将 text 直接生成 command tasks (不执行).
        """
        from ghoshell_moss.core.helpers.stream import create_thread_safe_stream

        sender, receiver = create_thread_safe_stream()

        async def _parse_task():
            with sender:
                async with self.interpreter_in_ctx(kind) as interpreter:
                    interpreter.with_callback(sender.append)
                    if isinstance(text, str):
                        interpreter.feed(text)
                    else:
                        async for delta in text:
                            interpreter.feed(delta)
                    await interpreter.wait_parse_done()

        t = asyncio.create_task(_parse_task())
        async for task in receiver:
            if task is None:
                break
            yield task
        await t

    # --- runtime methods --- #

    @abstractmethod
    def push_task(self, *tasks: CommandTask) -> None:
        """
        添加 task 到运行时. 这些 task 会阻塞在 Channel Runtime 队列中直到获取执行机会.
        """
        pass

    @abstractmethod
    async def stop_interpretation(self) -> None:
        """
        临时实现的中断方法. 原理设计有问题.
        todo: 重新设计 shell 的中断逻辑.
        """
        pass

    @abstractmethod
    async def clear(self) -> None:
        """
        清空所有的命令.
        注意 clear 是树形广播的, clear 一个 父 channel 也会 clear 所有的子 channel.
        """
        pass

    async def start(self) -> None:
        """
        启动 Shell 的 runtime.
        """
        await self.__aenter__()

    async def close(self) -> None:
        """
        shell 停止运行.
        """
        await self.__aexit__(None, None, None)

    @abstractmethod
    async def __aenter__(self):
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
