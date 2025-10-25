import contextlib
from abc import ABC, abstractmethod
from typing import List, Dict, Literal, Optional, AsyncIterable
from ghoshell_moss.concepts.channel import Channel, ChannelMeta, ChannelFullPath
from ghoshell_moss.concepts.interpreter import Interpreter
from ghoshell_moss.concepts.command import Command, CommandTask, CommandToken
from ghoshell_container import IoCContainer
import asyncio

__all__ = [
    'Speech', 'SpeechStream',
    'InterpreterKind',
    'MOSSShell',
]


class SpeechStream(ABC):
    """
    shell 发送文本的专用模块. 本身是非阻塞的.
    todo: 考虑把 OutputStream 通用成 Command.
    """
    id: str
    """所有文本片段都有独立的全局唯一id, 通常是 command_token.part_id"""

    cmd_task: Optional[CommandTask] = None
    """stream 生成的 command task"""

    committed: bool = False
    """是否完成了这个 stream 的提交. """

    def buffer(self, text: str, *, complete: bool = False) -> None:
        """
        添加文本片段到输出流里.
        由于文本可以通过 tts 生成语音, 而 tts 有独立的耗时, 所以通常一边解析 command token 一边 buffer 到 tts 中.
        而音频允许播放的时间则会靠后, 必须等上一段完成后才能开始播放下一段.

        :param text: 文本片段
        :type complete: 输出流是否已经结束.
        """
        if self.committed:
            # 不 buffer.
            return
        if text:
            # 文本不为空.
            self._buffer(text)
            if self.cmd_task is not None:
                # buffer 到 cmd task
                self.cmd_task.tokens = self.buffered()
        if complete:
            # 提交.
            self.commit()

    @abstractmethod
    def _buffer(self, text: str) -> None:
        """
        真实的 buffer 逻辑,
        """
        pass

    def commit(self) -> None:
        if self.committed:
            return
        self.committed = True
        self._commit()

    @abstractmethod
    def _commit(self) -> None:
        """真实的结束 stream 讯号. 如果 stream 通过 tts 实现, 这个讯号会通知 tts 完成输出. """
        pass

    def as_command_task(self, commit: bool = False) -> Optional[CommandTask]:
        """
        将 speech stream 转化为一个 command task, 使之可以发送到 Shell 中阻塞.
        """
        from ghoshell_moss.concepts.command import BaseCommandTask, CommandMeta, CommandWrapper
        if self.cmd_task is not None:
            return self.cmd_task

        if commit:
            # 是否要标记提交. stream 可能在生成 task 的时候, 还没有完成内容的提交.
            self.commit()

        async def _speech_lifecycle() -> None:
            try:
                # 标记开始播放.
                self.start()
                # 等待输入结束, 播放结束.
                await self.wait()
            finally:
                # 关闭播放.
                self.close()

        meta = CommandMeta(
            name="__speech__",
            # 默认主轨运行.
            chan="",
        )

        command = CommandWrapper(meta, _speech_lifecycle)
        task = BaseCommandTask.from_command(
            command,
        )
        task.cid = self.id
        # 添加默认的 tokens.
        task.tokens = self.buffered()
        self.cmd_task = task
        return task

    @abstractmethod
    def buffered(self) -> str:
        """
        返回已经缓冲的文本内容, 可能经过了加工.
        """
        pass

    @abstractmethod
    async def wait(self) -> None:
        """
        阻塞等待到播放完成. start & commit 是两个必要的开关.
        commit 意味着文本片段生成完毕.
        start 意味着允许开始播放.
        """
        pass

    @abstractmethod
    def start(self) -> None:
        """
        start to output
        """
        pass

    @abstractmethod
    def close(self):
        """
        关闭一个 Stream.
        """
        pass


class Speech(ABC):
    """
    文本输出模块. 通常和语音输出模块结合.
    """

    @abstractmethod
    def new_stream(self, *, batch_id: Optional[str] = None) -> SpeechStream:
        """
        创建一个新的输出流, 第一个 stream 应该设置为 play
        """
        pass

    @abstractmethod
    def outputted(self) -> List[str]:
        """
        清空之前生成的文本片段, speech 必须能感知到所有输出.
        """
        pass

    @abstractmethod
    def clear(self) -> List[str]:
        """
        清空所有输出中的 output
        """
        pass


InterpreterKind = Literal['clear', 'defer_clear', 'run', 'dry_run']


class MOSSShell(ABC):
    """
    Model-Operated System Shell
    面向模型提供的 Shell, 让 AI 可以操作自身所处的系统.

    Shell 自身也可以作为 Channel 向上提供, 而自己维护一个完整的运行时. 这需要上一层下发的实际上是 command tokens.
    这样才能实现本地 shell 的流式处理.
    """

    container: IoCContainer
    speech: Speech

    @abstractmethod
    def with_speech(self, speech: Speech) -> None:
        """
        注册 Output 对象.
        """
        pass

    # --- channels --- #

    @property
    @abstractmethod
    def main_channel(self) -> Channel:
        """
        Shell 自身的主轨. 主轨同时可以用来注册所有的子轨.
        主轨的名称必须是空字符串.
        定位类似于 python 的 __main__ 模块.
        """
        pass

    # --- runtime methods --- #
    @abstractmethod
    def channels(self) -> Dict[str, Channel]:
        """
        返回当前上下文里的所有 channels.
        只有启动后可以获取.

        其中以 "" 为 key 的就是 main channel
        其它的 channel 以路径为 key. 比如:
        robot/
        ├── body/
        └── head/

        最终生成的 channels:
        - "": main channel
        - robot: 机器人的主 channel
        - robot.body: body channel
        - robot.head: head channel
        """
        pass

    @abstractmethod
    def system_prompt(self) -> str:
        """
        如何使用 MOSShell 的系统指令.
        """
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        shell 是否在运行中.
        """
        pass

    @abstractmethod
    def is_idle(self) -> bool:
        """
        是否在闲置状态.
        """
        pass

    @abstractmethod
    async def wait_until_idle(self, timeout: float | None = None) -> None:
        """
        等待到 shell 运行结束.
        """
        pass

    @abstractmethod
    async def wait_until_closed(self) -> None:
        """
        阻塞等到运行结束.
        """
        pass

    @abstractmethod
    async def commands(
            self,
            available_only: bool = True,
            /,
            config: Dict[ChannelFullPath, Channel] | None = None
    ) -> Dict[ChannelFullPath, Dict[str, Command]]:
        """
        当前运行时所有的可用的命令.
        注意, key 是 channel path. 例如 foo.bar:baz 表示 command 来自 channel `foo.bar`, 名称是 'baz'
        """
        pass

    @abstractmethod
    async def channel_metas(
            self,
            available: bool = True,
            /,
            config: Dict[ChannelFullPath, Channel] | None = None
    ) -> Dict[ChannelFullPath, ChannelMeta]:
        """
        当前运行状态中的 Channel meta 信息.
        key 是 channel path, 例如 foo.bar
        如果为空, 表示为主 channel.
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
            channel_metas: Optional[Dict[ChannelFullPath, ChannelMeta]] = None,
    ) -> Interpreter:
        interpreter = await self.interpreter(kind=kind, stream_id=stream_id, channel_metas=channel_metas)
        async with interpreter:
            yield interpreter

    @abstractmethod
    async def interpreter(
            self,
            kind: InterpreterKind = "clear",
            *,
            stream_id: Optional[str] = None,
            channel_metas: Optional[Dict[ChannelFullPath, ChannelMeta]] = None,
    ) -> Interpreter:
        """
        实例化一个 interpreter 用来做解释.
        :param kind: 实例化 Interpreter 时的前置行为:
                     clear 表示清空所有运行中命令.
                     defer_clear 表示延迟清空, 但一旦有新命令, 就会被清空.
                     run 表示正常运行.
                     dry_run 表示 interpreter 虽然会正常执行, 但不会把生成的 command task 推送给 shell.
        :param stream_id: 设置一个指定的 stream id, interpreter 整个运行周期生成的 command token 都会用它做标记.
        :param channel_metas: 如果传入了动态的 channel metas, 则运行时可用的命令由真实命令和这里传入的 channel metas 取交集.
                              是一种动态修改运行时能力的办法.
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
        from ghoshell_moss.helpers.stream import create_thread_safe_stream
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
        from ghoshell_moss.helpers.stream import create_thread_safe_stream
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
        from ghoshell_moss.helpers.stream import create_thread_safe_stream
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
    def add_task(self, *tasks: CommandTask) -> None:
        """
        添加 task 到运行时. 这些 task 会阻塞在 Channel Runtime 队列中直到获取执行机会.
        """
        pass

    @abstractmethod
    async def clear(self, *chans: str) -> None:
        """
        清空指定的 channel. 如果 chans 为空, 则清空所有的 channel.
        注意 clear 是树形广播的, clear 一个 父 channel 也会 clear 所有的子 channel.
        """
        pass

    @abstractmethod
    async def defer_clear(self, *chans: str) -> None:
        """
        标记 channel 在得到新命令的时候, 先清空.
        如果 chans 为空, 则得到任何命令会清空所有管道.
        """
        pass

    # --- lifecycle --- #

    @abstractmethod
    async def start(self) -> None:
        """
        启动 Shell 的 runtime.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        shell 停止运行.
        """
        pass

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return None
