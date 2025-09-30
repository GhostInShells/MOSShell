from abc import ABC, abstractmethod
from typing import List, Iterable, Dict, Literal, Optional, AsyncIterable
from typing_extensions import Self
from ghoshell_moss.concepts.channel import Channel, ChannelMeta, Client
from ghoshell_moss.concepts.interpreter import Interpreter
from ghoshell_moss.concepts.command import Command, CommandTask, CommandToken
from ghoshell_container import IoCContainer
import asyncio


class OutputStream(ABC):
    """
    shell 发送文本的专用模块. 本身是非阻塞的.
    todo: 考虑把 OutputStream 通用成 Command.
    """
    id: str
    """所有文本片段都有独立的全局唯一id, 通常是 command_part_id"""

    @abstractmethod
    def buffer(self, text: str, *, complete: bool = False) -> None:
        """
        添加文本片段到输出流里.
        由于文本可以通过 tts 生成语音, 而 tts 有独立的耗时, 所以通常一边解析 command token 一边 buffer 到 tts 中.
        而音频允许播放的时间则会靠后, 必须等上一段完成后才能开始播放下一段.

        :param text: 文本片段
        :type complete: 输出流是否已经结束.
        """
        pass

    def commit(self) -> None:
        self.buffer("", complete=True)

    @abstractmethod
    def start(self) -> None:
        """
        允许文本片段开始播放. 这时可能文本片段本身都未生成完, 如果是流式的 tts, 则可以一边 buffer, 一边 tts, 一边播放. 三者并行.
        """
        pass

    @abstractmethod
    def wait_sync(self, timeout: float | None = None) -> bool:
        """
        阻塞等待到文本输出完毕. 当文本输出是一个独立的模块时, 需要依赖这个函数实现阻塞.
        """
        pass

    @abstractmethod
    def is_done(self) -> bool:
        pass

    @abstractmethod
    def as_command_task(self, commit: bool = False) -> Optional[CommandTask]:
        """
        将 wait done 转化为一个 command task.
        这个 command task 通常在主轨 (channel name == "") 中运行.
        """
        pass

    @abstractmethod
    def close(self):
        """
        关闭一个 Stream.
        """
        pass

    # --- asyncio --- #

    @abstractmethod
    async def astart(self) -> None:
        pass

    @abstractmethod
    async def wait(self) -> bool:
        pass

    @abstractmethod
    async def aclose(self) -> None:
        pass


class Output(ABC):
    """
    文本输出模块. 通常和语音输出模块结合.
    """

    @abstractmethod
    def new_stream(self, *, batch_id: Optional[str] = None) -> OutputStream:
        """
        创建一个新的输出流, 第一个 stream 应该设置为 play
        """
        pass

    @abstractmethod
    def outputted(self) -> List[str]:
        pass

    @abstractmethod
    def clear(self) -> List[str]:
        """
        清空所有输出中的 output
        """
        pass


NewInterpreterKind = Literal['clear', 'defer_clear', 'dry_run']


class MOSSShell(ABC):
    """
    Model-Operated System Shell
    面向模型提供的 Shell, 让 AI 可以操作自身所处的系统.
    """

    container: IoCContainer

    @abstractmethod
    def with_output(self, output: Output) -> None:
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
        """
        pass

    @abstractmethod
    def channels(self) -> Dict[str, Channel]:
        pass

    @abstractmethod
    def register(self, *channels: Channel, parent: str = "") -> None:
        """
        注册 channel.
        """
        pass

    @abstractmethod
    def configure(self, *metas: ChannelMeta) -> None:
        """
        配置 channel meta, 运行时会以配置的 channel meta 为准.
        """
        pass

    # --- runtime --- #

    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """
        shell 是否在运行中.
        """
        pass

    @abstractmethod
    def is_idle(self) -> bool:
        pass

    @abstractmethod
    async def wait_until_idle(self, timeout: float | None = None) -> None:
        """
        等待到 shell 运行结束.
        """
        pass

    @abstractmethod
    async def channel_metas(self) -> Dict[str, ChannelMeta]:
        """
        返回所有的 Channel Meta 信息.
        可以被 configure_channel_metas 修改.
        """
        pass

    @abstractmethod
    def commands(self, available: bool = True) -> Dict[str, Command]:
        """
        当前运行时所有的可用的命令.
        """
        pass

    # --- interpret --- #

    @abstractmethod
    async def interpret(
            self,
            kind: NewInterpreterKind = "clear",
            *,
            stream_id: Optional[str] = None,
    ) -> Interpreter:
        """
        实例化一个 interpreter 用来做解释.
        """
        pass

    async def parse_tokens(
            self,
            text: str | AsyncIterable[str],
            kind: NewInterpreterKind = "dry_run",
    ) -> AsyncIterable[CommandToken]:
        from ghoshell_moss.helpers.stream import create_thread_safe_stream
        sender, receiver = create_thread_safe_stream()

        async def _parse_token():
            with sender:
                interpreter = await self.interpret(kind)
                interpreter.parser().with_callback(sender.append)
                async with interpreter:
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

    async def parse_tasks(
            self,
            text: str | AsyncIterable[str],
            kind: NewInterpreterKind = "dry_run",
    ) -> AsyncIterable[CommandTask]:
        from ghoshell_moss.helpers.stream import create_thread_safe_stream
        sender, receiver = create_thread_safe_stream()

        async def _parse_task():
            with sender:
                interpreter = await self.interpret(kind)
                interpreter.with_callback(sender.append)
                async with interpreter:
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
    async def append(self, *tasks: CommandTask) -> None:
        """
        添加 task 到运行时.
        """
        pass

    @abstractmethod
    def clear(self, *chans: str) -> None:
        """
        清空指定的 channel. 如果
        """
        pass

    @abstractmethod
    async def defer_clear(self, *chans: str) -> None:
        pass

    # --- lifecycle --- #

    @abstractmethod
    async def start(self) -> None:
        """
        runtime 启动.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """
        runtime 停止运行.
        """
        pass

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return self
