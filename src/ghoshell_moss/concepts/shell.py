from abc import ABC, abstractmethod
from typing import List, Iterable, Dict, Literal, Optional
from typing_extensions import Self
from .channel import Channel
from .interpreter import Interpreter, SyncInterpreter
from .command import BaseCommandTask, Command, CommandTaskSeq, CommandTask
from ghoshell_container import IoCContainer
from contextlib import asynccontextmanager


class OutputStream(ABC):
    """
    shell 发送文本的专用模块.
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
    def output_start(self) -> None:
        """
        允许文本片段开始播放. 这时可能文本片段本身都未生成完, 如果是流式的 tts, 则可以一边 buffer, 一边 tts, 一边播放. 三者并行.
        """
        pass

    @abstractmethod
    def wait_done(self, timeout: float | None = None) -> None:
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
    def clear(self) -> List[str]:
        """
        清空所有输出中的 output
        """
        pass


class Controller(ABC):

    @abstractmethod
    def system_prompt(self) -> str:
        """
        控制模块的默认的 prompt.
        """
        pass

    @abstractmethod
    async def loop(self, times: int, ct_: str) -> CommandTaskSeq:
        """
        解析 __text__ 里的 Command Token 语法, 返回一个 Command Task Seq
        通常这个 Seq 由 解析出来的 CommandTask + 下一轮 Loop 构成. N + 1 个 Task.
        """
        pass

    @abstractmethod
    async def group(self, __text__: str) -> CommandTaskSeq:
        """
        所有的 __text__ 解析完毕后, 一次性输出.
        """
        pass

    @abstractmethod
    async def clear(self, __text__: str = "") -> None:
        """
        清空指定的 channel. 不包括自己. 如果 text 为空的话, 清空所有的子孙 channel.
        """
        pass

    @abstractmethod
    async def wait_for(self, __text__: str, timeout: float | None = None) -> None:
        pass

    @abstractmethod
    async def wait_done(self, timeout: float | None = None) -> None:
        pass


class ChannelRuntime(ABC):
    """
    管理 channel 的所有通讯状态.
    """

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    async def append(self, *tasks: CommandTask) -> None:
        pass

    @abstractmethod
    async def clear(self) -> None:
        pass

    @abstractmethod
    async def defer_clear(self) -> None:
        pass

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def is_busy(self) -> bool:
        pass

    @abstractmethod
    async def wait_idle(self, timeout: float | None = None) -> bool:
        pass

    @abstractmethod
    def get_child(self, name: str) -> "ChannelRuntime":
        pass

    @abstractmethod
    def bootstrap(self) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass

    @abstractmethod
    def join(self) -> None:
        pass


class ShellRuntime(ABC):

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def is_idle(self) -> bool:
        pass

    @abstractmethod
    def interpret(self, kind: Literal['clear', 'defer_clear', 'try'] = "clear") -> None:
        pass

    @abstractmethod
    async def append(self, *commands: BaseCommandTask) -> None:
        pass

    @abstractmethod
    async def clear(self, *chans: str) -> None:
        pass

    @abstractmethod
    async def defer_clear(self, *chans: str) -> None:
        pass

    @abstractmethod
    async def system_prompt(self) -> str:
        pass

    @abstractmethod
    async def commands(self) -> Dict[str, List[Command]]:
        """
        get commands from shell
        """
        pass

    @abstractmethod
    async def wait_until_idle(self, timeout: float | None = None) -> None:
        pass

    @abstractmethod
    async def start(self) -> None:
        pass

    @abstractmethod
    async def stop(self) -> None:
        pass

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()


class MOSSShell(ABC):
    """
    Model-Operated System Shell
    面向模型提供的 Shell, 让 AI 可以操作自身所处的系统.
    """

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    # --- properties --- #

    @property
    @abstractmethod
    def main(self) -> Channel:
        """
        Shell 自身的主轨. 主轨同时可以用来注册所有的子轨.
        """
        pass

    # --- interpret --- #
    #
    # @property
    # @abstractmethod
    # def interpreter(self) -> Interpreter:
    #     pass

    # --- runtime --- #

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def is_idle(self) -> bool:
        pass

    @abstractmethod
    def clear(self, *chans: str) -> Self:
        pass

    @abstractmethod
    def defer_clear(self, *chans: str) -> Self:
        pass

    @abstractmethod
    def reset(self, *chans: str) -> None:
        pass

    # --- lifecycle --- #

    @abstractmethod
    async def runtime(self) -> ShellRuntime:
        pass

    def __enter__(self):
        self.bootstrap()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
        return None

    @abstractmethod
    def bootstrap(self) -> None:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass
