from abc import ABC, abstractmethod
from typing import List, Iterable, Dict, Literal, Optional
from typing_extensions import Self
from .channel import Channel
from .interpreter import Interpreter, AsyncInterpreter
from .command import BasicCommandTask, Command, CommandTaskSeq
from ghoshell_container import IoCContainer
from contextlib import asynccontextmanager


class TextStream(ABC):
    """
    shell 发送文本的专用模块.
    """
    id: str

    @abstractmethod
    async def buffer(self, text: str) -> None:
        pass

    @abstractmethod
    async def end(self) -> None:
        pass

    @abstractmethod
    async def play(self) -> None:
        """
        设置文本允许输出, 进入输出队列.
        """
        pass

    @abstractmethod
    async def wait_done(self, timeout: float | None = None) -> None:
        """
        等待文本输出结束.
        """
        pass

    # 一个使用的示例.

    async def __aenter__(self) -> Self:
        await self.play()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.end()
        await self.wait_done()


class Controller(ABC):

    @abstractmethod
    def output(self, batch_id: Optional[str] = None, play: bool = True) -> TextStream:
        """
        默认需要支持 output.
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


# @abstractmethod
# class Stream(ABC):
#     @abstractmethod
#     def write(self, chars: bytes | None) -> None:
#         pass
#
#     @abstractmethod
#     def read(self, wait_until_done: bool = False) -> bytes | None:
#         pass

class ShellRuntime(ABC):

    @abstractmethod
    def is_running(self) -> bool:
        pass

    @abstractmethod
    def is_idle(self) -> bool:
        pass

    @abstractmethod
    def interpret(self, kind: Literal['clear', 'defer_clear', 'try'] = "clear") -> AsyncInterpreter:
        pass

    @abstractmethod
    async def append(self, *commands: BasicCommandTask) -> None:
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
    Model-oriented Operating System Shell
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
