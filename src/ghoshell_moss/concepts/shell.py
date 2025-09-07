from abc import ABC, abstractmethod
from typing import List, Iterable, Dict, Literal
from typing_extensions import Self
from .channel import Channel
from .interpreter import Interpreter, AsyncInterpreter
from .command import CommandTask, Command
from ghoshell_container import IoCContainer
from contextlib import asynccontextmanager


class TextOutput(ABC):

    @abstractmethod
    async def new_batch(self, batch_id: str | None = None, output: bool = False) -> str:
        pass

    @abstractmethod
    async def write(self, batch_id: str, output: str) -> str:
        pass

    @abstractmethod
    async def output(self, batch_id: str) -> None:
        pass

    @abstractmethod
    async def wait_batch_done(self, batch_id: str) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class Controller(ABC):

    @abstractmethod
    async def loop(self, times: int, __text__: str) -> None:
        pass

    @abstractmethod
    async def group(self, __text__: str) -> None:
        pass

    @abstractmethod
    async def clear(self, __text__: str) -> None:
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
    async def append(self, *commands: CommandTask) -> None:
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
