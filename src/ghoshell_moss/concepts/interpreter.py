import threading

from .command import CommandToken, CommandTask, CommandMeta
from abc import ABC, abstractmethod
from typing import Iterable, Callable, Optional


class AsyncInterpreter(ABC):
    uuid: str

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

    @abstractmethod
    def put(self, delta: str) -> None:
        pass

    @abstractmethod
    def cancel(self) -> None:
        pass

    @abstractmethod
    async def wait_until_done(self) -> None:
        pass

    @abstractmethod
    def parsed(self) -> Iterable[CommandToken]:
        pass

    @abstractmethod
    def executed(self) -> Iterable[CommandTask]:
        pass


class Interpreter(ABC):
    uuid: str

    @abstractmethod
    def put(self, delta: str) -> None:
        pass

    @abstractmethod
    def cancel(self) -> None:
        pass

    @abstractmethod
    def wait_until_done(self) -> None:
        pass

    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def stop(self) -> None:
        pass

    def __enter__(self):
        self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
