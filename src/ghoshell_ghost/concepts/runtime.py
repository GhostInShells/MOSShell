from abc import ABC, abstractmethod
from typing_extensions import Self
from .ghost import Ghost
from .session import Session
from ghoshell_moss import MOSSShell


class GhostRuntime(ABC):

    @property
    @abstractmethod
    def session(self) -> Session:
        pass

    @property
    @abstractmethod
    def ghost(self) -> Ghost:
        pass

    @property
    @abstractmethod
    def shell(self) -> MOSSShell:
        pass

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
