from typing_extensions import Self
from abc import ABC, abstractmethod
from .mindflow import Observation, Logos, Mindflow
from .session import Conversation, Session
from .channel import Channel


class Ghost(ABC):

    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def instruction(self) -> str:
        pass

    @abstractmethod
    def channel(self) -> Channel:
        """
        ghost channel
        """
        pass

    @abstractmethod
    def mindflow(self) -> Mindflow:
        """
        mindflow that the ghost holds
        """
        pass

    @abstractmethod
    def articulate(self, observation: Observation) -> Logos:
        """
        articulate the logos from observation
        """
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
