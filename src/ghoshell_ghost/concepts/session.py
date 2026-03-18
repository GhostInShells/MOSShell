from abc import ABC, abstractmethod

from ghoshell_common.contracts import LoggerItf, Workspace, Configs
from ghoshell_container import IoCContainer
from .conversation import ConversationStore
from .eventbus import EventBus
from .messenger import Messenger
from .models import Models


class Session(ABC):

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        pass

    @property
    @abstractmethod
    def models(self) -> Models:
        pass

    @property
    @abstractmethod
    def logger(self) -> LoggerItf:
        pass

    @property
    @abstractmethod
    def workspace(self) -> Workspace:
        pass

    @property
    @abstractmethod
    def configs(self) -> Configs:
        pass

    @property
    @abstractmethod
    def conversations(self) -> ConversationStore:
        pass

    @property
    @abstractmethod
    def messenger(self) -> Messenger:
        pass

    @property
    @abstractmethod
    def eventbus(self) -> EventBus:
        pass
