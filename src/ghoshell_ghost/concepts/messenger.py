from abc import ABC, abstractmethod
from ghoshell_moss.message import Message


class Sender(ABC):
    pass


class Receiver(ABC):
    pass


class Messenger(ABC):

    def sender(self) -> Sender:
        pass

    def receiver(self) -> Receiver:
        pass
