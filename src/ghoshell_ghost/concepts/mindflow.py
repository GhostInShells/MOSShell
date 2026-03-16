from typing import Generic, TypeVar
from ghoshell_ghost.concepts.ghost import Ghost
from abc import ABC, abstractmethod

GHOST = TypeVar('GHOST', bound=Ghost)


class MindNode(Generic[GHOST], ABC):
    """
    并行思考范式的核心设计思路,
    """

    @abstractmethod
    def get_ghost(self) -> GHOST:
        pass


class Mindflow(ABC):
    """
    Mindflow 是一种并行思考拓扑的设计范式.
    """
    pass
