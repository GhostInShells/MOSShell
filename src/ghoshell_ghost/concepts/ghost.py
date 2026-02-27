from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import TypeVar, Generic, TYPE_CHECKING
from dataclasses import dataclass
from ghoshell_moss import Channel, MOSSShell
from ghoshell_container import IoCContainer

from .ghost_state import GhostState

if TYPE_CHECKING:
    from .runtime import GhostRuntime


class GhostConfig(BaseModel, ABC):
    name: str = Field()
    description: str = Field()


GHOST_CONFIG = TypeVar("GHOST_CONFIG", bound=GhostConfig)


@dataclass
class GhostStateNode:
    state: GhostState
    edges: list[str]


class Ghost(Generic[GHOST_CONFIG], ABC):

    @property
    @abstractmethod
    def config(self) -> GHOST_CONFIG:
        pass

    @abstractmethod
    def default_state(self) -> GhostStateNode:
        pass

    @abstractmethod
    def error_state(self) -> GhostStateNode:
        pass

    @abstractmethod
    def ghost_states(self) -> dict[str, GhostStateNode]:
        pass

    @abstractmethod
    def channels(self) -> dict[str, Channel]:
        pass

    @abstractmethod
    def run(
            self,
            shell: MOSSShell | None = None,
            *,
            session_id: str | None = None,
            container: IoCContainer | None = None,
    ) -> "GhostRuntime":
        pass
