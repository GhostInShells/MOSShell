from typing import Iterable, Optional

from ghoshell_container import IoCContainer

from ghoshell_ghost.concepts.eventbus import EventModel
from ghoshell_ghost.concepts.ghost import Ghost
from ghoshell_moss import Message

_atom_instance: Optional["Atom"] = None
"""进程级单例"""

_atom_container: Optional['IoCContainer'] = None
"""进程级容器"""


class Atom(Ghost):

    @classmethod
    def prototype(cls) -> str:
        pass

    @classmethod
    def version(cls) -> str:
        pass

    def identifier(self) -> str:
        pass

    def description(self, *args, **kwargs) -> str:
        pass

    def init_environment(self, *args, **kwargs) -> None:
        pass

    @classmethod
    def get_env_instance(cls, *args, **kwargs) -> 'Ghost':
        pass

    def event_models(self) -> Iterable[type[EventModel]]:
        from ghoshell_atom.framework.workspace.utils import get_env_models
        yield from get_env_models()

    @property
    def container(self) -> IoCContainer:
        if _atom_container is None:
            raise NotImplementedError("todo")
        return _atom_container

    def default_mode(self) -> "GhostMode":
        pass

    def modes(self) -> dict[str, "GhostMode"]:
        pass

    def error_mode(self) -> "GhostMode":
        pass

    def meta_instructions(self) -> list[Message]:
        pass

    def run(self, session_id: str | None = None, *args, **kwargs) -> "GhostRuntime":
        pass

    def get_running_session(self) -> "Session":
        pass
