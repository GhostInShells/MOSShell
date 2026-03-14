from typing import Iterable
from ghoshell_ghost.concepts.ghost import EventModel


def get_env_models() -> Iterable[type[EventModel]]:
    raise NotImplementedError("todo")
