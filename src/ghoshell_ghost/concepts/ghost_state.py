from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from typing_extensions import Self
from ghoshell_moss import Channel
from pydantic import BaseModel
from .session import Session
from .eventbus import Event
import asyncio


class GhostStateKwargs(BaseModel):
    pass


class GhostStateConfig(BaseModel):
    pass


GHOST_STATE_CONFIG = TypeVar('GHOST_STATE_CONFIG', bound=GhostStateConfig)
GHOST_STATE_ARGS = TypeVar('GHOST_STATE_ARGS', bound=GhostStateKwargs)


class RealtimeActions(ABC):

    @abstractmethod
    async def intercept(self, event: Event) -> Event | None:
        pass

    @abstractmethod
    async def run(self) -> Self | None:
        pass

    @abstractmethod
    def __repr__(self):
        pass


class RealtimeActionLoop:

    def __init__(self, actions: RealtimeActions) -> None:
        self.current_action = actions

    async def __aenter__(self) -> Self:
        pass

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    async def update(self, actions: RealtimeActions) -> None:
        pass

    async def _loop(self) -> None:
        action_task = None
        while self.current_action:
            action_task = asyncio.create_task(self.current_action.run())
            new_actions = await action_task
            if new_actions is None:
                break
            self.current_action = new_actions


class GhostState(Generic[GHOST_STATE_CONFIG, GHOST_STATE_ARGS], ABC):

    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def config(self) -> GHOST_STATE_CONFIG:
        pass

    @abstractmethod
    def kwarg_model(self) -> type[GHOST_STATE_ARGS] | None:
        pass

    @abstractmethod
    def channels(self) -> dict[str, Channel]:
        pass

    @abstractmethod
    def default_actions(self) -> RealtimeActions:
        pass

    @abstractmethod
    async def on_event(self, event: Event, session: Session) -> RealtimeActions | None:
        pass
